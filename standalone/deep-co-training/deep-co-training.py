import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
from DCT.losses import loss_cot, loss_diff, loss_sup
from DCT.ramps import Warmup, sigmoid_rampup
from DCT.util.utils import reset_seed, get_datetime, track_maximum, get_lr, get_train_format, dotdict
from DCT.util.checkpoint import CheckPoint, mSummaryWriter
from DCT.util import load_optimizer
from DCT.util import load_preprocesser
from DCT.util import load_callbacks
from DCT.util import load_dataset
from DCT.util import load_model
from torchsummary import summary
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage, Ratio
from advertorch.attacks import GradientSignAttack
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import torch
import numpy as np
import time
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name='../../config/deep-co-training/deep-co-training.yaml')
def run(cfg: DictConfig) -> DictConfig:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processor --------
    train_transform, val_transform = load_preprocesser(cfg.dataset.dataset, "dct")

    # -------- Get the dataset --------
    manager, train_loader, val_loader = load_dataset(
        cfg.dataset.dataset,
        "dct",

        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,

        train_transform=train_transform,
        val_transform=val_transform,

        num_workers=0,
        pin_memory=True,

        verbose=1
    )

    # The input shape needed to generate the model
    input_shape = train_loader._iterables[0].dataset[0][0].shape

    # -------- Prepare the models ---------
    torch.cuda.empty_cache()

    model_func = load_model(cfg.dataset.dataset, cfg.model.model)
    commun_args = dict(
        manager=manager,
        num_classes=cfg.dataset.num_classes,
        input_shape=list(input_shape),
    )

    m1, m2 = model_func(**commun_args), model_func(**commun_args)
    m1, m2 = m1.cuda(), m2.cuda()

    summary(m1, input_shape)

    # --------- Tensorboard and Checkpoint title ---------
    # Prepare sufix
    sufix_title = ''
    sufix_title += f'_{cfg.train_param.learning_rate}-lr'
    sufix_title += f'_{cfg.train_param.supervised_ratio}-sr'
    sufix_title += f'_{cfg.train_param.nb_epoch}-e'
    sufix_title += f'_{cfg.train_param.batch_size}-bs'
    sufix_title += f'_{cfg.train_param.seed}-seed'

    # deep-co-training parameters
    sufix_title = f'_{cfg.dct.epsilon}-eps'
    sufix_title += f'_{cfg.dct.warmup_length}-wl'
    sufix_title += f'_{cfg.dct.lambda_cot_max}-lcm'
    sufix_title += f'_{cfg.dct.lambda_diff_max}-ldm'

    # -------- Tensorboard logging --------
    tensorboard_title = f'{get_datetime()}_{cfg.model.model}_{sufix_title}'
    log_dir = f'{cfg.path.tensorboard_path}/{tensorboard_title}'
    print('Tensorboard log at: ', log_dir)

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=cfg.model.model)

    # --------- Optimizer, callbacks, loss and checkpoint ---------
    optimizer = load_optimizer(cfg.dataset.dataset, "dct",
            model1=m1, model2=m2, learning_rate=cfg.train_param.learning_rate)
    callbacks = load_callbacks(cfg.dataset.dataset, "dct",
            optimizer=optimizer, nb_epoch=cfg.train_param.nb_epoch)

    # Losses
    # All losses are define in DCT/loss.py

    # warmup
    lambda_cot = Warmup(cfg.dct.lambda_cot_max, cfg.dct.warmup_length, sigmoid_rampup)
    lambda_diff = Warmup(cfg.dct.lambda_diff_max, cfg.dct.warmup_length, sigmoid_rampup)
    callbacks += [lambda_cot, lambda_diff]

    # Checkpoint
    checkpoint_title = f'{cfg.model.model}_{sufix_title}'
    checkpoint_path = f'{cfg.path.checkpoint_path}/{checkpoint_title}'
    checkpoint = CheckPoint([m1, m2], optimizer, mode="max", name=checkpoint_path)

    # --------- Adversarial generation ---------
    adv_generator_1 = GradientSignAttack(
        m1, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=cfg.dct.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
    )

    adv_generator_2 = GradientSignAttack(
        m2, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=cfg.dct.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
    )

    # --------- Metric and print formater ---------
    metrics_fn = dict(
        ratio_s=[Ratio(), Ratio()],
        ratio_u=[Ratio(), Ratio()],
        acc_s=[CategoricalAccuracy(), CategoricalAccuracy()],
        acc_u=[CategoricalAccuracy(), CategoricalAccuracy()],
        f1_s=[FScore(), FScore()],
        f1_u=[FScore(), FScore()],

        avg_total=ContinueAverage(),
        avg_sup=ContinueAverage(),
        avg_cot=ContinueAverage(),
        avg_diff=ContinueAverage(),
    )
    maximum_tracker = track_maximum()

    def reset_metrics():
        for item in metrics_fn.values():
            if isinstance(item, list):
                for f in item:
                    f.reset()
            else:
                item.reset()

    header, train_formater, val_formater = get_train_format('dct')

    # --------- Training and Validation function ---------
    def train(epoch):
        start_time = time.time()
        print("")

        reset_metrics()
        m1.train()
        m2.train()

        for batch, (S1, S2, U) in enumerate(train_loader):
            x_s1, y_s1 = S1
            x_s2, y_s2 = S2
            x_u, y_u = U

            x_s1, x_s2, x_u = x_s1.cuda().float(), x_s2.cuda().float(), x_u.cuda().float()
            y_s1, y_s2, y_u = y_s1.cuda().long(), y_s2.cuda().long(), y_u.cuda().long()

            logits_s1 = m1(x_s1)
            logits_s2 = m2(x_s2)
            logits_u1 = m1(x_u)
            logits_u2 = m2(x_u)

            # pseudo labels of U
            pred_u1 = torch.argmax(logits_u1, 1)
            pred_u2 = torch.argmax(logits_u2, 1)

            # ======== Generate adversarial examples ========
            # fix batchnorm ----
            m1.eval()
            m2.eval()

            # generate adversarial examples ----
            adv_data_s1 = adv_generator_1.perturb(x_s1, y_s1)
            adv_data_u1 = adv_generator_1.perturb(x_u, pred_u1)

            adv_data_s2 = adv_generator_2.perturb(x_s2, y_s2)
            adv_data_u2 = adv_generator_2.perturb(x_u, pred_u2)

            m1.train()
            m2.train()

            # predict adversarial examples ----
            adv_logits_s1 = m1(adv_data_s2)
            adv_logits_s2 = m2(adv_data_s1)

            adv_logits_u1 = m1(adv_data_u2)
            adv_logits_u2 = m2(adv_data_u1)

            # ======== calculate the differents loss ========
            # zero the parameter gradients ----
            optimizer.zero_grad()
            m1.zero_grad()
            m2.zero_grad()

            # losses ----
            l_sup = loss_sup(logits_s1, logits_s2, y_s1, y_s2)

            l_cot = loss_cot(logits_u1, logits_u2)

            l_diff = loss_diff(
                logits_s1, logits_s2, adv_logits_s1, adv_logits_s2,
                logits_u1, logits_u2, adv_logits_u1, adv_logits_u2
            )

            total_loss = l_sup + lambda_cot() * l_cot + lambda_diff() * l_diff
            total_loss.backward()
            optimizer.step()

            # ======== Calc the metrics ========
            with torch.set_grad_enabled(False):
                # accuracies ----
                pred_s1 = torch.argmax(logits_s1, dim=1)
                pred_s2 = torch.argmax(logits_s2, dim=1)

                acc_s1 = metrics_fn["acc_s"][0](pred_s1, y_s1)
                acc_s2 = metrics_fn["acc_s"][1](pred_s2, y_s2)
                acc_u1 = metrics_fn["acc_u"][0](pred_u1, y_u)
                acc_u2 = metrics_fn["acc_u"][1](pred_u2, y_u)

                # ratios  ----
                adv_pred_s1 = torch.argmax(adv_logits_s1, 1)
                adv_pred_s2 = torch.argmax(adv_logits_s2, 1)
                adv_pred_u1 = torch.argmax(adv_logits_u1, 1)
                adv_pred_u2 = torch.argmax(adv_logits_u2, 1)

                ratio_s1 = metrics_fn["ratio_s"][0](adv_pred_s1, y_s1)
                ratio_s2 = metrics_fn["ratio_s"][0](adv_pred_s2, y_s2)
                ratio_u1 = metrics_fn["ratio_s"][0](adv_pred_u1, y_u)
                ratio_u2 = metrics_fn["ratio_s"][0](adv_pred_u2, y_u)
                # ========

                avg_total = metrics_fn["avg_total"](total_loss.item())
                avg_sup = metrics_fn["avg_sup"](l_sup.item())
                avg_diff = metrics_fn["avg_diff"](l_diff.item())
                avg_cot = metrics_fn["avg_cot"](l_cot.item())

                # logs
                print(train_formater.format(
                    "Training: ",
                    epoch + 1,
                    int(100 * (batch + 1) / len(train_loader)),
                    "", avg_sup.mean(size=100), avg_cot.mean(size=100), avg_diff.mean(size=100), avg_total.mean(size=100),
                    "", acc_s1.mean(size=100), acc_u1.mean(size=100),
                    time.time() - start_time
                ), end="\r")

        # using tensorboard to monitor loss and acc\n",
        tensorboard.add_scalar('train/total_loss', avg_total.mean(size=100), epoch)
        tensorboard.add_scalar('train/Lsup', avg_sup.mean(size=100), epoch)
        tensorboard.add_scalar('train/Lcot', avg_cot.mean(size=100), epoch)
        tensorboard.add_scalar('train/Ldiff', avg_diff.mean(size=100), epoch)
        tensorboard.add_scalar("train/acc_1", acc_s1.mean(size=100), epoch)
        tensorboard.add_scalar("train/acc_2", acc_s2.mean(size=100), epoch)

        tensorboard.add_scalar("detail_acc/acc_s1", acc_s1.mean(size=100), epoch)
        tensorboard.add_scalar("detail_acc/acc_s2", acc_s2.mean(size=100), epoch)
        tensorboard.add_scalar("detail_acc/acc_u1", acc_u1.mean(size=100), epoch)
        tensorboard.add_scalar("detail_acc/acc_u2", acc_u2.mean(size=100), epoch)

        tensorboard.add_scalar("detail_ratio/ratio_s1", ratio_s1.mean(size=100), epoch)
        tensorboard.add_scalar("detail_ratio/ratio_s2", ratio_s2.mean(size=100), epoch)
        tensorboard.add_scalar("detail_ratio/ratio_u1", ratio_u1.mean(size=100), epoch)
        tensorboard.add_scalar("detail_ratio/ratio_u2", ratio_u2.mean(size=100), epoch)

        # Return the total loss to check for NaN
        return total_loss.item()


    def val(epoch, msg=""):
        start_time = time.time()
        print("")

        reset_metrics()
        m1.eval()
        m2.eval()

        with torch.set_grad_enabled(False):
            for batch, (X, y) in enumerate(val_loader):
                x = X.cuda().float()
                y = y.cuda().long()

                with autocast():
                    logits_1 = m1(x)
                    logits_2 = m2(x)

                    # losses ----
                    l_sup = loss_sup(logits_1, logits_2, y, y)

                # ======== Calc the metrics ========
                # accuracies ----
                pred_1 = torch.argmax(logits_1, dim=1)
                pred_2 = torch.argmax(logits_2, dim=1)

                acc_1 = metrics_fn["acc_s"][0](pred_1, y)
                acc_2 = metrics_fn["acc_s"][1](pred_2, y)

                avg_sup = metrics_fn["avg_sup"](l_sup.item())

                # logs
                print(val_formater.format(
                    "Validation: ",
                    epoch + 1,
                    int(100 * (batch + 1) / len(train_loader)),
                    "", avg_sup.mean(size=100), 0.0, 0.0, avg_sup.mean(size=100),
                    "", acc_1.mean(size=100), 0.0,
                    time.time() - start_time
                ), end="\r")

        tensorboard.add_scalar("val/acc_1", acc_1.mean(size=100), epoch)
        tensorboard.add_scalar("val/acc_2", acc_2.mean(size=100), epoch)

        tensorboard.add_scalar(
            "max/acc_1", maximum_tracker("acc_1", acc_1.mean(size=100)), epoch)
        tensorboard.add_scalar(
            "max/acc_2", maximum_tracker("acc_2", acc_2.mean(size=100)), epoch)

        tensorboard.add_scalar(
            "detail_hyperparameters/lambda_cot", lambda_cot(), epoch)
        tensorboard.add_scalar(
            "detail_hyperparameters/lambda_diff", lambda_diff(), epoch)
        tensorboard.add_scalar(
            "detail_hyperparameters/learning_rate", get_lr(optimizer), epoch)

        # Apply callbacks
        for c in callbacks:
            c.step()

        # call checkpoint
        checkpoint.step(acc_1.mean(size=100))

    # -------- Training loop --------
    print(header)

    if cfg.train_param.resume:
        checkpoint.load_last()

    start_epoch = checkpoint.epoch_counter
    end_epoch = cfg.train_param.nb_epoch

    for e in range(start_epoch, end_epoch):
        train(e)
        val(e)

        tensorboard.flush()

    # -------- Save the hyper parameter and the metrics 
    hparams = {
        'dataset': cfg.dataset.dataset,
        'model': cfg.model.model,
        'supervised_ratio': cfg.train_param.supervised_ratio,
        'batch_size': cfg.train_param.batch_size,
        'nb_epoch': cfg.train_param.nb_epoch,
        'learning_rate': cfg.train_param.learning_rate,
        'seed': cfg.train_param.seed,
        'train_folds': cfg.train_param.train_folds,
        'val_folds': cfg.train_param.val_folds,
        'epsilon': cfg.dct.epsilon,
        'warmup_length': cfg.dct.warmup_length,
        'lambda_cot_max': cfg.dct.lambda_cot_max,
        'lambda_diff_max': cfg.dct.lambda_diff_max,
    }

    # convert all value to str
    hparams = dict(zip(hparams.keys(), map(str, hparams.values())))

    final_metrics = {
        "max_acc_1": maximum_tracker.max["acc_1"],
        "max_acc_2": maximum_tracker.max["acc_2"],
    }

    tensorboard.add_hparams(hparams, final_metrics)
    tensorboard.flush()
    tensorboard.close()


if __name__ == '__main__':
    run()
