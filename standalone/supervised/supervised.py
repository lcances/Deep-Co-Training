import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from DCT.util import load_optimizer
from DCT.util import load_preprocesser
from DCT.util import load_callbacks
from DCT.util import load_dataset
from DCT.util import load_model
from DCT.util.checkpoint import CheckPoint, mSummaryWriter
from DCT.util.utils import reset_seed, get_datetime, track_maximum, get_lr, get_train_format
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='../../config/supervised/supervised.yaml')
def run(cfg: DictConfig) -> DictConfig:
    print(OmegaConf.to_yaml(cfg))
    print('current dir: ', os.getcwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processer --------
    train_transform, val_transform = load_preprocesser(cfg.dataset.dataset, "supervised")

    # -------- Get the dataset --------
    _, train_loader, val_loader = load_dataset(
        cfg.dataset.dataset,
        "supervised",

        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,

        train_transform=train_transform,
        val_transform=val_transform,

        num_workers=0, # With the cache enable, it is faster to have only one worker
        pin_memory=True,

        verbose=1
    )

    # The input shape of the data is used to generate the model 
    input_shape = tuple(train_loader.dataset[0][0].shape)

    # -------- Prepare the model --------
    torch.cuda.empty_cache()
    model_func = load_model(cfg.dataset.dataset, cfg.model.model)
    model = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    model = model.cuda()

    summary(model, input_shape)

    # -------- Tensorboard and checkpoint --------
    # -- Prepare suffix
    sufix_title = ''
    sufix_title += f'_{cfg.train_param.learning_rate}-lr'
    sufix_title += f'_{cfg.train_param.supervised_ratio}-sr'
    sufix_title += f'_{cfg.train_param.nb_epoch}-e'
    sufix_title += f'_{cfg.train_param.batch_size}-bs'
    sufix_title += f'_{cfg.train_param.seed}-seed'

    # -------- Tensorboard logging --------
    tensorboard_title = f'{get_datetime()}_{cfg.model.model}_{sufix_title}'
    log_dir = f'{cfg.path.tensorboard_path}/{tensorboard_title}'
    print('Tensorboard log at: ', log_dir)

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=model_func.__name__)

    # -------- Optimizer, callbacks, loss and checkpoint
    optimizer = load_optimizer(cfg.dataset.dataset, "supervised", learning_rate=cfg.train_param.learning_rate, model=model)
    callbacks = load_callbacks(cfg.dataset.dataset, "supervised", optimizer=optimizer, nb_epoch=cfg.train_param.nb_epoch)
    loss_ce = nn.CrossEntropyLoss(reduction="mean")

    checkpoint_title = f'{cfg.model.model}_{sufix_title}'
    checkpoint_path = f'{cfg.path.checkpoint_path}/{checkpoint_title}'
    checkpoint = CheckPoint(model, optimizer, mode="max", name=checkpoint_path)

    # -------- Metrics and print formater --------
    fscore_fn = FScore()
    acc_fn = CategoricalAccuracy()
    avg = ContinueAverage()
    reset_metrics = lambda : [m.reset() for m in [fscore_fn, acc_fn, avg]]
    maximum_tracker = track_maximum()

    header, train_formater, val_formater = get_train_format('supervised')

    # -------- Training and Validation function --------
    def train(epoch):
        start_time = time.time()
        print("")

        reset_metrics()
        model.train()

        for i, (X, y) in enumerate(train_loader):
            X = X.cuda().float()
            y = y.cuda().long()

            logits = model(X)
            loss = loss_ce(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.set_grad_enabled(False):
                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                acc = acc_fn(pred_arg, y).mean(size=100)
                fscore = fscore_fn(pred, y_one_hot).mean(size=100)
                avg_ce = avg(loss.item()).mean(size=100)

                # logs
                print(train_formater.format(
                    "Training: ",
                    epoch + 1,
                    int(100 * (i + 1) / len(train_loader)),
                    "", avg_ce,
                    "", acc, fscore,
                    time.time() - start_time
                ), end="\r")

        tensorboard.add_scalar("train/Lce", avg_ce, epoch)
        tensorboard.add_scalar("train/f1", fscore, epoch)
        tensorboard.add_scalar("train/acc", acc, epoch)

    def val(epoch):
        start_time = time.time()
        print("")
        reset_metrics()
        model.eval()

        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(val_loader):
                X = X.cuda().float()
                y = y.cuda().long()

                logits = model(X)
                loss = loss_ce(logits, y)

                # metrics
                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                acc = acc_fn(pred_arg, y).mean(size=100)
                fscore = fscore_fn(pred, y_one_hot).mean(size=100)
                avg_ce = avg(loss.item()).mean(size=100)

                # logs
                print(val_formater.format(
                    "Validation: ",
                    epoch + 1,
                    int(100 * (i + 1) / len(val_loader)),
                    "", avg_ce,
                    "", acc, fscore,
                    time.time() - start_time
                ), end="\r")

        tensorboard.add_scalar("val/Lce", avg_ce, epoch)
        tensorboard.add_scalar("val/f1", fscore, epoch)
        tensorboard.add_scalar("val/acc", acc, epoch)

        tensorboard.add_scalar("hyperparameters/learning_rate", get_lr(optimizer), epoch)

        tensorboard.add_scalar("max/acc", maximum_tracker("acc", acc), epoch)
        tensorboard.add_scalar("max/f1", maximum_tracker("f1", fscore), epoch)

        checkpoint.step(acc)
        for c in callbacks:
            c.step()

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

    # -------- Save the hyper parameters and the metrics --------
    hparams = dict(
        dataset=cfg.dataset.dataset,
        model=cfg.model.model,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        nb_epoch=cfg.train_param.nb_epoch,
        learning_rate=cfg.train_param.learning_rate,
        seed=cfg.train_param.seed,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,
    )

    # convert all value to str
    hparams = dict(zip(hparams.keys(), map(str, hparams.values())))

    final_metrics = {
        "max_acc": maximum_tracker.max["acc"],
        "max_f1": maximum_tracker.max["f1"],
    }

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == '__main__':
    run()
