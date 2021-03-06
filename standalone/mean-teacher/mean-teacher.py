import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
from torchsummary import summary
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from DCT.ramps import Warmup, sigmoid_rampup

from DCT.util.utils import reset_seed, get_datetime, track_maximum, get_lr, get_train_format, dotdict
from DCT.util.checkpoint import CheckPoint, mSummaryWriter
from DCT.util import load_optimizer
from DCT.util import load_preprocesser
from DCT.util import load_callbacks
from DCT.util import load_dataset
from DCT.util import load_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import torch
import numpy
import time
import sys
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='../../config/mean-teacher/mean-teacher.yaml')
def run(cfg: DictConfig) -> DictConfig:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processor --------
    train_transform, val_transform = load_preprocesser(cfg.dataset.dataset, "mean-teacher")

    # -------- Get the dataset ---------
    manager, train_loader, val_loader = load_dataset(
        cfg.dataset.dataset,
        "mean-teacher",

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

    # The input shape is needed to generate the model
    input_shape = tuple(train_loader._iterables[0].dataset[0][0].shape)

    # -------- Prepare the models --------
    torch.cuda.empty_cache()

    model_func = load_model(cfg.dataset.dataset, cfg.model.model)

    student = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    teacher = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)

    student = student.cuda()
    teacher = teacher.cuda()

    # We do not need gradient for the teacher model
    for p in teacher.parameters():
        p.detach()

    summary(student, input_shape)
    # -------- TEnsorboard and Checkpoint title --------
    # Prepare sufix
    sufix_title = ''
    sufix_title += f'_{cfg.train_param.learning_rate}-lr'
    sufix_title += f'_{cfg.train_param.supervised_ratio}-sr'
    sufix_title += f'_{cfg.train_param.nb_epoch}-e'
    sufix_title += f'_{cfg.train_param.batch_size}-bs'
    sufix_title += f'_{cfg.train_param.seed}-seed'

    # mean teacher parameters
    sufix_title = f'_{cfg.mt.alpha}-emaa'
    sufix_title += f'_{cfg.mt.warmup_length}-wl'
    sufix_title += f'_{cfg.mt.lambda_ccost_max}-lccm'

    # ccost function and method
    if cfg.mt.ccost_use_softmax: sufix_title += "cc-SOFTMAX"

    # -------- Tensorboard logging --------
    tensorboard_title = f'{get_datetime()}_{cfg.model.model}_{sufix_title}'
    log_dir = f'{cfg.path.tensorboard_path}/{tensorboard_title}'
    print('Tensorboard log at: ', log_dir)

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=cfg.model.model)

    # -------- Optimizer, callbacks, loss and checkpoint --------
    optimizer = load_optimizer(cfg.dataset.dataset, "mean-teacher", student=student, teacher=teacher, learning_rate=cfg.train_param.learning_rate)
    callbacks = load_callbacks(cfg.dataset.dataset, "mean-teacher", optimizer=optimizer, nb_epoch=cfg.train_param.nb_epoch)

    # losses
    loss_ce = nn.CrossEntropyLoss(reduction="mean")  # Supervised loss
    consistency_cost = nn.MSELoss(reduction="mean")  # Unsupervised loss

    # warmup
    lambda_cost = Warmup(cfg.mt.lambda_ccost_max, cfg.mt.warmup_length, sigmoid_rampup)
    callbacks += [lambda_cost]

    # Checkpoint
    checkpoint_title = f'{cfg.model.model}_{sufix_title}'
    checkpoint_path = f'{cfg.path.checkpoint_path}/{checkpoint_title}'
    checkpoint = CheckPoint(student, optimizer, mode="max", name=checkpoint_path)

    # -------- Metric and print formater ----------
    def metrics_calculator():
        def c(logits, y):
            with torch.no_grad():
                y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                pred = torch.softmax(logits, dim=1)
                arg = torch.argmax(logits, dim=1)

                acc = c.fn.acc(arg, y).mean(size=100)
                f1 = c.fn.f1(pred, y_one_hot).mean(size=100)

                return acc, f1,

        c.fn = dotdict(
            acc=CategoricalAccuracy(),
            f1=FScore(),
        )

        return c

    calc_student_s_metrics = metrics_calculator()
    calc_student_u_metrics = metrics_calculator()
    calc_teacher_s_metrics = metrics_calculator()
    calc_teacher_u_metrics = metrics_calculator()

    avg_Sce = ContinueAverage()
    avg_Tce = ContinueAverage()
    avg_ccost = ContinueAverage()

    def reset_metrics():
        for d in [calc_student_s_metrics.fn, calc_student_u_metrics.fn, calc_teacher_s_metrics.fn, calc_teacher_u_metrics.fn]:
            for fn in d.values():
                fn.reset()

    maximum_tracker = track_maximum()

    header, train_formater, val_formater = get_train_format('mean-teacher')

    # --------- Training and Validation function --------
    softmax_fn = nn.Softmax(dim=1)
    noise_fn = transforms.Lambda(lambda x: x + (torch.rand(x.shape).cuda() * 10 + 10))

    def update_teacher_model(student_model, teacher_model, alpha, epoch):

        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (epoch + 1), alpha)

        for param, ema_param in zip(student_model.parameters(), teacher_model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data,  alpha=1-alpha)

    def train(epoch):
        start_time = time.time()
        print("")

        nb_batch = len(train_loader)

        reset_metrics()
        student.train()

        for i, (S, U) in enumerate(train_loader):
            x_s, y_s = S
            x_u, y_u = U

            x_s, x_u = x_s.cuda().float(), x_u.cuda().float()
            y_s, y_u = y_s.cuda().long(), y_u.cuda().long()

            # Predictions
            student_s_logits = student(x_s)
            student_u_logits = student(x_u)
            teacher_s_logits = teacher(noise_fn(x_s))
            teacher_u_logits = teacher(noise_fn(x_u))

            # Calculate supervised loss (only student on S)
            loss = loss_ce(student_s_logits, y_s)

            # Calculate consistency cost (mse(student(x), teacher(x))) x is S + U
            student_logits = torch.cat((student_s_logits, student_u_logits), dim=0)
            teacher_logits = torch.cat((teacher_s_logits, teacher_u_logits), dim=0)
            ccost = consistency_cost(softmax_fn(student_logits), softmax_fn(teacher_logits))

            total_loss = loss + lambda_cost() * ccost

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.set_grad_enabled(False):
                # Teacher prediction (for metrics purpose)
                _teacher_loss = loss_ce(teacher_s_logits, y_s)

                # Update teacher
                update_teacher_model(student, teacher, cfg.mt.alpha, epoch*nb_batch + i)

                # Compute the metrics for the student
                student_s_metrics = calc_student_s_metrics(student_s_logits, y_s)
                student_u_metrics = calc_student_u_metrics(student_u_logits, y_u)
                student_s_acc, student_u_acc, student_s_f1, student_u_f1 = * \
                    student_s_metrics, *student_u_metrics

                # Compute the metrics for the teacher
                teacher_s_metrics = calc_teacher_s_metrics(teacher_s_logits, y_s)
                teacher_u_metrics = calc_teacher_u_metrics(teacher_u_logits, y_u)
                teacher_s_acc, teacher_u_acc, teacher_s_f1, teacher_u_f1 = * \
                    teacher_s_metrics, *teacher_u_metrics

                # Running average of the two losses
                student_running_loss = avg_Sce(loss.item()).mean(size=100)
                teacher_running_loss = avg_Tce(_teacher_loss.item()).mean(size=100)
                running_ccost = avg_ccost(ccost.item()).mean(size=100)

                # logs
                print(train_formater.format(
                    "Training: ", epoch + 1, int(100 * (i + 1) / nb_batch),
                    "", student_running_loss, running_ccost, *
                    student_s_metrics, *student_u_metrics,
                    "", teacher_running_loss, *teacher_s_metrics, *teacher_u_metrics,
                    time.time() - start_time
                ), end="\r")

        tensorboard.add_scalar("train/student_acc_s", student_s_acc, epoch)
        tensorboard.add_scalar("train/student_acc_u", student_u_acc, epoch)
        tensorboard.add_scalar("train/student_f1_s", student_s_f1, epoch)
        tensorboard.add_scalar("train/student_f1_u", student_u_f1, epoch)

        tensorboard.add_scalar("train/teacher_acc_s", teacher_s_acc, epoch)
        tensorboard.add_scalar("train/teacher_acc_u", teacher_u_acc, epoch)
        tensorboard.add_scalar("train/teacher_f1_s", teacher_s_f1, epoch)
        tensorboard.add_scalar("train/teacher_f1_u", teacher_u_f1, epoch)

        tensorboard.add_scalar("train/student_loss", student_running_loss, epoch)
        tensorboard.add_scalar("train/teacher_loss", teacher_running_loss, epoch)
        tensorboard.add_scalar("train/consistency_cost", running_ccost, epoch)


    # In[43]:


    def val(epoch):
        start_time = time.time()
        print("")
        reset_metrics()
        student.eval()

        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(val_loader):
                X = X.cuda().float()
                y = y.cuda().long()

                # Predictions
                student_logits = student(X)
                teacher_logits = teacher(X)

                # Calculate supervised loss (only student on S)
                loss = loss_ce(student_logits, y)
                _teacher_loss = loss_ce(teacher_logits, y)  # for metrics only
                ccost = consistency_cost(softmax_fn(
                    student_logits), softmax_fn(teacher_logits))

                # Compute the metrics
                y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                # ---- student ----
                student_metrics = calc_student_s_metrics(student_logits, y)
                student_acc, student_f1 = student_metrics

                # ---- teacher ----
                teacher_metrics = calc_teacher_s_metrics(teacher_logits, y)
                teacher_acc, teacher_f1 = teacher_metrics

                # Running average of the two losses
                student_running_loss = avg_Sce(loss.item()).mean(size=100)
                teacher_running_loss = avg_Tce(_teacher_loss.item()).mean(size=100)
                running_ccost = avg_ccost(ccost.item()).mean(size=100)

                # logs
                print(val_formater.format(
                    "Validation: ", epoch +
                    1, int(100 * (i + 1) / len(val_loader)),
                    "", student_running_loss, running_ccost, *student_metrics, 0.0, 0.0,
                    "", teacher_running_loss, *teacher_metrics, 0.0, 0.0,
                    time.time() - start_time
                ), end="\r")

        tensorboard.add_scalar("val/student_acc", student_acc, epoch)
        tensorboard.add_scalar("val/student_f1", student_f1, epoch)
        tensorboard.add_scalar("val/teacher_acc", teacher_acc, epoch)
        tensorboard.add_scalar("val/teacher_f1", teacher_f1, epoch)
        tensorboard.add_scalar("val/student_loss", student_running_loss, epoch)
        tensorboard.add_scalar("val/teacher_loss", teacher_running_loss, epoch)
        tensorboard.add_scalar("val/consistency_cost", running_ccost, epoch)

        tensorboard.add_scalar(
            "hyperparameters/learning_rate", get_lr(optimizer), epoch)
        tensorboard.add_scalar(
            "hyperparameters/lambda_ccost_max", lambda_cost(), epoch)

        tensorboard.add_scalar(
            "max/student_acc", maximum_tracker("student_acc", student_acc), epoch)
        tensorboard.add_scalar(
            "max/teacher_acc", maximum_tracker("teacher_acc", teacher_acc), epoch)
        tensorboard.add_scalar(
            "max/student_f1", maximum_tracker("student_f1", student_f1), epoch)
        tensorboard.add_scalar(
            "max/teacher_f1", maximum_tracker("teacher_f1", teacher_f1), epoch)

        checkpoint.step(teacher_acc)
        for c in callbacks:
            c.step()

    # --------- Training loop --------
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
        'alpha': cfg.mt.alpha,
        'warmup_length': cfg.mt.warmup_length,
        'lambda_ccost_max': cfg.mt.lambda_ccost_max,
        'ccost_use_softmax': cfg.mt.ccost_use_softmax,
    }

    # convert all value to str
    hparams = dict(zip(hparams.keys(), map(str, hparams.values())))

    final_metrics = {
        "max_student_acc": maximum_tracker.max["student_acc"],
        "max_teacher_acc": maximum_tracker.max["teacher_acc"],
        "max_student_f1": maximum_tracker.max["student_f1"],
        "max_teacher_f1": maximum_tracker.max["teacher_f1"],
    }

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == '__main__':
    run()
