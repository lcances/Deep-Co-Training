# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from advertorch.attacks import GradientSignAttack

import numpy as np
import torch.optim as optim
import os
import time
import random
import math
import pickle
import argparse
from random import shuffle
from tqdm import tqdm

import sys
sys.path.append("../src")

from metrics import CategoricalAccuracy, Ratio
from ramps import Warmup, sigmoid_rampup
from losses import loss_cot, loss_diff, loss_sup, p_loss_diff, p_loss_sup
from models import repro

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

# ======================================================================================================================
#           ARGUMENTS
# ======================================================================================================================
parser = argparse.ArgumentParser(description='Deep Co-Training for Semi-Supervised Image Recognition')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--batchsize', '-b', default=100, type=int)
parser.add_argument('--lambda_cot_max', default=10, type=int)
parser.add_argument('--lambda_diff_max', default=0.5, type=float)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--warm_up', default=80.0, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--decay', default=1e-4, type=float)
parser.add_argument('--epsilon', default=0.02, type=float)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--cifar10_dir', default='/corpus/corpus/CIFAR10', type=str)
parser.add_argument('--svhn_dir', default='./data', type=str)
parser.add_argument('--tensorboard_dir', default='tensorboard/', type=str)
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)
parser.add_argument('--base_lr', default=0.05, type=float)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("--job_name", type=str)
args = parser.parse_args()


# for reproducibility
def set_seed(seed=args.seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
set_seed()

import datetime
def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]


if not os.path.isdir(args.tensorboard_dir):
    os.mkdir(args.tensorboard_dir)

start_epoch = 0
end_epoch = args.epochs
class_num = args.num_class 
batch_size = args.batchsize
nb_batch = 50000 / batch_size
best_acc = 0.0

title = "%s_%s_%slcm_%sldm_%swl" % (
    get_datetime(),
    args.job_name,
    args.lambda_cot_max,
    args.lambda_diff_max,
    args.warm_up,
)
tensorboard = SummaryWriter("%s/%s" % (args.tensorboard_dir, title))


U_batch_size = int(batch_size * 46./50.) # note that the ratio of labelled/unlabelled data need to be equal to 4000/46000
S_batch_size = batch_size - U_batch_size


def adjust_learning_rate(optimizer, epoch):
    """cosine scheduling"""
    epoch = epoch + 1
    lr = args.base_lr*(1.0 + math.cos((epoch-1)*math.pi/args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Prepare the dataset
transform_train = transforms.Compose([
    transforms.RandomAffine(0, translate=(1/16,1/16)), # translation at most two pixels
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

testset = torchvision.datasets.CIFAR10(root=args.cifar10_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

trainset = torchvision.datasets.CIFAR10(root=args.cifar10_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

S_idx = []
U_idx = []
dataiter = iter(trainloader)
train = [[] for x in range(args.num_class)]


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
    
    checkpoint = torch.load('./'+ args.checkpoint_dir + '/ckpt.best.' + args.sess + '_' + str(args.seed))
    
    net1 = checkpoint['net1']
    net2 = checkpoint['net2']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['np_state'])
    random.setstate(checkpoint['random_state'])
    
    with open("cifar10_labelled_index.pkl", "rb") as fp:
        S_idx = pickle.load(fp)

    with open("cifar10_unlabelled_index.pkl", "rb") as fp:
        U_idx = pickle.load(fp)
else:

    #Build the model and get the index of S and U
    print('Building model..')
    start_epoch = 0
    net1 = repro()
    net2 = repro()

    for i in range(len(trainset)):
        inputs, labels = dataiter.next()
        train[labels].append(i)

    for i in range(class_num):
        shuffle(train[i])
        S_idx = S_idx + train[i][0:400]
        U_idx = U_idx + train[i][400:]

    #save the indexes in case we need the exact ones to resume
    with open("cifar10_labelled_index.pkl","wb") as fp:
        pickle.dump(S_idx,fp)

    with open("cifar10_unlabelled_index.pkl","wb") as fp:
        pickle.dump(U_idx,fp)


# Prepare training
S_sampler = torch.utils.data.SubsetRandomSampler(S_idx)
U_sampler = torch.utils.data.SubsetRandomSampler(U_idx)

S_loader1 = torch.utils.data.DataLoader(trainset, batch_size=S_batch_size, sampler=S_sampler)
S_loader2 = torch.utils.data.DataLoader(trainset, batch_size=S_batch_size, sampler=S_sampler)
U_loader = torch.utils.data.DataLoader(trainset, batch_size=U_batch_size, sampler=U_sampler)

#net1 adversary object
adversary1 = GradientSignAttack(
net1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon, clip_min=-math.inf, clip_max=math.inf, targeted=False)

#net2 adversary object
adversary2 = GradientSignAttack(
net2, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon, clip_min=-math.inf, clip_max=math.inf, targeted=False)



step = int(len(trainset)/batch_size)


net1.cuda()
net2.cuda()
# net1 = torch.nn.DataParallel(net1)
# net2 = torch.nn.DataParallel(net2)
print('Using', torch.cuda.device_count(), 'GPUs.')


params = list(net1.parameters()) + list(net2.parameters())
optimizer = optim.SGD(params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.decay)

def checkpoint(epoch, option):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net1': net1,
        'net2': net2,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state':torch.cuda.get_rng_state(),
        'np_state': np.random.get_state(), 
        'random_state': random.getstate()
    }
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if(option=='best'):
        torch.save(state, './'+ args.checkpoint_dir +'/ckpt.best.' +
                   args.sess + '_' + str(args.seed))
    else:
        torch.save(state, './'+ args.checkpoint_dir +'/ckpt.last.' +
                   args.sess + '_' + str(args.seed))

# define the metrics
ratioS = [Ratio(), Ratio()]
ratioU = [Ratio(), Ratio()]
ratioSU = [Ratio(), Ratio()]
accS = [CategoricalAccuracy(), CategoricalAccuracy()]
accU = [CategoricalAccuracy(), CategoricalAccuracy()]
accSU = [CategoricalAccuracy(), CategoricalAccuracy()]

# define the warmups
lambda_cot = Warmup(args.lambda_cot_max, args.warm_up, sigmoid_rampup)
lambda_diff = Warmup(args.lambda_diff_max, args.warm_up, sigmoid_rampup)


def reset_all_metrics():
    all_metrics = [*ratioS, *ratioU, *ratioSU, *accS, *accU, *accSU]
    for m in all_metrics:
        m.reset()

def get_current_lr():
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def train(epoch):
    net1.train()
    net2.train()

    adjust_learning_rate(optimizer, epoch)

    # total_S1 = 0
    # total_S2 = 0
    # total_U1 = 0
    # total_U2 = 0
    # train_correct_S1 = 0
    # train_correct_S2 = 0
    # train_correct_U1 = 0
    # train_correct_U2 = 0
    running_loss = 0.0
    ls = 0.0
    lc = 0.0 
    ld = 0.0
    
    start_time = time.time()
    print("")
    
    # create iterator for b1, b2, bu
    S_iter1 = iter(S_loader1)
    S_iter2 = iter(S_loader2)
    U_iter = iter(U_loader)
    
    for i in range(step):
        inputs_S1, y_S1 = S_iter1.next()
        inputs_S2, y_S2 = S_iter2.next()
        inputs_U, y_U = U_iter.next() # note that labels_U will not be used for training.

        inputs_S1, y_S1 = inputs_S1.cuda(), y_S1.cuda()
        inputs_S2, y_S2 = inputs_S2.cuda(), y_S2.cuda()
        inputs_U, y_U = inputs_U.cuda(), y_U.cuda()


        logits_S1 = net1(inputs_S1)
        logits_S2 = net2(inputs_S2)
        logits_U1 = net1(inputs_U)
        logits_U2 = net2(inputs_U)

        _, pred_S1 = torch.max(logits_S1, 1)
        _, pred_S2 = torch.max(logits_S2, 1)

        # pseudo labels of U 
        _, pred_U1 = torch.max(logits_U1, 1)
        _, pred_U2 = torch.max(logits_U2, 1)

        # ======== Generate adversarial examples ========
        # fix batchnorm ----
        net1.eval()
        net2.eval()

        #generate adversarial examples ----
        adv_data_S1 = adversary1.perturb(inputs_S1, y_S1)
        adv_data_U1 = adversary1.perturb(inputs_U, pred_U1)

        adv_data_S2 = adversary2.perturb(inputs_S2, y_S2)
        adv_data_U2 = adversary2.perturb(inputs_U, pred_U2)

        net1.train()
        net2.train()

        # predict adversarial examples ----
        adv_logits_S1 = net1(adv_data_S2)
        adv_logits_S2 = net2(adv_data_S1)

        adv_logits_U1 = net1(adv_data_U2)
        adv_logits_U2 = net2(adv_data_U1)

        # ======== calculate the differents loss ========
        # zero the parameter gradients ----
        optimizer.zero_grad()
        net1.zero_grad()
        net2.zero_grad()

        # losses ----
        Loss_sup_S1, Loss_sup_S2, Loss_sup = p_loss_sup(logits_S1, logits_S2, y_S1, y_S2)
        Loss_cot = loss_cot(logits_U1, logits_U2)
        pld_S, pld_U, Loss_diff = p_loss_diff(logits_S1, logits_S2, adv_logits_S1, adv_logits_S2, logits_U1, logits_U2, adv_logits_U1, adv_logits_U2)
        
        total_loss = Loss_sup + lambda_cot.next() * Loss_cot + lambda_diff.next() * Loss_diff
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        # accuracies ----
        pred_SU1 = torch.cat((pred_S1, pred_U1), 0)
        pred_SU2 = torch.cat((pred_S2, pred_U2), 0)
        y_SU1 = torch.cat((y_S1, y_U), 0)
        y_SU2 = torch.cat((y_S2, y_U), 0)

        acc_S1 = accS[0](pred_S1, y_S1)
        acc_S2 = accS[1](pred_S2, y_S2)
        acc_U1 = accU[0](pred_U1, y_U)
        acc_U2 = accU[1](pred_U2, y_U)
        acc_SU1 = accSU[0](pred_SU1, y_SU1)
        acc_SU2 = accSU[1](pred_SU2, y_SU2)
        
        # ratios  ----
        _, adv_pred_S1 = torch.max(adv_logits_S1, 1)
        _, adv_pred_S2 = torch.max(adv_logits_S2, 1)
        _, adv_pred_U1 = torch.max(adv_logits_U1, 1)
        _, adv_pred_U2 = torch.max(adv_logits_U2, 1)

        adv_pred_SU1 = torch.cat((adv_pred_S1, adv_pred_U1), 0)
        adv_pred_SU2 = torch.cat((adv_pred_S2, adv_pred_U2), 0)
        adv_y_SU1 = torch.cat((y_S1, pred_U1), 0)
        adv_y_SU2 = torch.cat((y_S2, pred_U2), 0)

        ratio_S1 = ratioS[0](adv_pred_S1, y_S1)
        ratio_S2 = ratioS[1](adv_pred_S2, y_S2)
        ratio_U1 = ratioU[0](adv_pred_U1, pred_U1)
        ratio_U2 = ratioU[1](adv_pred_U2, pred_U2)
        ratio_SU1 = ratioSU[0](adv_pred_SU1, adv_y_SU1)
        ratio_SU2 = ratioSU[1](adv_pred_SU2, adv_y_SU2)
        # ========

        # train_correct_S1 += np.sum(pred_S1.cpu().numpy() == y_S1.cpu().numpy())
        # total_S1 += y_S1.size(0)

        # train_correct_U1 += np.sum(pred_U1.cpu().numpy() == labels_U.cpu().numpy())
        # total_U1 += labels_U.size(0)

        # train_correct_S2 += np.sum(pred_S2.cpu().numpy() == y_S2.cpu().numpy())
        # total_S2 += y_S2.size(0)

        # train_correct_U2 += np.sum(pred_U2.cpu().numpy() == labels_U.cpu().numpy())
        # total_U2 += labels_U.size(0)
        #
        running_loss += total_loss.item()
        ls += Loss_sup.item()
        lc += Loss_cot.item()
        ld += Loss_diff.item()
        
        # print statistics
        print("Epoch %s: %.2f%% : train acc: %.3f %.3f - Loss: %.3f %.3f %.3f %.3f - time: %.2f" % (
            epoch, (i / nb_batch) * 100,
            acc_SU1, acc_SU2,
            running_loss/(i+1), ls/(i+1), lc/(i+1), ld/(i+1),
            time.time() - start_time,
        ), end="\r")

    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', total_loss.item(), epoch)
    tensorboard.add_scalar('train/Lsup', Loss_sup.item(), epoch )
    tensorboard.add_scalar('train/Lcot', Loss_cot.item(), epoch )
    tensorboard.add_scalar('train/Ldiff', Loss_diff.item(), epoch )
    tensorboard.add_scalar("train/acc_1", acc_SU1, epoch )
    tensorboard.add_scalar("train/acc_2", acc_SU2, epoch )

    tensorboard.add_scalar("detail_loss/Lsus S1", Loss_sup_S1.item(), epoch)
    tensorboard.add_scalar("detail_loss/Lsus S2", Loss_sup_S2.item(), epoch)
    tensorboard.add_scalar("detail_loss/Ldiff S", pld_S.item(), epoch)
    tensorboard.add_scalar("detail_loss/Ldiff U", pld_U.item(), epoch)

    tensorboard.add_scalar("detail_acc/acc S1", acc_S1, epoch)
    tensorboard.add_scalar("detail_acc/acc S2", acc_S2, epoch)
    tensorboard.add_scalar("detail_acc/acc U1", acc_U1, epoch)
    tensorboard.add_scalar("detail_acc/acc U2", acc_U2, epoch)

    tensorboard.add_scalar("detail_ratio/ratio S1", ratio_S1, epoch)
    tensorboard.add_scalar("detail_ratio/ratio S2", ratio_S2, epoch)
    tensorboard.add_scalar("detail_ratio/ratio U1", ratio_U1, epoch)
    tensorboard.add_scalar("detail_ratio/ratio U2", ratio_U2, epoch)
    tensorboard.add_scalar("detail_ratio/ratio SU1", ratio_SU1, epoch)
    tensorboard.add_scalar("detail_ratio/ratio SU2", ratio_SU2, epoch)

def test(epoch):
    global best_acc
    net1.eval()
    net2.eval()
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs1 = net1(inputs)
            predicted1 = outputs1.max(1)
            total1 += targets.size(0)
            correct1 += predicted1[1].eq(targets).sum().item()

            outputs2 = net2(inputs)
            predicted2 = outputs2.max(1)
            total2 += targets.size(0)
            correct2 += predicted2[1].eq(targets).sum().item()

    print('\nnet1 test acc: %.3f%% (%d/%d) | net2 test acc: %.3f%% (%d/%d)'
        % (100.*correct1/total1, correct1, total1, 100.*correct2/total2, correct2, total2))
    
    tensorboard.add_scalar("val/acc 1", correct1 / total1, epoch)
    tensorboard.add_scalar("val/acc 2", correct2 / total2, epoch)

    tensorboard.add_scalar("detail_hyperparameters/lambda cot", lambda_cot, epoch)
    tensorboard.add_scalar("detail_hyperparameters/lambda diff", lambda_diff, epoch)
    tensorboard.add_scalar("detail_hyperparameters/lr", get_current_lr(), epoch)

    acc = ((100.*correct1/total1)+(100.*correct2/total2))/2
    if acc > best_acc:
        best_acc = acc
        checkpoint(epoch, 'best')

for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)
    checkpoint(epoch, 'last')

tensorboard.export_scalars_to_json('./' + args.tensorboard_dir + 'output.json')
tensorboard.close()
