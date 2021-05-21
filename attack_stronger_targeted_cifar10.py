from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from models import PreActResNet18
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
# TODO:
parser.add_argument('--model-path',
                    default='./cp_cifar10/res18_normal.pth',
                    help='model for white-box attack evaluation')
parser.add_argument('--target1', type=int, default=3)
parser.add_argument('--target2', type=int, default=5)

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# TODO:
# Specific target setting
cs_target1 = args.target1
cs_target2 = args.target2


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, target_setting = False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            with autocast():
                if target_setting:
                    output = model(X - delta)
                else:
                    output = model(X + delta)
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            if target_setting:
                d = clamp(d - alpha * torch.sign(g), -epsilon, epsilon)
            else:
                d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        if target_setting:
            all_loss = F.cross_entropy(model(X-delta), y, reduction='none').detach()
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def attack(model, img, label, criterion, eps=0.031, iters=10, step=0.007, target_setting=False):
    adv = img.detach()
    adv.requires_grad = True

    for j in range(iters):
        out_adv = model(adv.clone())
        loss = criterion(out_adv, label)
        loss.backward()

        noise = adv.grad
        adv.data = adv.data + step * noise.sign()
        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()

    return adv.detach()


def getLabel(index):
    if index == 0:
        return "airplane"
    if index == 1:
        return "automobile"
    if index == 2:
        return "bird"
    if index == 3:
        return "cat"
    if index == 4:
        return "deer"
    if index == 5:
        return "dog"
    if index == 6:
        return "frog"
    if index == 7:
        return "horse"
    if index == 8:
        return "ship"
    if index == 9:
        return "truck"


# untargeted whitebox attack
def _pgd_whitebox_untargeted(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    X_pgd = attack(model, X_pgd, y, nn.CrossEntropyLoss(), epsilon, num_steps, step_size, False) # untargeted
    out_pgd = model(X_pgd)
    err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()

    return err, err_pgd


# targeted whitebox attack
def _pgd_whitebox_targeted(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    f2t_nat = np.logical_and(out.data.max(1)[1].cpu().numpy() == cs_target1, y.data.cpu().numpy() == cs_target2).sum()
    t2f_nat = np.logical_and(out.data.max(1)[1].cpu().numpy() == cs_target2, y.data.cpu().numpy() == cs_target1).sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
# TODO:
# Pick where y == cs_target & attack it to another target
    X_pgd_cst1 = torch.index_select(X_pgd, 0, (y == cs_target1).nonzero(as_tuple=False).squeeze())
    X_pgd_cst2 = torch.index_select(X_pgd, 0, (y == cs_target2).nonzero(as_tuple=False).squeeze())
    y_cst1 = torch.ones(X_pgd_cst1.shape[0]).long().cuda() * cs_target2
    y_cst2 = torch.ones(X_pgd_cst2.shape[0]).long().cuda() * cs_target1
    X_pgd_f2t = attack(model, X_pgd_cst1, y_cst1, nn.CrossEntropyLoss(), epsilon, num_steps, step_size, True) # targeted
    X_pgd_t2f = attack(model, X_pgd_cst2, y_cst2, nn.CrossEntropyLoss(), epsilon, num_steps, step_size, True) # targeted

    out_pgd_f2t = model(X_pgd_f2t)
    out_pgd_t2f = model(X_pgd_t2f)
    f2t_pgd = (out_pgd_f2t.data.max(1)[1].cpu().numpy() == cs_target2).sum()
    t2f_pgd = (out_pgd_t2f.data.max(1)[1].cpu().numpy() == cs_target1).sum()
    # print(f'f2t:{out_pgd_f2t.data.max(1)[1]}')
    # print(f't2f:{out_pgd_t2f.data.max(1)[1]}')
    # print('err pgd (white-box): ', err_pgd)
    return f2t_nat, t2f_nat, f2t_pgd, t2f_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    t2f_nat_total = 0
    f2t_nat_total = 0
    t2f_rob_total = 0
    f2t_rob_total = 0

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox_untargeted(model, X, y)
        f2t_nat_err, t2f_nat_err, f2t_rob_err, t2f_rob_err= _pgd_whitebox_targeted(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        f2t_nat_total += f2t_nat_err
        t2f_nat_total += t2f_nat_err
        f2t_rob_total += f2t_rob_err
        t2f_rob_total += t2f_rob_err
    print('natural_acc:\t', 100 - natural_err_total.cpu().numpy()/100, '%')
    print('robust _acc:\t', 100 - robust_err_total.cpu().numpy()/100, '%')
    print('=======================================================================')
    print(f'target1 is {cs_target1}: {getLabel(cs_target1)}.\ttarget2 is {cs_target2}: {getLabel(cs_target2)}.')
    # print('target1 to target2 nat:\t', 100 - f2t_nat_total/10, '%')
    # print('target2 to target1 nat:\t', 100 - t2f_nat_total/10, '%')
    # print('target1 to target2 rob:\t', 100 - f2t_rob_total/10, '%')
    # print('target2 to target1 rob:\t', 100 - t2f_rob_total/10, '%')    
    print('target1 to target2 nat:\t', f2t_nat_total)
    print('target2 to target1 nat:\t', t2f_nat_total)
    print('target1 to target2 rob:\t', f2t_rob_total)
    print('target2 to target1 rob:\t', t2f_rob_total)
    print('=======================================================================')


def main():

    # print('pgd white-box attack')
    print(args.model_path)
    model = PreActResNet18().to(device)
    model.load_state_dict(torch.load(args.model_path))

    eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    main()
