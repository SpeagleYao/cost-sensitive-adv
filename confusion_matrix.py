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
                    default='./cp_cifar10/res18_natural.pth',
                    help='model for white-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
cfs_mat_nat = np.zeros((10, 10))
cfs_mat_rob = np.zeros((10, 10))


def cfs_print(cfs):
    print("\t", end="")
    for i in range(10):
        print("\t  %2d" % i, end="")
    print("\n    --------------------------------------------------------------------------------")
    for i in range(10):
        print("%2d |" % i, end="")
        for j in range(10):
            print("\t%4d" % cfs[i][j], end="")
        print("")


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    global cfs_mat_nat
    global cfs_mat_rob
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    for i in range(len(X)):
        cfs_mat_nat[y.data[i]][out.data.max(1)[1][i]]+=1
    
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    out_pgd = model(X_pgd)
    for i in range(len(X)):
        cfs_mat_rob[y.data[i]][out_pgd.data.max(1)[1][i]]+=1
    err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()

    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('nat_acc:\t', round(100 - natural_err_total.cpu().numpy()/100, 2), '%')
    print('rob_acc:\t', round(100 - robust_err_total.cpu().numpy()/100 , 2), '%')
    print('====================================================================================')
    print("Natural confusion matrix:")
    cfs_print(cfs_mat_nat)
    print('====================================================================================')
    print("Robust  confusion matrix:")
    cfs_print(cfs_mat_rob)
    print('====================================================================================')
    print("")    

def main():

    # print('pgd white-box attack')
    print(args.model_path)
    model = PreActResNet18().to(device)
    model.load_state_dict(torch.load(args.model_path))

    eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    main()
