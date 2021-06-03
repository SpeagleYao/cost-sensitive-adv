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
from robustbench.utils import load_model


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 512)')
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

def attack(model, img, label, criterion, eps=0.031, iters=10, step=0.007, target_setting=False):
    adv = img.detach()
    adv.requires_grad = True

    for j in range(iters):
        out_adv = model(adv.clone())
        loss = criterion(out_adv, label)
        loss.backward()

        noise = adv.grad
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
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
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
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
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
        data, target = data.cuda(), target.cuda()
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
    print('natural_acc:\t', round(100 - natural_err_total.cpu().numpy()/100, 2), '%')
    print('robust _acc:\t', round(100 - robust_err_total.cpu().numpy()/100 , 2), '%')
    print('===================================================================')
    print(f'target1 is {cs_target1}: {getLabel(cs_target1)}.\ttarget2 is {cs_target2}: {getLabel(cs_target2)}.')
    print('target1 to target2 nat att suc:\t', round(f2t_nat_total/10, 2), '%')
    print('target2 to target1 nat att suc:\t', round(t2f_nat_total/10, 2), '%')
    print('target1 to target2 rob att suc:\t', round(f2t_rob_total/10, 2), '%')
    print('target2 to target1 rob att suc:\t', round(t2f_rob_total/10, 2), '%')    
    # print('target1 to target2 nat:\t', f2t_nat_total)
    # print('target2 to target1 nat:\t', t2f_nat_total)
    # print('target1 to target2 rob:\t', f2t_rob_total)
    # print('target2 to target1 rob:\t', t2f_rob_total)
    print('===================================================================')
    print("")


def main():

    # print('pgd white-box attack')
    # Load a model from the model zoo
    # Standard	                            Standardly trained model	                                                                94.78%	0.00%
    # Gowal2020Uncovering_28_10_extra	    Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples	    89.48%	62.76%
    # Rebuffi2021Fixing_28_10_cutmix_ddpm	Fixing Data Augmentation to Improve Adversarial Robustness	                                87.33%	60.73% (synthetic)
    # Wu2020Adversarial_extra   	        Adversarial Weight Perturbation Helps Robust Generalization	                                88.25%	60.04%
    # Zhang2020Geometry 	                Geometry-aware Instance-reweighted Adversarial Training	                                    89.36%	59.64%
    # Carmon2019Unlabeled	                Unlabeled Data Improves Adversarial Robustness	                                            89.69%	59.53%
    # Sehwag2020Hydra	                    HYDRA: Pruning Adversarially Robust Neural Networks	                                        88.98%	57.14%
    # Wang2020Improving	                    Improving Adversarial Robustness Requires Revisiting Misclassified Examples	                87.50%	56.29%
    # Hendrycks2019Using	                Using Pre-Training Can Improve Model Robustness and Uncertainty	                            87.11%	54.92%	
    model_name_set = ['Standard',
                        'Gowal2020Uncovering_28_10_extra', 
                        'Rebuffi2021Fixing_28_10_cutmix_ddpm', 
                        'Wu2020Adversarial_extra', 
                        'Zhang2020Geometry', 
                        'Carmon2019Unlabeled',
                        'Sehwag2020Hydra',
                        'Wang2020Improving',
                        'Hendrycks2019Using'
                    ]
    # model_name = 'Rebuffi2021Fixing_28_10_cutmix_ddpm'
    for i in range(len(model_name_set)):
        model_name = model_name_set[i]
        model = load_model(model_name=model_name,
                        dataset='cifar10',
                        threat_model='Linf')
        # model = nn.DataParallel(model)
        model = model.cuda()
        print(model_name)
        eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    main()
