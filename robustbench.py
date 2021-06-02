import torch
from models import *

# TODO:
model_path = './cp_cifar10/res18_natural.pth'

# from robustbench.utils import load_model
# # Load a model from the model zoo
# model = load_model(model_name='Carmon2019Unlabeled',
#                    dataset='cifar10',
#                    threat_model='Linf')

device = torch.device("cuda:0")
model = PreActResNet18()
model.load_state_dict(torch.load(model_path))

# Evaluate the Linf robustness of the model using AutoAttack
from robustbench.eval import benchmark
clean_acc, robust_acc = benchmark(model,
                                  dataset='cifar10',
                                  n_examples=1000,
                                  threat_model='Linf',
                                  eps=8/255,
                                  device=device)
print(clean_acc, robust_acc)

# import torch

# from robustbench import benchmark
# from myrobust model import MyRobustModel

# threat_model = "Linf"  # One of {"Linf", "L2", "corruptions"}
# dataset = "cifar10"  # For the moment "cifar10" only is supported

# model = MyRobustModel()
# model_name = "<Name><Year><FirstWordOfTheTitle>"
# device = torch.device("cuda:0")

# clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=1000, dataset=dataset,
#                                   threat_model=threat_model, eps=8/255, device=device, to_disk=True)