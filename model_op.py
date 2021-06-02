import torch
from models import *
from torchvision.models import resnet50


# 可替换为自己的模型及输入
model = WRN_70_16()
input = torch.randn(1, 3, 32, 32)
# flops, params = profile(model, inputs=(input, ))
# flops, params = clever_format([flops, params], "%.3f")
print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

# PreActResNet18 have 11,172,170(11.2M) paramerters in total
# ResNet50       have 25,557,032(25.6M) paramerters in total
# WRN-28-10      have 36,479,194(36.5M) paramerters in total
# WRN-76-16      have 266,796,506(266.8M) paramerters in total