#!/usr/bin/env bash

# - airplane : 0
# - automobile : 1
# - bird : 2
# - cat : 3
# - deer : 4
# - dog : 5
# - frog : 6
# - horse : 7
# - ship : 8
# - truck : 9

python confusion_matrix.py --model-path './cp_cifar10/res18_natural.pth' >> cfs_result.txt
python confusion_matrix.py --model-path './cp_cifar10/res18_normal.pth'  >> cfs_result.txt
python confusion_matrix.py --model-path './cp_cifar10/res18_trades.pth'  >> cfs_result.txt
python confusion_matrix.py --model-path './cp_cifar10/res18_alltar.pth'  >> cfs_result.txt
python confusion_matrix.py --model-path './cp_cifar10/res18_sp_35.pth'   >> cfs_result.txt
python confusion_matrix.py --model-path './cp_cifar10/res18_sp_53.pth'   >> cfs_result.txt
python confusion_matrix.py --model-path './cp_cifar10/res18_ow_35.pth'   >> cfs_result.txt
python confusion_matrix.py --model-path './cp_cifar10/res18_ow_53.pth'   >> cfs_result.txt
