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

for cst1 in `echo 0`
do
    for cst2 in `echo 2`
    do
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_natural.pth' --target1 ${cst1} --target2 ${cst2} >> tresult_02.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_normal.pth'  --target1 ${cst1} --target2 ${cst2} >> tresult_02.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_trades.pth'  --target1 ${cst1} --target2 ${cst2} >> tresult_02.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_alltar.pth'  --target1 ${cst1} --target2 ${cst2} >> tresult_02.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_sp_02.pth'   --target1 ${cst1} --target2 ${cst2} >> tresult_02.txt
        # python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_sp_53.pth'   --target1 ${cst1} --target2 ${cst2} >> test_result.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_ow_02.pth'   --target1 ${cst1} --target2 ${cst2} >> tresult_02.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_ow_20.pth'   --target1 ${cst1} --target2 ${cst2} >> tresult_02.txt
    done
done
