# Cost-sensitive Adversarial Learning
Combination of cost-sensitive learning and adversarial learning

### News
Ready to do some fundamental experiments on basic settings!

Doing experiments to see how one specific attack from one kind to another is influenced.

### Settings
- Normal settings: normal adversarial training
- Specific settings: only generate adversarial examples between given classes
- Test: check accuracy of natural, adversarial, and specific attack results

### Expeiments
- Do experiments between two specific classes, e.g. frog-truck.
- Do experiments between two types of classes, e.g. animal-vehicle.

### What I've done
In the first experiments, I now have trained 5 models as follows:
- res18_natural: Train the model with natural data
- res18_normal : Train the model with natural data + all untargeted pgd data
- res18_trades : Train the model with TRADES loss
- res18_alltar : Train the model with natural data + all targeted pgd data
- res18_sp_35  : Train the model with natural data + all targeted pgd data between class 3&5
- res18_ow_35  : Train the model with natural data + all targeted pgd data from class 3 to class 5

All these models are tested under natural testset, untargeted pgd testset, and targeted pgd testset between two specific classes.

Then, I ran test under several SOTA models, and output their test results under targeted attack and confusion matrixes.

Targeted attack results can be seen in [here](tresult_robench.txt).

Confusuin matrixes can be seen in [here](cfs_robench_result.txt).
<!-- Results can be seen in [here](result.txt). -->