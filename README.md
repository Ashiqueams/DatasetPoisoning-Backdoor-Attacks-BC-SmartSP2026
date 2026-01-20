# Dataset Poisoning Attacks on Behavioral Cloning Policies #
<p align="center">
  <a href="https:/arxiv.org/abs/2511.20992">View on ArXiv</a> |
  <a href="https://sites.google.com/view/dataset-poisoning-in-bc">Project Website</a>
</p>

***Akansha Kalra, Soumil Datta, Ethan Gilmore, Duc La, Guanhong Tao, Daniel S. Brown***


# Environment setup
Begin by installing our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 22.04 ,
```console
$ conda env create -f BC_attack_env.yml
```
And activate it by 
```console
$ conda activate bc_attack
```
## To create poisoned dataset ##
Run the following command , 
```console
$ python3 poison_data.py
```
Note: The function add_trojan in poison_data.py defines the type of patch-by default it generates data posioned with Gaussian Patch . To generate poisoned data using red patch -uncomment line 31 and comment line 34. 
## To train BC model on poisoned datasets ##
```console
$ python3 train.py
```
## To Evaluate the Backdoor Control Accuracy of the learned BC models ##
```console
$ python3 eval.py
```

## To reproduce our results for Test Time Trigger Attacks ##
```console
$ python3 Test_Time_Trigger_Attacks.py
```
