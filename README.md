# FedInverse: Evaluating Privacy Leakage in Federated Learning
This repo contains the Pytorch-based implementation code for our published paper **FedInverse** in The Twelfth International Conference on Learning Representations (**ICLR2024**).

![FedInverse Example.png](https://github.com/Jun-B0518/FedInverse/blob/main/FedInverse%20Framework.png)
## Abstract
Federated Learning (FL) is a distributed machine learning technique where multiple devices (such as smartphones or IoT devices) train a shared global model by using their local data. FL promises better data privacy as the individual data isn't shared with servers or other participants. However, this research uncovers a groundbreaking insight: a model inversion (MI) attacker, who acts as a benign participant, can invert the shared global model and obtain the data belonging to other participants. In such scenarios, distinguishing between attackers and benign participants becomes challenging, leading to severe data leakage risk in FL. In addition, we found that even the most advanced defense approaches could not effectively address this issue. Therefore, it is important to evaluate such data leakage risks of an FL system before using it. Motivated by that, we propose FedInverse to evaluate whether the FL global model can be inverted by MI attackers. In particular, FedInverse can be optimized by leveraging the Hilbert-Schmidt independence criterion (HSIC) as a regularizer to adjust the diversity of the MI attack generator. We test FedInverse with three typical MI attackers, GMI, KED-MI, and VMI. The experiments show that FedInverse can effectively evaluate the data leakage risk that attackers successfully obtain the data belonging to other participants.
## Requirements
This code has been tested on Ubuntu 16.04/20.04, with Python3.7, Pytorch 1.7, OpenCV 4.5.0, and CUDA 11.0/11.4.
## Get Started
### Preparation
- Install required Python packages
- Download relevant datasets including MNIST, CELEBA, etc.
### FL Running
- Run FL without defense
```python
python federated_main.py --model=cnn --net MCNN --dataset=mnist --num_users 100 -R=10 -C='0.1' -B=60 -E=1 --iid=1 --testacc='1.0' --optimizer=sgd --lossfunc=crossentropy --lr='0.01' --measure 'None' --lamdax 0 --lamday 0 --hsic_training False  
```
- Run FL with BiDO
```python
python federated_main.py --model=cnn --net MCNN --dataset=mnist --num_users 100 -R=10 -C='0.1' -B=60 -E=1 --iid=1 --testacc='1.0' --optimizer=sgd --lossfunc=crossentropy --lr='0.01' --measure 'HSIC' --lamdax 2 --lamday 20 --hsic_training True  
```
### GMI Attack on FL round models
- Pretrain GAN on prior dataset
```python
python train_gan.py --dataset mnist --prior_dataset fmnist 
```
or 
```python
python train_gan.py --dataset celeba --prior_dataset celeba 
```
**Note:** here you can choose diverse prior datasets.
- Launch a GMI attack on the round model
```python
python federated_main.py --model=cnn --net MCNN --dataset=mnist --num_users 100 -R=10 -C='0.1' -B=60 -E=1 --iid=1 --testacc='1.0' --optimizer=sgd --lossfunc=crossentropy --lr='0.01' --measure 'HSIC' --lamdax 2 --lamday 20 --hsic_training True  
```
**Note:** Round model indicates the global model received by the (malicious) participant in each communication round.
### KED-MI Attack on FL round models
- Pretrain GAN guided by the round model on prior dataset
```python
python k+1_gan_fl.py --dataset emnist --defense reg --targetor_name 'mnist_MCNN_idd[1]_R[1]_C[0.1]_E[1]_B[10]_Acc[83.34].tar' 
```
or 
```python
python k+1_gan_fl.py --dataset celeba --defense reg --targetor_name 'celeba_VGG16_idd[1]_R[10]_C[1.0]_E[50]_B[64]_Acc[71.37].tar' 
```
**Note:** The parameter "--targetor_name" refers to the attacked FL round model.
- Launch a KED-MI attack on the round model
```python
python recovery_batchhsic_fl.py --dataset mnist --priordata emnist --defense reg --attack_improve BATCHHSIC --times 10 --lamda 0 --sigma 0 --targetor_name 'mnist_MCNN_idd[1]_R[1]_C[0.2]_E[1]_B[10]_Acc[83.49].tar' --g_name 'G_mnist_MCNN_idd[1]_R[1]_C[0.2]_E[1]_B[10]_Acc[83.49].tar.tar' --d_name 'D_mnist_MCNN_idd[1]_R[1]_C[0.2]_E[1]_B[10]_Acc[83.49].tar.tar' --iter 5000 --seeds 500 --improved_flag --verbose True 
```
or 
```python
python recovery_batchhsic_fl.py --dataset celeba --priordata celeba --defense reg --attack_improve BATCHHSIC --times 5 --lamda 0 --sigma 0 --targetor_name 'celeba_VGG16_idd[1]_R[1]_C[1.0]_E[30]_B[32]_Acc[6725.].tar' --g_name 'G_celeba_VGG16_idd[1]_R[1]_C[1.0]_E[30]_B[32]_Acc[6725.].tar.tar' --d_name 'D_celeba_VGG16_idd[1]_R[1]_C[1.0]_E[30]_B[32]_Acc[6725.].tar.tar' --iter 5000 --seeds 5 --improved_flag --verbose True 
```
## Reference
If you find our work helpful in your research, please consider citing:
```
@inproceedings{peng2022BiDO,
title={FedInverse: Evaluating Privacy Leakage in Federated Learning},
author={Wu, Di and Bai, Jun and Song, Yiliao and Chen, Junjun and Zhou, Wei and Xiang, Yong and Sajjanhar, Atul},
booktitle={The Twelfth International Conference on Learning Representations (ICLR2024)},
year={2024}
}
```
## Acknowledgements
This code relies on partial contributions from other repositories, including [GMI](https://arxiv.org/abs/1911.07135), [KED-MI](https://arxiv.org/abs/2010.04092), [VMI](https://arxiv.org/abs/2201.10787), [MID](https://arxiv.org/abs/2009.05241), [BiDO](https://arxiv.org/abs/2206.05483).



