# FedInverse: Evaluating Privacy Leakage in Federated Learning
This repo contains the Pytorch-based implementation code for our published paper **FedInverse** in The Twelfth International Conference on Learning Representations (**ICLR2024**).

![FedInverse Example.png](https://github.com/Jun-B0518/FedInverse/blob/main/FedInverse%20Framework.png)
## Abstract
Federated Learning (FL) is a distributed machine learning technique where multiple devices (such as smartphones or IoT devices) train a shared global model by using their local data. FL promises better data privacy as the individual data isn't shared with servers or other participants. However, this research uncovers a groundbreaking insight: a model inversion (MI) attacker, who acts as a benign participant, can invert the shared global model and obtain the data belonging to other participants. In such scenarios, distinguishing between attackers and benign participants becomes challenging, leading to severe data leakage risk in FL. In addition, we found that even the most advanced defense approaches could not effectively address this issue. Therefore, it is important to evaluate such data leakage risks of an FL system before using it. Motivated by that, we propose FedInverse to evaluate whether the FL global model can be inverted by MI attackers. In particular, FedInverse can be optimized by leveraging the Hilbert-Schmidt independence criterion (HSIC) as a regularizer to adjust the diversity of the MI attack generator. We test FedInverse with three typical MI attackers, GMI, KED-MI, and VMI. The experiments show that FedInverse can effectively evaluate the data leakage risk that attackers successfully obtain the data belonging to other participants.
## Requirements
This code has been tested on Ubuntu 16.04/20.04, with Python3.7, Pytorch 1.7, OpenCV 4.5.0, and CUDA 11.0/11.4.
## Get started
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



