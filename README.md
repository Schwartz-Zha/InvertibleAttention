# InvertibleAttention
This is the main repo for the paper [Invertible Attention](https://arxiv.org/abs/2106.09003).


This is the main repo for this paper, all four kinds of invertible attention are implemented in "inv_attention.py" This repo also has a driver for the first experiment "Vlidating Invertibility" in the paper.

Other experiments are provided in different repos. For "Generative Modelling", it's provided [here](https://github.com/Schwartz-Zha/InvertibleAttention-iResNet).

## Required packages
numpy
torch
torchvision
PIL

## Dataset
Three datasets are used in this experiment, Cifar10, SVHN (Cropped 32x32) and CelebA Aligned Faces (218x178). The licences for them are listed in "dataset_license/" directory, different from the LICENSE under the main directory. The first two datasets can be automatically downloaded by the torchvision code, the users only need to manually download CelebA aligned faces dataset. 

## How to use
In "Validating Experiments", there're two sets of experiments. The first set is conducted on images of size 32x32 from three datasets, Cifar10, SVHN and CelebA Aligned Faces. Among these three datasets, only CeleA aligned face images need to be resized. The driver command are listed below. 

As for the first set of experiments, my experiments are conducted on a single RTX 3090, and each epoch only takes a few seconds to finish. Well, concatenation style attention may take a bit longer, but each experiment usually finishes within 15 min.

```shell
#cifar10
python attention_test.py --dataset cifar10 --epochs 100 --batch 64 --save_dir results/concat_cifar10/ --model concat

python attention_test.py --dataset cifar10 --epochs 100 --batch 64 --save_dir results/dot_cifar10/ --model dot

python attention_test.py --dataset cifar10 --epochs 100 --batch 64 --save_dir results/embedded_cifar10/ --model embedded

python attention_test.py --dataset cifar10 --epochs 100 --batch 64 --save_dir results/gaussian_cifar10/ --model gaussian
#svhn
python attention_test.py --dataset svhn --epochs 100 --batch 64 --save_dir results/concat_svhn/ --model concat

python attention_test.py --dataset svhn --epochs 100 --batch 64 --save_dir results/dot_svhn/ --model dot

python attention_test.py --dataset svhn --epochs 100 --batch 64 --save_dir results/embedded_svhn/ --model embedded

python attention_test.py --dataset svhn --epochs 100 --batch 64 --save_dir results/gaussian_svhn/ --model gaussian
#celebA
python attention_test.py --dataset celebA --epochs 100 --batch 64 --save_dir results/concat_celebA/ --model concat

python attention_test.py --dataset celebA --epochs 100 --batch 64 --save_dir results/dot_celebA/ --model dot

python attention_test.py --dataset celebA --epochs 100 --batch 64 --save_dir results/embedded_celebA/ --model embedded

python attention_test.py --dataset celebA --epochs 100 --batch 64 --save_dir results/gaussian_celebA/ --model gaussian
```

The second set of experiments are conducted on the original size of CelebA aligned images, each image is 218x178. At this scale, we are only able to set batch size to 1 given limited video memory size. 


In my experiments, I found the average one-epoch training time for Dot-product, Gaussian, and Embedded Gaussian style attention is around 15min, but 80min for Concatenation style (on a single RTX 3090). 
```shell
#Epoch = 1, celebA full scale
python attention_test.py --dataset celebA --epochs 1 --batch 1 --save_dir results/concat_celebA_fullscale_epoch1/ --model concat --fullscale True

python attention_test.py --dataset celebA --epochs 1 --batch 1 --save_dir results/dot_celebA_fullscale_epoch1/ --model dot --fullscale True

python attention_test.py --dataset celebA --epochs 1 --batch 1 --save_dir results/dot_celebA_fullscale_epoch1/ --model dot --fullscale True

python attention_test.py --dataset celebA --epochs 1 --batch 1 --save_dir results/gaussian_celebA_fullscale_epoch1/ --model gaussian --fullscale True

python attention_test.py --dataset celebA --epochs 1 --batch 1 --save_dir results/embedded_celebA_fullscale_epoch1/ --model embedded --fullscale True

```

## Ackownledgement
We need to thank the authors of [Invertible Residual Networks](https://github.com/jhjacobsen/invertible-resnet) for kindly providing their training code in full detail and sharing it with MIT License. 

### Bibliography
```
@misc{zha2021invertible,
      title={Invertible Attention}, 
      author={Jiajun Zha and Yiran Zhong and Jing Zhang and Liang Zheng and Richard Hartley},
      year={2021},
      eprint={2106.09003},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```