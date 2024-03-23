# OptimalDiffusion
## Introduction:
This is the codebase for the paper
"Exploring the Optimal Choice for Generative Processes in Diffusion Models: Ordinary vs Stochastic Differential Equations" published in NeurIPS 2023 ([poster link](https://neurips.cc/virtual/2023/poster/72998)). The arxiv version can be found [here](https://arxiv.org/abs/2306.02063).
This work aims at exploring how the diffusion coefficient affects the sample generation quality in score-based diffusion models.

## About codes:
- Gaussian (1D): one may simply run the jupyter notebook therein.
- Swissroll and Gaussian mixtures: please refer to the jupyter notebook in `Toy-examples/`.
- MNIST and CIFAR-10: please refer to the folder `Image-examples`. Before running bash scripts, you may consider modifying parameters therein.
```
Image-examples/mnist/mnist_run.sh: for both training and visualizing results.
Image-examples/cifar/cifar_train_run.sh: for training models from scratch.
Image-examples/cifar/cifar_fid_run_single.sh: compute FID using a single GPU.
Image-examples/cifar/cifar_fid_run.sh: compute FID using 3 GPUs.
Image-examples/cifar/cifar_sml_run.sh: compute Score-matching loss.
```

## Acknowledgements:
This code repository uses source codes from the following resources:

| Title with link | Authors of codes | License |
|-----------------|------------------|---------|
| [sdeflow-light](https://github.com/CW-Huang/sdeflow-light) | Chin-Wei Huang |  MIT License |
| [pytorch-fid](https://github.com/mseitzer/pytorch-fid) | Maximilian Seitzer | Apache License 2.0 |
