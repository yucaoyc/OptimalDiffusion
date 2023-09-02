# OptimalDiffusion
## Introduction:
This code repository aims to explore how the diffusion coefficient affects the sample generation quality in score-based diffusion models.

## How to run codes:
- 1D Gaussian: one may simply run the jupyter notebook therein.
- Swissroll and Gaussian mixtures: one may simply run the jupyter notebook in `Toy-examples/`.
- mnist:
```
cd mnist # move to the folder
python3 mnist_train.py # train the model
python3 mnist_visualize.py # generate assets
```
- cifar: you may run the bash scripts therein directly (after possible modification of hardware parameters).

## Acknowledgements:
This code repository uses source codes from the following resources:

| Title with link | Authors of codes | License |
|-----------------|------------------|---------|
| [sdeflow-light](https://github.com/CW-Huang/sdeflow-light) | Chin-Wei Huang |  MIT License |
| [pytorch-fid](https://github.com/mseitzer/pytorch-fid) | Maximilian Seitzer | Apache License 2.0 |
