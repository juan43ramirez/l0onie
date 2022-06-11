# l0onie

This repository contains the official implementation for the paper *L0onie: Compressing COINs with L0-constraints.*


This code enables the training of sparse SIREN models using constrained $L_0$
regularization for the task of image compression. We train models to minimize the 
reconstruction error of an image subject to a constraint on the compression ratio, measured
in bits-per-pixel. We implement and solve the constrained optimizationn problems using
the [Cooper](https://github.com/cooper-org/cooper) library.

**Fun fact:** Loonie is a colloquial name for the Canadian one dollar *coin*!

![](https://i.imgur.com/0pRCx4X.png)


*This figure depicts a circulating unit of currency. Its use in this work contends as fair
use for it is limited to a commentary relating to the **image** of the
currency itself. The image of all Canadian coins is the copyright of
the Royal Canadian Mint.*

## Setup

Create an environment using your favorite tool and install the required packages.

```pip install -r requirements.txt```

### Configs

We use `ml_collections.config_dict` to handle hyper-parameters for our experiments.
You can find config files under the `configs` folder.
In the [Examples](#examples) section, we demonstrate how to use these configs to trigger runs.

### Weights and Biases

If you don't have a Weights and Biases account, or prefer not to log metrics to
their servers, you can use the flag `--config.wandb.use_wandb=False`.


## Examples

Experiments can be triggered by runing the `main.py` script. We provide config
files to facilitate running the three supported experiment kinds:
training a fully dense model ([COIN](https://github.com/EmilienDupont/coin)),
training an $L_0$-sparsifiable model (L$_0$onie) or applying magnitude pruning on a
pre-trained model and then fine-tuning its remaining weights.

### L0onie
Train a L0onie to achieve a BPP of 0.15.
```
python main.py --config=configs/debug_task.py:gated --config.train.target_bpp=0.15
```

### COIN
Train a SIREN model with 5 hidden layers of 30 units each.

```
python main.py --config=configs/debug_task.py:coin_baseline --config.model.hidden_dims='(30,30,30,30,30)'
```

### Magnitude Pruning
Prune a pre-trained model to achieve a BPP of 0.07 and then fine-tune for 10000
steps.

```
python main.py --config=configs/debug_task.py:magnitude_pruning --config.model.hidden_dims='(30,30,30,30,30)' --config.train.target_bpp=0.07
```

Note that when performing magnitude pruning, we search for a pre-trained "COIN" model with the provided `hidden_dims`
in the WandB server. Therefore, in order for this command to
work you must first run the previous [baseline](#coin).

## Project structure

```
l0onie
├── bash_scripts
├── codec_baselines
├── configs
├── gated
├── get_results
├── kodak_dataset
├── magnitude_pruning
├── tests
├── utils
├── main.py            # Script used for running experiments
├── README.md
├── requirements.txt
```


### Bash scripts

We executed our experiments on a computer cluster with Slurm job management. The bash scripts used for triggering these runs are contained in the `bash_scripts` folder.

### Codec

We use the [CompressAI](https://interdigitalinc.github.io/CompressAI/) library to
compress images using JPEG. `baselines.py` automates this process for the Kodak dataset.


### Configs

`configs` contains files with basic configurations for the experiments presented throughout our paper.
For instance, `baseline.py` indicates the learning rate, optimizer and other details used
for our replication of COIN experiments.

These files were designed to be used in conjunction with the scripts in
the `bash_scripts` folder. Arguments that are required to trigger a run, but
were *not* specified in the config file are marked explicitly. You can find these
values in the corresponding `bash_scripts` file, as well as in the appendix of the
paper.



### Gated
The `gated` folder contains the main components used to construct l0onies.

Fully connected $L_0$ layers with *unstructured* sparsity are found inside of `layers.py`.

The `models.py` and `siren.py` modules implement MLPs composed of $L_0$ layers and Sine activations.
They also contain functionality for "pruning" these models, i.e. absorbing their
stochastic gates into their parameters.


### Results: plots

The `get_results` folder includes scripts for producing the plots and tables found in the paper.
These scripts depend on calls to the Weights and Biases API for retrieving logged metrics.
Calling `wandb_utils.py` retrieves and saves some metrics locally which are necessary for calling other scripts.

### Magnitude Pruning

We provide a `FrozenGatedModel`, which has untrainable gates associated with
each parameter. This model is leveraged in `magnitude_pruning.py`, where parameters
are "pruned" based on magnitude by setting their respective gates to zero.


### Tests

Automated tests implemented on Pytest are included in the `tests` folder.

### Utils

`utils` contains various project utils.

```
├── utils
│   ├── basic_utils.py
│   ├── cmp_utils.py        # Construct Cooper ConstrainedMinimizationProblem
│   ├── coin_utils.py       # Image compression utils, e.g. PSNR and BPP measurement
│   ├── core_utils.py       # Utils for main.py, e.g. train and compress loops
│   ├── wandb_utils.py      # Formatting for logging to WandB
```

## Acknowledgements
Our implementation of $L_0$-sparse models is based on the [code](https://github.com/AMLab-Amsterdam/L0_regularization)
by C. Louizos, M. Welling, and D. P. Kingma for the paper *Learning Sparse Neural Networks through L0 Regularization*, ICLR, 2018.

Our implementation of SIRENs (neural networks with Sine activation functions) and the initialization of their parameters is based on the [code](https://github.com/EmilienDupont/coin) by E. Dupont, A. Goliński, M. Alizadeh, Y. W. Teh and A. Doucet for the paper *COIN: COmpression with Implicit Neural representations*, ICLR - Neural Compression Workshop, 2021.
