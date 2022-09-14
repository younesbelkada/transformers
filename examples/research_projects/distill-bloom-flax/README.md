# Flax-Distill BLOOM

Author: @younesbelkada

This is the folder used to perform fast teacher / student open-source distillation project using Flax/JAX by leveraging the inference speed of the framework.

Inspired from `distillation` project from @VictorSanh

## Getting started

Let's get started

### Cloning a virtual environment

In Jax/Flax we use python venvs. 

```
sudo apt install python3.8-venv
```

Then,

```
python3 -m venv disk/venvs/distill-bloom
source disk/venvs/distill-bloom/bin/activate
```

Then, `git clone && cd transformers`

```
pip install -e ".[dev]"
```

Please refer to [this tutorial](https://cloud.google.com/tpu/docs/run-calculation-jax#install_jax_on_your_vm) to install JAX on your TPU.

Then install Flax by doing:
```
git clone https://github.com/google/flax.git
pip install --user -e flax
```
The second line might not work, try [this hack](https://stackoverflow.com/questions/64278198/error-can-not-perform-a-user-install-user-site-packages-are-not-visible-in) to make it work.

Try the commands to train a model using Flax on MNIST, you might need to 
```
pip install setuptools==59.5.0
```
to make it work

Install t5x by running:
```
git clone https://github.com/google-research/t5x.git
cd t5x
pip install -e .
```


### Downloading the dataset

Please make sure you have enougy storage to download the data BigScience dataset (~1.5TB)

For each dataset run:
```
python3 scripts/binarize_data.py --nb_datasets [NB_DATASETS] --batch_size [BATCH_SIZE] --index [INDEX]
```

`index` stands for the index of the dataset list that is retrieved from [here](https://huggingface.co/bigscience-data) - This useful when doing multiprocessing using tmux.

## Updates

## Setup

Install the dependencies with the command `pip install -r requirements.txt`.

