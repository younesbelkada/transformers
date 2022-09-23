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

## How the script works

### Step 1: Partition student and teacher model

Check the `logical_axis_rules_full` variable in `distill_utils.py` file. This should look to something like:
```
logical_axis_rules_full = [
    ("batch", "dp"),
    ("mlp", "mp"),
    ("heads", "mp"),
    ("vocab", "mp"),
    # shard both activations and weight matrices on the remaining available axis
    ("embed", "mp"),
    ("embed", "dp"),
    ("kv", None),
    ("joined_kv", None),
    ("relpos_buckets", None),
    ("abspos_buckets", None),
    ("length", None),
    ("layers", None),
    ("stack", None),
    ("mlp_activations", None),
]
```
With each element of the tuple corresponding to: 
1- The name of the kernel axis that are defined by each `DenseGeneral` layer
2- Where to shard this kernel axis -> `mp` stands for model parallelism (tensor parallelism) and `dp` stands for data parallelism. For example, the tuple `("embed", "mp")` means that each parameter that has the an axis name `embed` should be sharded on the `mp` axis.

Specifically, we first load the model using `_do_init_=False` to get the parameters in a frozen dict.

Then we have to define a `mesh` to explicitly tell how our partitioning would look like.

Here since we are using a single v3-8, we use a small hack to have control on the HBM devices we decide to manually set the mesh to our custom mesh as follows:

```
mesh_shape = (self.params.dp_devices, int(self.params.mp_devices/2))
devices = np.array(jax.devices()[:int(jax.device_count()/2):]).reshape(*mesh_shape)
self.student_partitioner.mesh = maps.Mesh(devices, ("dp", "mp"))
```

We assign half of the devices to the student model and the other half to the teacher model. I believe this is the hack on how you control on which devices to put the teacher/student model.

#### What is mesh_axes?

`mesh_axes` defines the structure of the mesh for each axis and each component for inference (params, flax_mutables, etc)
We use the `params` attribute to get the params spec of the parameters!


### Step 2: Partition the student optimizer

