from jax import jit, vmap

import jax.nn as nn
import jax.numpy as jnp


def one_hot(batch_x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    # return jnp.array([x[:, None] == jnp.arange(k) for x in batch_x], dtype)
    return jnp.array([x[:, None] == jnp.arange(k) for x in batch_x], dtype)


@jit
def ce_loss(logits_student, logits_teacher):
    """
    Distillation loss as defined in Distill-BERT https://arxiv.org/pdf/1910.01108.pdf
    """
    probs_teacher = nn.softmax(logits_teacher, axis=-1)
    probs_student = nn.softmax(logits_student, axis=-1)

    loss = probs_teacher * (-jnp.log(probs_student))
    return jnp.sum(loss)


@jit
def lm_loss(logits_student, one_hot_label):
    """
    Distillation loss as defined in Distill-BERT https://arxiv.org/pdf/1910.01108.pdf
    """
    probs_student = nn.softmax(logits_student, axis=-1)

    loss = one_hot_label * (-jnp.log(probs_student))
    return jnp.sum(loss)


# 2D parameter and activation partitioning
# logical_axis_rules_full = [
#     ("batch", "dp"),
#     ("mlp", "mp"),
#     ("heads", "mp"),
#     ("vocab", "mp"),
#     # shard both activations and weight matrices on the remaining available axis
#     ("embed", "mp"),
#     ("embed", "dp"),
#     ("kv", None),
#     ("joined_kv", None),
#     ("relpos_buckets", None),
#     ("abspos_buckets", None),
#     ("length", None),
#     ("layers", None),
#     ("stack", None),
#     ("mlp_activations", None),
# ]

logical_axis_rules_full = [
    ("batch", "dp"),
    ("mlp", "mp"),
    ("heads", "mp"),
    ("vocab", "mp"),
    # shard both activations and weight matrices on the remaining available axis
    ("embed", "mp"),
    ("kv", None),
    ("joined_kv", None),
    ("relpos_buckets", None),
    ("abspos_buckets", None),
    ("length", None),
    ("layers", None),
    ("stack", None),
    ("mlp_activations", None),
]
