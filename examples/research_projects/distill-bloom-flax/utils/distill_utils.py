import jax.nn as nn
import jax.numpy as jnp


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def ce_loss(logits_student, logits_teacher):
    """
    Distillation loss as defined in Distill-BERT https://arxiv.org/pdf/1910.01108.pdf
    """
    probs_teacher = nn.softmax(logits_teacher, axis=-1)
    probs_student = nn.softmax(logits_student, axis=-1)

    loss = probs_teacher * (-jnp.log(probs_student))
    return jnp.sum(loss)


def lm_loss(logits_student, one_hot_label):
    """
    Distillation loss as defined in Distill-BERT https://arxiv.org/pdf/1910.01108.pdf
    """
    probs_student = nn.softmax(logits_student, axis=-1)

    loss = one_hot_label * (-jnp.log(probs_student))
    return jnp.sum(loss)
