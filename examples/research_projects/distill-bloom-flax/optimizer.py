import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import core, struct, traverse_util
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.compilation_cache import compilation_cache as cc


class TrainState(struct.PyTreeNode):
    step: int
    params: core.FrozenDict[str, Any]
    opt_state: optax.OptState
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    dropout_rng: jnp.ndarray = None
    epoch: int = 0
    train_time: float = 0.0  # total time the model trained
    train_samples: int = 0  # number of samples seen

    def apply_gradients(self, *, grads, **kwargs):
        opt_state = {}
        for k, param in self.params.items():
            update_fn = self.tx[k].update
            updates, new_opt_state = update_fn(grads[k], self.opt_state[k], param)
            self.params[k] = optax.apply_updates(param, updates)
            opt_state[k] = new_opt_state

        return self.replace(
            step=self.step + 1,
            params=self.params,
            opt_state=freeze(opt_state),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = {}
        init_fn = tx.init
        for k, p in params.items():
            opt_state[k] = init_fn(p)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=freeze(opt_state),
            **kwargs,
        )