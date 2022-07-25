# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
import time

import jax
import jax.nn as nn
import jax.numpy as jnp
import wandb

import optax

from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze
from jax import jit, vmap, pmap
from jax.experimental import PartitionSpec as P

from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState


from logging_utils import logger
from utils.distill_utils import one_hot, logical_axis_rules_full


class Distiller:
    def __init__(self, params, dataset, teacher, student, student_params, teacher_params):
        self.params = params
        self.dataset = dataset

        self.teacher_model = teacher  # Empty modules
        self.student_model = student

        self.student_params = student_params
        self.teacher_params = teacher_params

        logger.info("Initializing Flax functions and utilities")

        rng = jax.random.PRNGKey(self.params.seed)
        self.rng, self.dropout_rng = jax.random.split(rng)

        logger.info("Initializing Optax optimizer states")

        tx = getattr(optax, self.params.optimizer_name)(self.params.learning_rate)

        logger.info("Initializing Partitionner")

        num_mp_partitions = jax.device_count()
        self.partitioner = PjitPartitioner(num_mp_partitions, logical_axis_rules=logical_axis_rules_full)

        param_axes = jax.eval_shape(self._init_fn)["params_axes"]

        state = InferenceState(
            step=jnp.array(0),
            params=freeze(self.student_model.params_shape_tree),
            params_axes=freeze(param_axes),
            flax_mutables=None,
            flax_mutables_axes=param_axes,
        )

        mesh_axes = self.partitioner.get_mesh_axes(state)
        self.params_spec = mesh_axes.params

        self._partition_student_model()
        self._partition_teacher_model()
        self._init_p_forward_fn()

        logger.info("Initializing Flax TrainState object")

        self.state = TrainState.create(
            apply_fn=self.student_model.__call__,
            params=self.student_params,
            tx=tx,
        )

    def _init_fn(self):
        input_shape = (1, 1)
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        rng = jax.random.PRNGKey(0)
        return self.student_model.module.init(rng, input_ids, attention_mask, return_dict=False)

    def _partition_student_model(self):
        shard_params = self.partitioner.partition(self.student_model.to_bf16, (self.params_spec,), self.params_spec)
        self.student_params = shard_params(freeze(self.student_params))

    def _partition_teacher_model(self):
        shard_params = self.partitioner.partition(self.teacher_model.to_bf16, (self.params_spec,), self.params_spec)
        self.teacher_params = shard_params(freeze(self.teacher_params))

    def _init_p_forward_fn(self):
        self._ce_loss = self.partitioner.partition(
            self._ce_loss, in_axis_resources=(P("data"), P("data")), out_axis_resources=None
        )
        # out axis has to be None since the output is a scalar

        self._lm_loss = self.partitioner.partition(
            self._lm_loss, in_axis_resources=(P("data"), P("data")), out_axis_resources=None
        )
        # out axis has to be None since the output is a scalar

        self.batched_student_step = self.partitioner.partition(
            self._student_step,
            in_axis_resources=(self.params_spec, P("data"), P("data"), P("data")),
            out_axis_resources=None,
        )
        self.batched_teacher_step = self.partitioner.partition(
            self._teacher_step, in_axis_resources=(self.params_spec, P("data")), out_axis_resources=P("data")
        )

    def _compute_loss(self, params, logits_teacher, batch, one_hot_labels):
        loss = self.batched_student_step(params, logits_teacher, batch, one_hot_labels)
        return jnp.mean(loss)

    def _student_step(self, params, logits_teacher, sequence, one_hot_label):
        # STEP1: get student logits
        logits_student = self.student_model(sequence, params=params).logits[:, -1, :]

        # STEP2: get ce loss
        _ce_loss = self._ce_loss(logits_student, logits_teacher)
        _lm_loss = self._lm_loss(logits_student, one_hot_label)
        return jnp.array(_ce_loss + _lm_loss)

    # @jit
    def _teacher_step(self, params, sequence):
        # sequence = jnp.expand_dims(sequence, 0)
        final_logits = self.teacher_model(sequence, params=params).logits[:, -1, :]
        return final_logits

    def _ce_loss(self, logits_student, logits_teacher):
        """
        Distillation loss as defined in Distill-BERT https://arxiv.org/pdf/1910.01108.pdf
        """
        probs_teacher = nn.softmax(logits_teacher, axis=-1)
        probs_student = nn.softmax(logits_student, axis=-1)

        loss = probs_teacher * (-jnp.log(probs_student))
        return jnp.array(jnp.sum(loss))

    def _lm_loss(self, logits_student, one_hot_label):
        """
        Distillation loss as defined in Distill-BERT https://arxiv.org/pdf/1910.01108.pdf
        """
        probs_student = nn.softmax(logits_student, axis=-1)

        loss = one_hot_label * (-jnp.log(probs_student))
        return jnp.array(jnp.sum(loss))

    def train(self):
        wandb.init(
            project=self.params.wandb_project,
            entity=self.params.wandb_entity,
            config=self.params,
            dir=self.params.wandb_logs,
        )

        for epoch in range(self.params.epochs):
            for batch in self.dataset:

                # Loop over each token, get the predictions from the teacher + student and perform backpropagation
                for i in range(1, self.params.max_seq_len - 1):
                    # step 1: get the teacher loss
                    # logits_teacher = self.batched_teacher_step(teacher_model.params, batch[:, :i])
                    logits_teacher = self.batched_teacher_step(self.teacher_params, batch[:, :i])
                    # step2: one hot encode the next tokens
                    one_hot_labels = one_hot(batch[:, i + 1], self.params.vocab_size)

                    # step3: perform backpropagation by computing student's loss
                    # Line below are copied and adapted from https://github.com/huggingface/transformers/blob/2c5747edfe383eee073119de784fa148befe9f2d/examples/flax/summarization/run_summarization_flax.py#L786
                    grad_fn = jax.value_and_grad(self._compute_loss)
                    loss, grad = grad_fn(self.state.params, logits_teacher, batch[:, :i], one_hot_labels)
                    # grad = jax.lax.pmean(grad, "batch")
                    # End copied lines

                    self.state = self.state.apply_gradients(grads=grad)

                    # step4: yey! log the results
                    wandb.log({"loss": loss.item()})
                break
