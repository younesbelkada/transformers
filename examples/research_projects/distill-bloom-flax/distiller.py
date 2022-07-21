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

import wandb

from logging_utils import logger

import jax.numpy as jnp
import jax.nn as nn

from jax import grad, jit, vmap
from jax.example_libraries import optimizers

from flax.training.train_state import TrainState

from utils.distill_utils import ce_loss, lm_loss, one_hot



class Distiller:
    def __init__(self, params, dataset, teacher, student, student_params):
        self.params = params
        self.dataset = dataset
        self.teacher = teacher
        self.student = student

        self.student_params = student_params

    def train(self):
        # wandb.init(project=self.params.wandb_project, entity=self.params.wandb_entity, config=self.params, dir=self.params.wandb_logs)

        self.batched_student_step = vmap(self._step_student, in_axes=(None, 0, 0, 0))
        self.batched_step_teacher = vmap(self._step_teacher)

        opt_init, opt_update, get_params = getattr(optimizers, self.params.optimizer_name)(self.params.step_size)
        opt_state = opt_init(self.student_params)

        self.state = TrainState.create(
            apply_fn=self.student.apply,
            params=self.student_params,   
        )


        for epoch in range(self.params.epochs):
            for batch in self.dataset:
                # step1: get the logits of the teacher
                for i in range(1, self.params.max_seq_len-1):

                    logits_teacher = self.batched_step_teacher(batch[:, :i])
                    # logits_student = batched_step_student(batch[:, :i])

                    one_hot_labels = one_hot(batch[:, i+1], self.params.vocab_size)
                    self.student_params = self._gradient_update(self.student_params, logits_teacher, batch[:, :i], one_hot_labels)
                break
    
    def _step_student(self, params, logits_teacher, sequence, one_hot_label):
        # STEP1: get student logits
        sequence = jnp.expand_dims(sequence, 0)
        logits_student = self.student(sequence, params=params).logits[:, -1, :]

        # STEP2: get ce loss
        _ce_loss = ce_loss(logits_student, logits_teacher)
        _lm_loss = lm_loss(logits_student, one_hot_label)
        return _ce_loss + _lm_loss
    
    def _compute_loss(self, params, logits_teacher, batch, one_hot_labels):
        loss = self.batched_student_step(params, logits_teacher, batch, one_hot_labels)
        return jnp.mean(loss)

    def _step_teacher(self, sequence):
        sequence = jnp.expand_dims(sequence, 0)
        final_logits = self.teacher(sequence).logits[:, -1, :]
        return final_logits
    
    def _gradient_update(self, params, logits_teacher, batch, one_hot_labels):
        grads = grad(self._compute_loss)(params, logits_teacher, batch, one_hot_labels)
        return 
        
    def evaluate(self):
        pass