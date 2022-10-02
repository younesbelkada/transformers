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
import jax, flax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import wandb

# from flax.training.train_state import TrainState
import optax

from optax import EmptyState
from flax.linen import partitioning as flax_partitioning
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental import PartitionSpec as P

from flax.linen.partitioning import with_sharding_constraint

from t5x.partitioning import PjitPartitioner
from t5x.train_state import FlaxOptimTrainState, InferenceState
from t5x.checkpoints import Checkpointer
import t5x.optimizers as optimizers



from logging_utils import logger
from optimizer import TrainState
from utils.distill_utils import one_hot, logical_axis_rules_full

AxisMetadata = flax_partitioning.AxisMetadata

# copied from: https://github.com/sanchit-gandhi/seq2seq-speech/blob/cfc6d73959486f5bd71c623ddd95843d62f5a614/run_flax_speech_recognition_seq2seq.py#L338
def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)

def to_fp16(t):
    return jax.tree_map(lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x, t)

def apply_gradient_descent(param, grad, learning_rate=0.0001):
    new_param = param - learning_rate * grad
    return new_param

class Distiller:
    def __init__(self, params, dataset, teacher, student, student_params, teacher_params, dtype):
        self.params = params
        self.dataset = dataset
        self.dtype = dtype

        self.teacher_model = teacher  # Empty modules
        self.student_model = student

        self.student_params = student_params
        self.teacher_params = teacher_params

        logger.info("Initializing Flax functions and utilities")

        rng = jax.random.PRNGKey(self.params.seed)
        self.rng, self.dropout_rng = jax.random.split(rng)

        logger.info("Initializing Partitioner")

        # Here we initialize the Jax partitionner to get the param specs
        self._init_student_partitioner()

        # Step 1: partition the student model! 
        self._partition_student_model()

        # Step 2: initialize optimizer
        # No need to partition the optimizer state since this is done automatically if the student
        # model is partitioned
        self._init_optimizer()

        # Here we initialize the Jax partitionner to get the param specs - this time with the teacher params
        self._init_teacher_partitioner()

        # Step 3: partition the student model
        self._partition_teacher_model()

        # Step 4: partition all the necessary forward functions
        self._init_p_forward_fn()

        # TODOs: wrap everything in one function for the training step -> partition the top most function
        # Checkpointer => for faster loading of the models and optimier states (save it as a GDA)

        # Change the rule names -> use the default rules
        # Dont pjit or partition the grad function

        # Future: push scripts so that each device on the host will be syncronized
        # For the final exp: v3-128 ?? Ask them for v3-256

    def _init_student_partitioner(self):
        r"""
            Utility function to initialize the partitionner. This is needed to partition the rest 
            of the attributes (model, optimizer) that are used later.

            A dummy state needs to be created to get the mesh_axes object.
            Function inspired from the snippet: https://github.com/huggingface/bloom-jax-inference/blob/2a04aa519d262729d54adef3d19d63879f81ea89/sharding_example.py#L67-L82 
        """
        num_mp_partitions = jax.device_count()
        self.student_partitioner = PjitPartitioner(num_mp_partitions, logical_axis_rules=logical_axis_rules_full)
        # self.student_partitioner = PjitPartitioner(model_parallel_submesh=(1, 1, 1, 1), logical_axis_rules=logical_axis_rules_full)
        # try out different values of the submesh -> 
        # self.partitioner = PjitPartitioner(model_parallel_submesh=(2, 2, 1, 2), logical_axis_rules=logical_axis_rules_full)
        
        # Get the mesh -> this will define how the TPU HBMs would be partitioned
        # Divide the MP partitioning by 2

        # not possible to overwrite the mesh like that
        # use the default names (data, model)
        # mesh_shape = (self.params.dp_devices, int(self.params.mp_devices/2))
        # devices = np.array(jax.devices()[:int(jax.device_count()/2)]).reshape(*mesh_shape)
        # self.student_partitioner.mesh = maps.Mesh(devices, ("data", "mp"))

        param_axes = jax.eval_shape(self._student_init)["params_axes"]

        dummy_state = InferenceState(
            step=jnp.array(0),
            params=freeze(self.student_model.params_shape_tree),
            params_axes=freeze(param_axes),
            flax_mutables=None,
            flax_mutables_axes=param_axes,
        )

        mesh_axes = self.student_partitioner.get_mesh_axes(dummy_state)
        self.student_params_spec = mesh_axes.params

        # Delete the intermediate variable
        del dummy_state
    
    def _init_teacher_partitioner(self):
        r"""
            Utility function to initialize the partitionner. This is needed to partition the rest 
            of the attributes (model, optimizer) that are used later.

            A dummy state needs to be created to get the mesh_axes object.
            Function inspired from the snippet: https://github.com/huggingface/bloom-jax-inference/blob/2a04aa519d262729d54adef3d19d63879f81ea89/sharding_example.py#L67-L82 
        """
        num_mp_partitions = jax.device_count()
        self.teacher_partitioner = PjitPartitioner(num_mp_partitions, logical_axis_rules=logical_axis_rules_full)
        # self.partitioner = PjitPartitioner(model_parallel_submesh=(2, 2, 1, 2), logical_axis_rules=logical_axis_rules_full)

        param_axes = jax.eval_shape(self._teacher_init)["params_axes"]

        # Get the mesh -> this will define how the TPU HBMs would be partitioned
        # Divide the MP partitioning by 2
        # mesh_shape = (self.params.dp_devices, int(self.params.mp_devices/2))
        # devices = np.array(jax.devices()[int(jax.device_count()/2):]).reshape(*mesh_shape)
        # self.teacher_partitioner.mesh = maps.Mesh(devices, ("data", "mp"))

        dummy_state = InferenceState(
            step=jnp.array(0),
            params=freeze(self.teacher_model.params_shape_tree),
            params_axes=freeze(param_axes),
            flax_mutables=None,
            flax_mutables_axes=param_axes,
        )

        mesh_axes = self.teacher_partitioner.get_mesh_axes(dummy_state)
        self.teacher_params_spec = mesh_axes.params

        # Delete the intermediate variable
        del dummy_state
    
    def _init_checkpointer(self):
        r"""
            Initializes the checkpointer, with that it is possible to load and save the optimizer states of stopped trainings
            Loads and overwrites the optimizer state with the one that has been saved if applicable
        """
        path_checkpoint = self.params.path_save_checkpoint if not self.params.restore_from_checkpoint else self.params.path_load_checkpoint
        
        state_shapes = jax.eval_shape(self._student_init)
        self.checkpointer = Checkpointer(state_shapes, path_checkpoint, use_gda=True, restore_dtype=jnp.bfloat16, save_dtype=jnp.bfloat16)

        if self.params.restore_from_checkpoint:
            # TODO: check if it works correctly
            self.state = self.checkpointer.restore(path=self.params.path_load_checkpoint)

            # TODO: update the dataset?

    def _init_optimizer(self):
        r"""
            Initialize the optimizer by creating a Optax optimizer
            The optimizer does not need to be partitionned if the partition function of the model has been called before 
            calling this function.

            Documentation at: https://flax.readthedocs.io/en/latest/advanced_topics/optax_update_guide.html 
        """
        optimizer_def = getattr(optimizers, self.params.optimizer_name)(self.params.learning_rate)


        # Needs to define a state spec for pjit to understand
        model_variables = flax.core.freeze({
            "params": self.student_params,
            "params_axes": jax.tree_map(lambda x: AxisMetadata(names=tuple(x)), self.student_params_spec)
        })

        def init_state(model_variables):
            return FlaxOptimTrainState.create(
                optimizer_def=optimizer_def,
                model_variables=model_variables
            )

        self.state = init_state(model_variables)

        del model_variables, self.student_params


    def _student_init(self):
        input_shape = (1, 1)
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        rng = jax.random.PRNGKey(0)
        return self.student_model.module.init(rng, input_ids, attention_mask, return_dict=False)
    
    def _teacher_init(self):
        input_shape = (1, 1)
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        rng = jax.random.PRNGKey(0)
        return self.teacher_model.module.init(rng, input_ids, attention_mask, return_dict=False)

    def _partition_student_model(self):
        r"""
            Function to partition the student model 
            Snippet inspired from: https://github.com/huggingface/bloom-jax-inference/blob/2a04aa519d262729d54adef3d19d63879f81ea89/sharding_example.py#L84
        """
        shard_params = self.student_partitioner.partition(lambda x: x, (self.student_params_spec,), self.student_params_spec)
        self.student_params = shard_params(freeze(self.student_params))

    def _partition_teacher_model(self):
        r"""
            Function to partition the teacher model 
            Snippet inspired from: https://github.com/huggingface/bloom-jax-inference/blob/2a04aa519d262729d54adef3d19d63879f81ea89/sharding_example.py#L84
        """
        shard_params = self.teacher_partitioner.partition(lambda x: x, (self.teacher_params_spec,), self.teacher_params_spec)
        self.teacher_params = shard_params(freeze(self.teacher_params))

    def _init_p_forward_fn(self):
        r"""
            Utility function to partition the loss computations and forward functions
        """
        self.batched_teacher_step = self.teacher_partitioner.partition(
            self._teacher_step, in_axis_resources=(self.teacher_params_spec, P("data")), out_axis_resources=P("data")
        )

        self._train_step = self.student_partitioner.partition(
            self._train_step, in_axis_resources=(self.student_params_spec, P("data"), P("data"), P("data")), out_axis_resources=None
        )
    
    def _teacher_step(self, params, batch):
        logits_teacher = self.teacher_model(batch, params=params).logits
        return logits_teacher

    def _log_param_norm(self, grad):
        layer_grad_norm = jax.tree_map(jnp.linalg.norm, grad).unfreeze()
        layer_param_norm = jax.tree_map(jnp.linalg.norm, self.state.params).unfreeze()

        logs = {
            "layer_grad_norm": layer_grad_norm,
            "layer_param_norm": layer_param_norm,
        }
        layer_param_norm = None
        layer_grad_norm = None

        return logs

    def _log_metrics(self, metrics, step, prefix=None):
        if jax.process_index() == 0:
            log_metrics = {}
            for k, v in metrics.items():
                if "layer" in k:
                    log_metrics[f"{k}/"] = v
                elif prefix is not None:
                    log_metrics[f"{prefix}/{k}"] = v
                else:
                    log_metrics[k] = v

            wandb.log(log_metrics, step)
    
    def _train_step(self, student_params, teacher_logits, batch, one_hot_labels):
        """
        Train step that implements Distillation loss as defined in Distill-BERT https://arxiv.org/pdf/1910.01108.pdf
        Together with LM loss for training auto regressive language models
        """
        # Step 1: get the teacher logits
        # teacher_logits = self.teacher_model(batch, params=self.teacher_params).logits

        # Step 2: get the student logits
        student_logits = self.student_model(batch, params=student_params).logits

        if self.params.dtype == "bfloat16":
            student_logits = to_bf16(student_logits)
            teacher_logits = to_bf16(teacher_logits)

        probs_teacher = nn.softmax(teacher_logits, axis=-1) + 1e-8 # For numerical stability 
        probs_student = nn.softmax(student_logits, axis=-1) + 1e-8 # For numerical stability 

        ce_loss = probs_teacher * (-jnp.log(probs_student))

        probs_student = nn.softmax(student_logits, axis=-1) + 1e-8 # For numerical stability 

        if self.params.use_vmap_trick:
            one_hot_labels = jnp.expand_dims(one_hot_labels, axis=0)

        lm_loss = one_hot_labels[:, 1:] * (-jnp.log(probs_student))
        return (ce_loss + lm_loss).mean()
    
    def _eval_step(self):
        pass

    def _one_hot(self, x, dtype=jnp.int8):
        """
            Create a one-hot encoding of x of size k. We store the vector
            in int8 to save memory    
        """
        return jnp.array(x[:, None] == jnp.arange(self.params.vocab_size), dtype)

    def train(self):
        wandb.init(
            project=self.params.wandb_project,
            entity=self.params.wandb_entity,
            config=self.params,
            dir=self.params.wandb_logs,
        )

        step = 0
        seen_tokens = 0

        # The gradient should compute with respect to the first arg - always
        # just partition the upper most function - nothing else
        grad_fn = jax.value_and_grad(self._train_step)

        for epoch in range(self.params.epochs):
            for i, batch in enumerate(self.dataset):
                # Get the predictions from the teacher + student and perform backpropagation
                if self.params.use_gradient_accumulation:

                    # TODO: partition it
                    grad = jax.tree_map(jnp.zeros_like, self.state.params)
                    grad = self.student_partitioner.partition(lambda x: x, (self.student_params_spec,), self.student_params_spec)(grad)
                    # grad = with_sharding_constraint(grad, self.student_params_spec)

                    loss = jnp.array([0])

                    for j in range(0, self.params.batch_size-self.params.micro_batch_size, self.params.micro_batch_size):
                        micro_batch = batch[j:(j+self.params.micro_batch_size)]

                        # step 1: get the teacher loss
                        logits_teacher = self.batched_teacher_step(self.teacher_params, micro_batch[:, :self.params.max_seq_len-1])

                        # step2: one hot encode the next tokens
                        # one_hot_labels = one_hot(micro_batch[:, :self.params.max_seq_len], self.params.vocab_size)
                        one_hot_labels = jax.pmap(self._one_hot, in_axes=(0))(micro_batch[:, :self.params.max_seq_len])

                        micro_batch = with_sharding_constraint(micro_batch, ("batch", "length"))
                        one_hot_labels = with_sharding_constraint(one_hot_labels, ("batch", "length"))

                        # get loss and gradients
                        # Getting the loss does not fail 
                        # But computing the gradients 

                        # Inspired from: https://github.com/borisdayma/dalle-mini/blob/f05d82b9c474a27f9def742ce97c1d265a2952e4/tools/train/train.py#L1210
                        # vmap does not apply the operations on the TPUs..
                        if self.params.use_vmap_trick:
                            interm_loss, micro_grad = jax.vmap(
                                grad_fn, in_axes=(None, 0, 0), out_axes=(None, 0)
                            )(self.state.params, micro_batch[:, :self.params.max_seq_len-1], one_hot_labels)
                            
                            micro_grad = jax.tree_util.tree_map(
                                lambda x: jnp.mean(x, axis=0), (micro_grad)
                            )
                        else:
                            interm_loss, micro_grad = grad_fn(self.state.params, logits_teacher, micro_batch[:, :self.params.max_seq_len-1], one_hot_labels)

                        micro_grad = with_sharding_constraint(micro_grad, self.student_params_spec)
                        # average accross batch size
                        # Do we need this?
                        # micro_grad = with_sharding_constraint(micro_grad, self.student_params_spec)

                        grad = jax.tree_map(lambda x, y : x+y, grad, micro_grad)
                        loss += interm_loss
                        print(interm_loss)

                        del one_hot_labels, micro_grad, interm_loss
                else:
                    # step 1: get the teacher loss
                    # logits_teacher = self.batched_teacher_step(teacher_model.params, batch[:, :i])
                    logits_teacher = self.batched_teacher_step(self.teacher_params, batch[:, :self.params.max_seq_len-1])

                    # step2: one hot encode the next tokens
                    # one_hot_labels = one_hot(batch[:, :self.params.max_seq_len], self.params.vocab_size)
                    one_hot_labels = jax.pmap(self._one_hot, in_axes=(0))(micro_batch[:, :self.params.max_seq_len])

                    # step3: perform backpropagation by computing student's loss
                    # Line below are copied and adapted from https://github.com/huggingface/transformers/blob/2c5747edfe383eee073119de784fa148befe9f2d/examples/flax/summarization/run_summarization_flax.py#L786
                    grad_fn = jax.value_and_grad(self._compute_loss)
                    loss, grad = grad_fn(self.student_params, logits_teacher, batch[:, :self.params.max_seq_len-1], one_hot_labels)

                # Average the gradients and the loss
                grad = jax.tree_map(lambda g: g / self.params.batch_size, grad)
                loss = jax.tree_map(lambda l: l / self.params.batch_size, loss)
                # End copied lines

                # If it prints here everything works as expected
                print("Loss ={}".format(loss.item()))

                # Inspired from: https://flax.readthedocs.io/en/latest/advanced_topics/optax_update_guide.html
                if self.params.optax_gradient:
                    updates, self.state = self.tx.update(grad, self.state)
                    self.student_params = optax.apply_updates(self.student_params, updates)
                    self._partition_student_model()

                    del updates
                # The t5x newest TrainState should take care of partitioning the opt state
                else:
                    self.state = self.state.apply_gradient(grad, learning_rate=self.params.learning_rate)

                # Update number of seen tokens
                seen_tokens += (batch.shape[0] * batch.shape[1])

                # step4: yey! log the results
                logs = {"loss": loss.item(), "learning_rate": self.params.learning_rate, "seen_tokens":seen_tokens}

                # log param norm and grad norm
                # Copied from: https://github.com/sanchit-gandhi/seq2seq-speech/blob/cfc6d73959486f5bd71c623ddd95843d62f5a614/run_flax_speech_recognition_seq2seq.py#L638
                # compute gradient norms over all layers, total encoder, total decoder and global for detailed monitoring
                norm_logs = self._log_param_norm(grad)
                logs.update(norm_logs)
                self._log_metrics(logs, step)

                del logs, norm_logs, grad

                step += 1

                if i % self.params.eval_steps == 0:
                    self._eval_step()
                #break

                if seen_tokens > 2e6:
                    break
            break
