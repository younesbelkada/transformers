import os
os.environ["FLAX_PROFILE"] = "true"

import jax
import jax.numpy as jnp
from dataloader import AutoRegressiveDataLoader, AutoRegressiveDataset
from distiller import Distiller
from hparams import Parameters
from transformers.models.bloom.modeling_flax_bloom import FlaxBloomForCausalLM


def main():
    # 1: Get hyper parameters
    params = Parameters.parse().hparams
    dtype = getattr(jnp, params.dtype)

    # 2: Get dataset
    dataset = AutoRegressiveDataset(params)
    dataloader = AutoRegressiveDataLoader(dataset, params.batch_size, num_workers=params.num_workers)

    # 3: Get teacher and student models
    # TODO: pass empty models and the distiller should take care of doing TP
    teacher = FlaxBloomForCausalLM.from_pretrained(params.teacher_path, from_pt=True, use_scan=False, dtype=dtype)
    teacher_config = teacher.config

    if params.dtype == "bfloat16":
        teacher_params = teacher.to_bf16(teacher.params)
    elif params.dtype == "float32":
        teacher_params = teacher.to_fp32(teacher.params)
    else:
        teacher_params = teacher.params

    teacher = FlaxBloomForCausalLM(teacher_config, _do_init=False, dtype=dtype, use_scan=False)

    student = FlaxBloomForCausalLM.from_pretrained(params.student_path, from_pt=True, use_scan=False, dtype=dtype, revision="init_step")
    student_config = student.config

    if params.init_type == "random":
        student = FlaxBloomForCausalLM(student_config, use_scan=False, dtype=dtype)
        rng = jax.random.PRNGKey(758493) 
        student_params = student.init_weights(rng, (1, 1))


    if params.dtype == "bfloat16":
        student_params = student.to_bf16(student.params)
    elif params.dtype == "float32":
        student_params = student.to_fp32(student.params)
    else:
        student_params = student.params

    student = FlaxBloomForCausalLM(student_config, _do_init=False, dtype=dtype, use_scan=False)
    

    # 4: Get distiller
    distiller = Distiller(params, dataloader, teacher, student, student_params, teacher_params, dtype)
    distiller.train()

    # Save the models in a global device array
    # Check t5x checkpointers -> check bloom inference repo


if __name__ == "__main__":
    main()
