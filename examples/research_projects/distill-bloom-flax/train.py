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
    teacher = FlaxBloomForCausalLM.from_pretrained(params.teacher_path, from_pt=True, use_scan=True, dtype=dtype)
    teacher_config = teacher.config

    if params.dtype == "bfloat16":
        teacher_params = teacher.to_bf16(teacher.params)
    elif params.dtype == "float32":
        teacher_params = teacher.to_fp32(teacher.params)
    else:
        teacher_params = teacher.params

    student = FlaxBloomForCausalLM.from_pretrained(params.student_path, from_pt=True, use_scan=True, dtype=dtype)
    student_config = student.config

    if params.dtype == "bfloat16":
        student_params = student.to_bf16(student.params)
    elif params.dtype == "float32":
        student_params = student.to_fp32(student.params)
    else:
        student_params = student.params

    student = FlaxBloomForCausalLM(student_config, _do_init=False, dtype=dtype, use_scan=True)
    teacher = FlaxBloomForCausalLM(teacher_config, _do_init=False, dtype=dtype, use_scan=True)

    # 4: Get distiller
    distiller = Distiller(params, dataloader, teacher, student, student_params, teacher_params, dtype)
    distiller.train()


if __name__ == "__main__":
    main()
