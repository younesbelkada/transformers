import jax.numpy as jnp
from dataloader import AutoRegressiveDataLoader, AutoRegressiveDataset
from distiller import Distiller
from hparams import Parameters
from transformers.models.bloom.modeling_flax_bloom import FlaxBloomForCausalLM


def main():
    # 1: Get hyper parameters
    params = Parameters.parse().hparams

    # 2: Get dataset
    dataset = AutoRegressiveDataset(params)
    dataloader = AutoRegressiveDataLoader(dataset, params.batch_size, num_workers=params.num_workers)

    # 3: Get teacher and student models
    # TODO: pass empty models and the distiller should take care of doing TP
    teacher = FlaxBloomForCausalLM.from_pretrained(params.teacher_path, from_pt=True, dtype=jnp.float16, use_scan=True)
    teacher_config = teacher.config
    teacher_params = teacher.params

    student = FlaxBloomForCausalLM.from_pretrained(params.student_path, from_pt=True, dtype=jnp.float16, use_scan=True)
    config = student.config
    student_params = student.params

    student = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, use_scan=True)
    teacher = FlaxBloomForCausalLM(teacher_config, _do_init=False, dtype=jnp.bfloat16, use_scan=True)

    # 4: Get distiller
    distiller = Distiller(params, dataloader, teacher, student, student_params, teacher_params)
    distiller.train()


if __name__ == "__main__":
    main()
