
from transformers import FlaxAutoModelForCausalLM

from dataloader import AutoRegressiveDataLoader, AutoRegressiveDataset
from distiller import Distiller
from hparams import Parameters

def main():
    # 1: Get hyper parameters
    params = Parameters.parse().hparams

    # 2: Get dataset
    dataset = AutoRegressiveDataset(params)
    dataloader = AutoRegressiveDataLoader(dataset, params.batch_size)

    # 3: Get teacher and student models
    # TODO: pass empty models and the distiller should take care of doing TP
    teacher = FlaxAutoModelForCausalLM.from_pretrained(params.teacher_path, from_pt=True)
    student = FlaxAutoModelForCausalLM.from_pretrained(params.student_path, from_pt=True)

    # 4: Get distiller
    distiller = Distiller(params, dataloader, teacher, student)
    distiller.train()

if __name__ == "__main__":
    main()