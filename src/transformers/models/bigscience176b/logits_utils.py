import os, errno
import torch

ROOT_PATH = "/gpfswork/rech/six/uan68tv/data/tensors_to_test/"

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_logits(logit_name, logits, layer_number, layer_type):
    logits_path = os.path.join(ROOT_PATH, "logits_tr", "layer_{}_{}".format(layer_number, layer_type))
    create_dir(logits_path)
    file_name = logit_name + ".p"
    file_path = os.path.join(logits_path, file_name)
    torch.save(logits, file_path)