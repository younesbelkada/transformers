import os
import torch
import torch.nn.functional as F

print(torch.__version__)
device = "cuda:0"

def torch_assert_equal(actual, expected, **kwargs):
    # assert_close was added around pt-1.9, it does better checks - e.g will check dimensions match
    if hasattr(torch.testing, "assert_close"):
        return torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, **kwargs)
    else:
        return torch.allclose(actual, expected, rtol=0.0, atol=0.0)

path_meg_logits = "/gpfswork/rech/six/uan68tv/data/tensors_to_test/logits_rank_0"
path_tr_logits = "/gpfswork/rech/six/uan68tv/data/tensors_to_test/logits_tr"

meg_layers_to_test = [path.lower() for path in os.listdir(path_meg_logits)]
tr_layers_to_test = [path.lower() for path in os.listdir(path_tr_logits)]

assert len(list(set(meg_layers_to_test) & set(meg_layers_to_test))) == len(meg_layers_to_test)

meg_layers_to_test = os.listdir(path_meg_logits)
tr_layers_to_test = os.listdir(path_tr_logits)

for folder in meg_layers_to_test:
    print("Testing {}".format(folder))
    tensors_to_test = os.listdir(os.path.join(path_meg_logits, folder))
    for tensors in tensors_to_test:
        meg_logits = torch.load(os.path.join(path_meg_logits, folder, tensors), map_location=device)
        tr_logits = torch.load(os.path.join(path_tr_logits, folder.lower(), tensors), map_location=device)
        try:
            torch_assert_equal(meg_logits, tr_logits)
            print("Success for {} | layer {} ".format(tensors, folder))
        except:
            print("Failed for {} | layer {} ".format(tensors, folder))
            # raise

