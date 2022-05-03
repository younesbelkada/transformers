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

path_meg_logits = "/gpfswork/rech/six/uan68tv/data/tensors_to_test/megatron-logits"
path_tr_logits = "/gpfswork/rech/six/uan68tv/data/tensors_to_test/transformers-logits"

input_linear_tr = torch.load(os.path.join(path_tr_logits, "context_layer_after_1.pt"), map_location=device)
input_linear_mgs = torch.load(os.path.join(path_meg_logits, "input_row_mlp_rank_0.pt"), map_location=device)

weight_tr = torch.load(os.path.join(path_tr_logits, "dense_weight_1.pt"), map_location=device)
weight_meg = torch.load(os.path.join(path_meg_logits, "weight_mlp_rank_0.pt"), map_location=device)

output_tr = torch.load(os.path.join(path_tr_logits, "output_attn_layer_1.pt"), map_location=device)
output_meg = torch.load(os.path.join(path_meg_logits, "output_layer_1_rank_0.pt"), map_location=device)

torch_assert_equal(F.linear(input_linear_tr, weight_tr), F.linear(input_linear_mgs, weight_meg))
torch_assert_equal(F.linear(input_linear_tr, weight_tr), output_meg)
torch_assert_equal(F.linear(input_linear_tr, weight_tr), output_tr)


