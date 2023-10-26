# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from ..utils import is_accelerate_available, is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AWQBackend, AWQLinearVersion


if is_torch_available():
    import torch.nn as nn

if is_accelerate_available():
    from accelerate import init_empty_weights


def replace_with_awq_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, has_been_replaced=False
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    backend = quantization_config.backend

    if is_auto_awq_available():
        if backend == AWQBackend.AUTOAWQ:
            from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
        elif backend == AWQBackend.LLMAWQ:
            from awq.quantize.qmodule import WQLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    if backend == AWQBackend.AUTOAWQ:
                        target_cls = (
                            WQLinear_GEMM if quantization_config.version == AWQLinearVersion.GEMM else WQLinear_GEMV
                        )
                    else:
                        target_cls = WQLinear

                    model._modules[name] = target_cls(
                        w_bit=quantization_config.w_bit,
                        group_size=quantization_config.q_group_size,
                        in_features=in_features,
                        out_features=out_features,
                        bias=module.bias is not None,
                        dev=module.weight.device,
                    )
                    has_been_replaced = True

                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_awq_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced