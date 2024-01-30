from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union


if TYPE_CHECKING:
    from ..modelling_utils import PreTrainedModel

import torch
from packaging import version

from . import (
    is_accelerate_available,
    is_auto_awq_available,
    is_auto_gptq_available,
    is_bitsandbytes_available,
    is_optimum_available,
    logging,
)


if is_bitsandbytes_available():
    import bitsandbytes as bnb

    from ..pytorch_utils import Conv1D

from .quantization_config import AwqConfig, BitsAndBytesConfig, GPTQConfig, QuantizationConfigMixin, QuantizationMethod


logger = logging.get_logger(__name__)


class QuantizationStatus(str, Enum):
    FRESH = "fresh"
    PREQUANTIZED = "prequantized"


class QuantizationConfigParser:
    """
    Parser for quantization configuration during model loading.
    Resolves conflicts between quantization config passed via `from_pretrained` and the one present in the model.
    Returns a quantizer instance to be used for model loading or None for not-quantized model.
    """

    @classmethod
    def get_quantizer(
        self,
        config,
        quantization_config_from_args=None,
    ):
        if hasattr(
            config, "quantization_config"
        ):  # the model is considered PREQUANTIZED and config from the model prevails
            quantization_config = self.validate_quantization_config(config.quantization_config)

            if quantization_config_from_args is not None:
                quantization_config_from_args = self.validate_quantization_config(quantization_config_from_args)
                quantization_config = self.handle_configs_collision(quantization_config, quantization_config_from_args)

            quantization_status = QuantizationStatus.PREQUANTIZED

        elif (
            quantization_config_from_args is not None
        ):  # only Q-config from args is present, assuming fresh quantization
            quantization_config = self.validate_quantization_config(quantization_config_from_args)

            if isinstance(quantization_config, AwqConfig):
                raise ValueError(
                    "You cannot quantize with AWQ a non-quantized model using transformers, please refer to the quantization documentation"
                    " to read more about how to quantize models with AWQ algorithm https://huggingface.co/docs/transformers/main_classes/quantization"
                )

            quantization_status = QuantizationStatus.FRESH

        else:  # Normal loading without quantization. No quant arguments present in args or model config.
            return

        config.quantization_config = quantization_config  # updating model's congig.quantization_config

        quantizer_class = self.get_quantizer_class(quantization_config)

        quantizer = quantizer_class(
            quantization_config=quantization_config,
            quantization_status=quantization_status,
        )
        return quantizer

    @staticmethod
    def validate_quantization_config(quantization_config):
        """
        checks quantization_config class
        converts dict format into QuantizationConfigMixin
        returns quantization_config as QuantizationConfigMixin
        """

        if isinstance(quantization_config, QuantizationConfigMixin):
            return quantization_config

        elif isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method", None)

            if quant_method is None:
                if quantization_config.get("load_in_8bit", False) or quantization_config.get("load_in_4bit", False):
                    logger.warning(
                        "The model's quantization_config has no `quant_method` attribute, but has `load_in_8bit` or `load_in_4bit` parameter set. "
                        "It is recommended to pass quantization config as `BitsAndBytesConfig`. "
                    )
                    quant_method = QuantizationMethod.BITS_AND_BYTES
                else:
                    raise ValueError(
                        "The model's quantization config from the arguments has no `quant_method` attribute."
                    )

            if quant_method == QuantizationMethod.BITS_AND_BYTES:
                config_class = BitsAndBytesConfig
            elif quant_method == QuantizationMethod.GPTQ:
                config_class = GPTQConfig
            elif quant_method == QuantizationMethod.AWQ:
                config_class = AwqConfig
            else:
                # unknown quantization_method, which is not None:
                raise NotImplementedError(
                    f"Unsupported quantization method detected: {quant_method}. Check for updates."
                )

            return config_class.from_dict(config_dict=dict(quantization_config))  # dict to class instance

        else:
            raise ValueError(
                f"Invalid type for `quantization_config`: {type(quantization_config)}. Should be a `dict` or a"
                " `QuantizationConfigMixin` subclass instance."
            )

    @staticmethod
    def handle_configs_collision(quantization_config, quantization_config_from_args):
        """
        handles situations where both quantization_config from args and quantization_config from model config are present
        args: config_quantization_config
        returns: selected and updated quantization_config
        """
        warning_msg = (
            "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading"
            " already has a `quantization_config` attribute. The `quantization_config` from the model will be prevail."
        )

        if isinstance(quantization_config, (GPTQConfig, AwqConfig)):
            # special case for GPTQ / AWQ config collision
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)
            warning_msg += f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."

        logger.warning(warning_msg)
        return quantization_config

    @staticmethod
    def get_quantizer_class(quantization_config):
        """selects HFQuantizer subclass based on quantization_config type and attrs"""
        if isinstance(quantization_config, BitsAndBytesConfig):
            if quantization_config.load_in_4bit:
                return Bnb4BitHFQuantizer
            elif quantization_config.load_in_8bit:
                return Bnb8BitHFQuantizer
            else:
                raise ValueError("BnB config should specify either `load_in_4bit` or `load_in_8bit`")
        elif isinstance(quantization_config, GPTQConfig):
            return GPTQHFQuantizer
        elif isinstance(quantization_config, AwqConfig):
            return AWQHFQuantizer
        else:
            raise ValueError(
                f"Unsupported quantization_config type {type(quantization_config)}; consider updating transformers version."
            )


def get_module_from_name(module, tensor_name: str) -> Tuple[Any, str]:
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]
    return module, tensor_name


class HFQuantizer(ABC):
    requires_parameters_quantization = False

    def __init__(
        self, quantization_config: QuantizationConfigMixin, quantization_status: QuantizationStatus, **kwargs
    ):
        self.quantization_config = quantization_config
        self.quantization_status = quantization_status
        self.modules_to_not_convert = []

    @staticmethod
    def get_locals_from_above():
        # this is a non-orthodox way to get above context for validation and adjustment.
        # allows not to think about individual vars to pass as arguments
        # allows combining validation with dtype, device_map
        # TODO: consider better alternatives or implement fully and replace other methods
        """
        Usage:
        ```
        locals_from_above = get_locals_from_above()
        if locals_from_above.get("from_tf", False) or locals_from_above.get("from_flax", False):
            raise ValueError
        locals_from_above[device_map] = ...
        ```
        """
        frame = inspect.currentframe()
        outer_frame = frame.f_back.f_back
        return outer_frame.f_locals

    @abstractmethod
    def validate_environment(self, *args, **kwargs):
        """checking env after instantiation, including modules presence and versions"""
        ...

    def process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs):
        """setting model attributes and/or converting model BEFORE weights loading"""
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        logger.info("Activating quantized loading for this model using {self.__class__}")

    def process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        model._is_quantized_training_enabled = self.is_model_trainable(model)

    def is_model_serializeable(self, model: PreTrainedModel) -> bool:
        return False

    def is_model_trainable(self, model: Optional[PreTrainedModel] = None) -> bool:
        return False

    def set_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        return torch_dtype

    def update_device_map(self, device_map):
        return device_map

    def get_special_dtypes_update(self, model, torch_dtype: torch.dtype) -> Dict[str, torch.dtype]:
        """returns dtypes for modules that are not quantized"""
        return {
            name: torch_dtype
            for name, _ in model.named_parameters()
            if any(m in name for m in self.modules_to_not_convert)
        }

    def adjust_target_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        return torch_dtype

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
        return max_memory

    def validate_device_map(self, device_map):
        """validates device map after process_model_before_weight_loading() and infer_auto_device_map()"""
        ...

    def check_quantized_param(
        self, model: PreTrainedModel, param_value: torch.Tensor, param_name: str, state_dict: Dict[str, Any]
    ) -> bool:
        """
        checks if a loaded state_dict component is part of quantized param + some validation; only defined if
        requires_parameters_quantization == True
        """
        return False

    def create_quantized_param(self, *args, **kwargs) -> torch.nn.Parameter:
        """
        takes needed components from state_dict and creates quantized param; only applicable if
        requires_parameters_quantization == True
        """
        if not self.requires_parameters_quantization:
            raise AttributeError(
                "`.create_quantized_param()` method is not supported by quantizer class {self.__cls__}."
            )

    def validate_saving(self, model):
        """checks if the quantized model can be successfully saved"""
        ...


class GPTQHFQuantizer(HFQuantizer):

    """
    quantization method:
        before weight loading converts layers into GPTQ layers.
    saving: from state_dict() as any normal model
        each quantized weight stores in state dict: .qweight, .qzeros, .scales
    loading: preprocess model into special GPTQ layers, then load into state_dict as usual
    """

    method = QuantizationMethod.GPTQ

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        from optimum.gptq import GPTQQuantizer

        self.quantizer = GPTQQuantizer.from_dict(self.quantization_config.to_dict())

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantized model.")

        if not (is_optimum_available() and is_auto_gptq_available()):
            raise ImportError(
                "Loading a GPTQ quantized model requires optimum (`pip install optimum`) and auto-gptq library (`pip install auto-gptq`)"
            )

        if version.parse(importlib.metadata.version("auto_gptq")) < version.parse("0.4.2"):
            raise ImportError(
                "You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq`"
            )

    def set_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.")
        return torch_dtype

    def process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs):
        super().process_model_before_weight_loading(model, **kwargs)
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")

        if self.quantization_status == QuantizationStatus.PREQUANTIZED:
            model = self.quantizer.convert_model(model)

    def process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        super().process_model_after_weight_loading(model, **kwargs)

        if self.quantization_status == QuantizationStatus.FRESH:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path

            self.quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = GPTQConfig.from_dict(self.quantizer.to_dict())
            # this line repeats config insertion done by GPTQQuantizer as dict, but now as GPTQConfig

        if self.quantization_status == QuantizationStatus.PREQUANTIZED:
            model = self.quantizer.post_init_model(model)

    def is_model_trainable(self, model: Optional[PreTrainedModel] = None):
        return True


class BnbHFQuantizer(HFQuantizer):
    """
    parent class from quantization methods from bitsandbytes
    """

    use_keep_in_fp32_modules = True
    method = QuantizationMethod.BITS_AND_BYTES
    requires_parameters_quantization = True

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not (is_accelerate_available() and is_bitsandbytes_available()):
            raise ImportError(
                "Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install accelerate` "
                "and the latest version of bitsandbytes: `pip install -i https://pypi.org/simple/ bitsandbytes`"
            )

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

    def set_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.float16
        return torch_dtype

    def update_device_map(self, device_map):
        """called right after quantizer init to fig initial device map"""
        if device_map is None:
            device_map = {"": torch.cuda.current_device()}
            logger.info(
                "The device_map was not initialized."
                "Setting device_map to `%s`."
                "If you want to use the model for inference, please set device_map ='auto'",
                str(torch.cuda.current_device()),
            )
        return device_map

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def validate_device_map(self, device_map):
        """called after infer_auto_device_map()"""
        device_map_without_lm_head = {
            key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
        }
        if "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
            raise ValueError(
                """
                Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the
                quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules
                in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to
                `from_pretrained`. Check
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
                for more details.
                """
            )

    def process_model_before_weight_loading(
        self,
        model: PreTrainedModel,
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        super().process_model_before_weight_loading(model, **kwargs)

        from ..integrations import get_keys_to_not_convert, replace_with_bnb_linear

        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if self.quantization_config.llm_int8_skip_modules is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # TODO: consider bringing replace_with_bnb_linear() code from ..integrations/bitsandbyter.py to here

        model.config.quantization_config = self.quantization_config

    def process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        super().process_model_after_weight_loading(model, **kwargs)


class Bnb8BitHFQuantizer(BnbHFQuantizer):
    """
    8-bit quantization from bitsandbytes quantization method:
        before loading: converts transformer layers into Linear8bitLt during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at fitst .cuda() call
    saving:
        from state dict, as usual; saves weights and 'SCB' component
    loading:
        need to locate SCB component and pass to the Linear8bitLt object
    """

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.37.2"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 8bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    def adjust_target_dtype(self, target_dtype: torch.dtype) -> torch.dtype:
        if target_dtype != torch.int8:
            logger.info("target_dtype {target_dtype} is replaced by `torch.int8` for 8-bit BnB quantization")
        return torch.int8

    def is_model_serializeable(self, model: PreTrainedModel = None) -> bool:
        return version.parse(importlib.metadata.version("bitsandbytes")) > version.parse("0.37.2")

    def check_quantized_param(
        self, model: PreTrainedModel, param_value: torch.Tensor, param_name: str, state_dict: Dict[str, Any]
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module._parameters[tensor_name], bnb.nn.Int8Params):
            if self.quantization_status == QuantizationStatus.PREQUANTIZED:
                if param_name.replace("weight", "SCB") not in state_dict.keys():
                    raise ValueError("Missing quantization component `SCB`")
                if param_value.dtype != torch.int8:
                    raise ValueError(
                        "Incompatible dtype `{param_value.dtype}` when loading 8-bit prequantized weight. Expected `torch.int8`."
                    )
            return True
        return False

    def create_quantized_param(
        self,
        model: PreTrainedModel,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
    ):
        """
        combines logic from _load_state_dict_into_meta_model and .integrations.bitsandbytes.py::set_module_quantized_tensor_to_device()
        needs aux items from state dicts, if found - removes them from unexpected_keys
        """

        fp16_statistics_key = param_name.replace("weight", "SCB")
        fp16_statistics = state_dict.get(fp16_statistics_key, None)

        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        if not isinstance(module._parameters[tensor_name], bnb.nn.Int8Params):
            raise ValueError(f"Parameter `{tensor_name}` should only be a `bnb.nn.Int8Params` instance.")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        new_value = param_value.to("cpu")
        if self.quantization_status == QuantizationStatus.PREQUANTIZED and not self.is_model_serializeable():
            raise ValueError(
                "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
            )

        # Support models using `Conv1D` in place of `nn.Linear` (e.g. gpt2) by transposing the weight matrix prior to quantization.
        # Since weights are saved in the correct "orientation", we skip transposing when loading.
        if issubclass(module.source_cls, Conv1D):
            if fp16_statistics is None:
                new_value = new_value.T

        kwargs = old_value.__dict__
        new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value
        if fp16_statistics is not None:
            setattr(module.weight, "SCB", fp16_statistics.to(target_device))
            unexpected_keys.remove(fp16_statistics_key)

    def is_model_trainable(self, model: Optional[PreTrainedModel] = None) -> bool:
        return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.37.0")

    def process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        super().process_model_after_weight_loading(model, **kwargs)
        model.is_8bit_serializable = self.is_model_serializeable()
        model.is_loaded_in_8bit = True

    def validate_saving(self, model):
        if not self.is_model_serializeable() and not getattr(model, "_hf_peft_config_loaded", False):
            raise NotImplementedError(
                "You are calling `save_pretrained` to a 8-bit converted model, but your `bitsandbytes` version doesn't support it. "
                "If you want to save 8-bit models, make sure to have `bitsandbytes>0.37.2` installed."
            )


class Bnb4BitHFQuantizer(BnbHFQuantizer):
    """
    4-bit quantization from bitsandbytes.py quantization method:
        before loading: converts transformer layers into Linear4bit during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear4bit into 4bit at the first .cuda() call
        saving:
            from state dict, as usual; saves weights and `quant_state` components
        loading:
            need to locate `quant_state` components and pass to Param4bit constructor
    """

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.39.0"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    def adjust_target_dtype(self, target_dtype: torch.dtype) -> torch.dtype:
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.19.0"):
            from accelerate.utils import CustomDtype

            if target_dtype != torch.int8:
                logger.info("target_dtype {target_dtype} is replaced by `CustomDtype.INT4` for 4-bit BnB quantization")
            return CustomDtype.INT4
        else:
            raise ValueError(
                "You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source to support fp4 auto device map"
                "calculation. You may encounter unexpected behavior, or pass your own device map"
            )

    def is_model_trainable(self, model: Optional[PreTrainedModel] = None) -> bool:
        return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.37.0")

    def is_model_serializeable(self, model: PreTrainedModel = None) -> bool:
        return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.41.3")

    def check_quantized_param(
        self, model: PreTrainedModel, param_value: torch.Tensor, param_name: str, state_dict: Dict[str, Any]
    ) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
            # Add here check for loaded components' dtypes once serialization is implemented
            return True
        elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
            # bias could be loaded by regular set_module_tensor_to_device() from accelerate,
            # but it would wrongly use uninitialized weight there.
            return True
        else:
            return False

    def create_quantized_param(
        self,
        model: PreTrainedModel,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
    ):
        """
        combines logic from _load_state_dict_into_meta_model and .integrations.bitsandbytes.py::set_module_quantized_tensor_to_device()
        """
        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        if tensor_name == "bias":
            if param_value is None:
                new_value = old_value.to(target_device)
            else:
                new_value = param_value.to(target_device)

            new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value
            return

        if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
            raise ValueError("this function only loads `Linear4bit components`")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        # construct `new_value` for the module._parameters[tensor_name]:
        if self.quantization_status == QuantizationStatus.PREQUANTIZED:
            # 4bit loading. Collecting components for restoring quantized weight
            # This can be expanded to make a universal call for any quantized weight loading

            if not self.is_model_serializeable():
                raise ValueError(
                    "Detected int4 weights but the version of bitsandbytes is not compatible with int4 serialization. "
                    "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                )

            if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
                param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
            ):
                raise ValueError(
                    f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components."
                )

            quantized_stats = {}
            for k, v in state_dict.items():
                if param_name + "." in k:
                    quantized_stats[k] = v
                    unexpected_keys.remove(k)

            new_value = bnb.nn.Params4bit.from_prequantized(
                data=param_value,
                quantized_stats=quantized_stats,
                requires_grad=False,
                device=target_device,
            )
        else:  # self.quantization_status == QuantizationStatus.FRESH
            new_value = param_value.to("cpu")

            # Support models using `Conv1D` in place of `nn.Linear` (e.g. gpt2) by transposing the weight matrix prior to quantization.
            # Since weights are saved in the correct "orientation", we skip transposing when loading.
            if issubclass(module.source_cls, Conv1D):
                new_value = new_value.T

            kwargs = old_value.__dict__
            new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value

    def process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        super().process_model_after_weight_loading(model, **kwargs)
        model.is_loaded_in_4bit = True

    def validate_saving(self, model):
        if not self.is_model_serializeable() and not getattr(model, "_hf_peft_config_loaded", False):
            raise NotImplementedError(
                "You are calling `save_pretrained` to a 4-bit converted model, but your `bitsandbytes` version doesn't support it. "
                "If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed."
            )


class AWQHFQuantizer(HFQuantizer):
    """
    4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://arxiv.org/abs/2306.00978):
        before loading: converts transformer layers usinf `replace_with_awq_linear`
        saving:
            from state dict, as usual
        loading:
            loading: preprocess model into special AWQ layers, then load into state_dict as usual
    """

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, torch_dtype, from_tf, from_flax):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run AWQ quantized model.")

        if not is_auto_awq_available():
            raise ImportError("Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)")

        if not is_accelerate_available():
            raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

    def validate_device_map(self, device_map):
        if device_map is None:
            logger.warning(
                "You have loaded an AWQ model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )

        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load an AWQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def set_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AWQ.")
        return torch_dtype

    def process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs):
        super().process_model_before_weight_loading(model, **kwargs)

        from ..integrations import get_keys_to_not_convert, replace_with_awq_linear

        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model, has_been_replaced = replace_with_awq_linear(
            model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
        )

        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )

    def process_model_after_weight_loading(self, model):
        super().process_model_after_weight_loading(model)
        if self.quantization_config.do_fuse:
            from ..integrations import fuse_awq_modules

            model = fuse_awq_modules(model, self.quantization_config)
            model._awq_is_fused = True  # TODO: consider storing this flag in model.config instead

    def is_model_trainable(self, model: Optional[PreTrainedModel] = None) -> bool:
        return False

    def validate_saving(self, model):
        if getattr(model, "_awq_is_fused", False):
            raise ValueError("You cannot save an AWQ model that uses fused modules!")