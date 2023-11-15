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
"Transformers plugin-related file"
from abc import ABC, abstractmethod

import importlib
import json

from ..dynamic_module_utils import get_class_from_dynamic_module
from ..utils import cached_file, is_torch_available

PLUGIN_FILE_NAME = "transformers_plugin.py"
PLUGIN_CONFIG_NAME = "plugin_config.json"
PLUGIN_REQ_FILENAME = "requirements.txt"

if is_torch_available():
    import torch

def check_module_installed(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        raise ValueError(
            f"{module_name} is not installed - this is needed to use the plugin. Please install it with `pip install {module_name}`."
        )


def replace_target_class(module, new_class, target_class_name, init_kwargs):
    for name, child in module.named_children():
        if child.__class__.__name__ == target_class_name:
            
            with torch.device("meta"):
                new_class_instance = new_class(**init_kwargs)

            for attr, value in vars(child).items():
                setattr(new_class_instance, attr, value)            

            setattr(module, name, new_class_instance)
        else:
            # Recursively call the function for nested modules
            replace_target_class(child, new_class, target_class_name, init_kwargs)

    
class TransformersPlugin(ABC):
    """
    A TransfromersPlugin class retrieves the pluging either from a remote file or from the local package
    in case the plugin is natively supported by HF transformers.
    """

    @classmethod
    def _check_environment(cls, plugin_id, revision=None, token=None):
        requirements_file = cached_file(
            plugin_id,
            PLUGIN_REQ_FILENAME,
            revision=revision,
            token=token,
        )

        with open(requirements_file, "r") as f:
            requirements = f.readlines()
            for requirement in requirements:
                # Ignore comments and blank lines
                if not requirement.strip().startswith('#') and requirement.strip() != '':
                    module_name = requirement.split('=')[0].strip()
                    check_module_installed(module_name)


    @classmethod
    def _check_model_type(cls, plugin_id, model_type):
        plugin_config = cached_file(
            plugin_id,
            PLUGIN_CONFIG_NAME,
        )

        plugin_config = json.load(open(plugin_config, "r", encoding="utf-8"))

        if model_type not in plugin_config["plugin_mapping"]:
            raise ValueError(
                f"Plugin {plugin_id} does not support model type {model_type}."
            )
    
    @classmethod
    def sanity_check_plugin(cls, plugin_id, model_type, revision=None, token=None):
        """
        Sanity check the plugin by ensuring that the plugin requirements are installed.
        """
        cls._check_environment(plugin_id, revision=revision, token=token)
        cls._check_model_type(plugin_id, model_type)

    @abstractmethod
    def process_model_pre_init(self, model):
        """
        Process the model before initializing it. This is useful in case you want to 
        replace modules in the model.
        """
        ...

    @abstractmethod
    def process_model_post_init(self, model):
        """
        Process the model after initializing it. This is useful in case you want to 
        replace modules or other attributes in the model (e.g. fusing some modules together).
        """
        ...
    
    @classmethod
    def from_remote_hub(
        cls,
        plugin_id,
        model_type,
        config=None,
        revision=None,
        token=None,
    ):
        r"""
        Loads a plugin from the remote Hugging Face Hub and creates an instance of `TransformersPlugin`.
        """
        # Get the plugin config to retrieve the correct plugin class
        plugin_config = cached_file(
            plugin_id,
            PLUGIN_CONFIG_NAME,
        ) 

        plugin_config = json.load(open(plugin_config, "r", encoding="utf-8"))
        plugin_module_name = plugin_config["plugin_mapping"][model_type]
        plugin_module_name = PLUGIN_FILE_NAME.replace(".py", f".{plugin_module_name}")

        plugin_module = get_class_from_dynamic_module(
            plugin_module_name,
            plugin_id,
            revision=revision,
            token=token,
        )

        return plugin_module(config=config)