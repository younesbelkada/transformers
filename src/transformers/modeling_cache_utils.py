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
from typing import List, Optional, Tuple, Union

import torch

class KeyValueCache:
    """
    An utility class to cache the key/value pairs for language models.
    """
    def __init__(self, batch_size, max_seq_length, hidden_size, num_attention_heads, device, dtype=torch.float32):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.kv_cache = torch.empty(
            (batch_size, 2, num_attention_heads, max_seq_length, hidden_size // num_attention_heads), device=device, dtype=dtype
        )

        self.current_position = 0

    def update_cache(self, key_states, value_states, seq_len=1):
        self.kv_cache[:, 0, :, self.current_position:self.current_position + seq_len] = key_states
        self.kv_cache[:, 1, :, self.current_position:self.current_position + seq_len] = value_states

        self.current_position += seq_len
    
    def reset(self):
        self.current_position = 0
    
    def get_cache(self):
        return self.kv_cache[:, 0, :, :self.current_position].squeeze(1), self.kv_cache[:, 1, :, :self.current_position].squeeze(1)