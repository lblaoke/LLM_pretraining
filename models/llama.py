from collections import *
from typing import *

import torch
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaRMSNorm

from loralib import Linear, Embedding
r = 1

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/activations.py#L193
class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/activations.py#L200
ACT2CLS = {
    "leaky_relu": torch.nn.LeakyReLU,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "sigmoid": torch.nn.Sigmoid,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU,
    "tanh": torch.nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L257
class MyLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super(LlamaAttention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, r=r)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, r=r)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, r=r)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias, r=r)
        self._init_rope()

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L211
class MyLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super(LlamaMLP, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False, r=r)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False, r=r)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False, r=r)
        self.act_fn = ACT2FN[config.hidden_act]

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L693
class MyLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super(LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MyLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = MyLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L910
class MyLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, self.padding_idx, r=r)
        self.layers = torch.nn.ModuleList(
            [MyLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L1118
class MyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False, r=r)

        # Initialize weights and apply final processing
        self.post_init()

if __name__=='__main__':
    llm_config = LlamaConfig.from_pretrained('configs/llama_9m.json')
    LLM = MyLlamaForCausalLM(llm_config)
    print(LLM)
