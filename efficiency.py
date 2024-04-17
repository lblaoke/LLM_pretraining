import torch
from transformers import AutoConfig
import models
import loralib

import argparse
from time import time
from tqdm import tqdm

sizes = [
    '9m', '20m', '35m', '40m', '60m', '71m', \
    '100m', '130m', '250m', '350m', \
    '1b', '3b', '7b'
]

total_params, trainable_params, mems = [], [], []

for s in tqdm(sizes):
    llm_config = AutoConfig.from_pretrained('configs/llama_' + s + '.json')
    LLM = models.MyLlamaForCausalLM(llm_config)
    LLM.to(7)

    total_param, trainable_param = 0, 0
    loralib.mark_only_lora_as_trainable(LLM)

    for p in LLM.parameters():
        total_param += p.numel()
        if p.requires_grad:
            trainable_param += p.numel()

    total_params.append(total_param)
    trainable_params.append(trainable_param)
    mems.append(torch.cuda.memory_allocated(7)/1024/1024)

    del LLM

print(f'{total_params=}')
print(f'{trainable_params=}')
print(f'{mems=} MB')
