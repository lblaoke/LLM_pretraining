import os

import torch
from transformers import AutoConfig, AutoTokenizer
import datasets

import argparse
from time import time

import tech_util
import test_eval
import iter_dataloader
import models
import loralib

# args
parser = argparse.ArgumentParser()

parser.add_argument('--data-dir'    , type=str  , default='/data/shared_data/c4'    )
parser.add_argument('--config'      , type=str  , default='configs/llama_1b.json'   )
parser.add_argument('--save'        , type=str  , default=None                      )

parser.add_argument('--seed'        , type=int  , default=1                         )
parser.add_argument('--max-length'  , type=int  , default=128                       )
parser.add_argument('--max-iter'    , type=int  , default=100                       )

parser.add_argument('--lr'          , type=float, default=0.05                      )
parser.add_argument('--batch-size'  , type=int  , default=1                         )

args = parser.parse_args()

# GPU
tech_util.init_random_seed(args.seed)

# dataset
print('\n==> Loading dataset...')
_time = time()

data_train = datasets.load_from_disk(os.path.join(args.data_dir, 'validation'))
data_train = data_train.shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length=args.max_length)
# data_train = iter_dataloader.PreprocessedIterableDataset(data_train, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
# dataloader = torch.utils.data.DataLoader(data_train, batch_size=None, num_workers=32)

print(f'Training set of {args.data_dir} loaded in {(time() - _time)/60:.1f} min.')

t = []

for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    d_train = iter_dataloader.PreprocessedIterableDataset(data_train, tokenizer, batch_size=b, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(d_train, batch_size=None, num_workers=8)

    # model
    # print('\n==> Initializing LLM...')
    _time = time()
    llm_config = AutoConfig.from_pretrained(args.config)
    LLM = models.MyLlamaForCausalLM(llm_config)
    LLM.cuda()
    # LLM = torch.nn.parallel.DistributedDataParallel(LLM, device_ids=None, output_device=None, broadcast_buffers=False)
    end = time()
    # print(f'{args.config} initialized in {(time() - _time)/60:.1f} min.')

    # training utils
    trainable_params = [p for p in LLM.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0, weight_decay=0)

    pad_idx = tokenizer.pad_token_id
    tokens_seen = 0

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    # pretraining
    # print('\n==> Pretraining...')
    loralib.mark_only_lora_as_trainable(LLM)
    _time = time()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx > args.max_iter: break

        batch = {k: v.cuda() for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item()

        opt.zero_grad()
        loss = LLM(**batch, labels=labels).loss
        loss.backward()
        opt.step()

        # print(f'Iter: {batch_idx}, Loss: {loss.item():.4f}')

        # if (batch_idx + 1) % 20 == 0:
        #     total_loss, evaluated_on_tokens = test_eval.test(LLM, preprocess_batched, pad_idx, 1, 1, 'cuda', args.batch_size)
        #     print(f'Testing Loss: {total_loss.item():.4f}\n')

    t.append((time() - _time)/100)
    print(f'{b=}, {t=}')
    del LLM

# evaluation
# print('\n==> Final evaluating...')

# saving model
# print('\n==> Saving checkpoint...')
