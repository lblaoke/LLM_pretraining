import torch
import datasets

def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch

def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

def test(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    val_data = datasets.load_from_disk('/data/shared_data/c4/validation')
    val_data = val_data.shuffle(seed=42)

    # if not args.single_gpu:
    #     val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # # Gather losses across all GPUs
    # gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    # torch.distributed.all_gather(gathered_losses, total_loss)
    # total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens
