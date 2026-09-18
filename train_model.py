from pathlib import Path

import torch

from model import Transformer
from train import (
    CharacterTokenizer,
    create_dataloader,
    create_optimizer,
    train
)


DATA_PATH = Path("data/input.txt")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")

SEQUENCE_LENGTH = 128
BATCH_SIZE = 32

D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 4

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

TRAINING_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_BATCHES = 50


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

text = DATA_PATH.read_text(
    encoding="utf-8"
)

split_index = int(len(text) * 0.9)

train_text = text[:split_index]
val_text = text[split_index:]

tokenizer = CharacterTokenizer(train_text)

_, train_dataloader = create_dataloader(
    text=train_text,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer
)

_, val_dataloader = create_dataloader(
    text=val_text,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer
)

model = Transformer(
    vocab_size=tokenizer.vocab_size,
    max_sequence_length=SEQUENCE_LENGTH,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS
)

model = model.to(device)

optimizer = create_optimizer(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

best_state_dict, history = train(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    steps=TRAINING_STEPS,
    device=device,
    eval_interval=EVAL_INTERVAL,
    eval_batches=EVAL_BATCHES
)

model.load_state_dict(best_state_dict)

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint = {
    "model_state_dict": {
        name: parameter.cpu()
        for name, parameter in model.state_dict().items()
    },
    "vocab_size": tokenizer.vocab_size,
    "max_sequence_length": SEQUENCE_LENGTH,
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "d_ff": D_FF,
    "num_layers": NUM_LAYERS,
    "char_to_id": tokenizer.char_to_id,
    "id_to_char": tokenizer.id_to_char,
    "history": history
}

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

print(f"saved checkpoint to {CHECKPOINT_PATH}")