import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CharacterTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.char_to_id = {
            char: i
            for i, char in enumerate(self.chars)
        }

        self.id_to_char = {
            i: char
            for i, char in enumerate(self.chars)
        }

    def encode(self, text):
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids):
        return "".join(
            self.id_to_char[token_id]
            for token_id in token_ids
        )


class LanguageModelDataset(Dataset):
    def __init__(self, tokens, sequence_length):
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, index):
        inputs = self.tokens[
            index:index + self.sequence_length
        ]

        targets = self.tokens[
            index + 1:index + self.sequence_length + 1
        ]

        return inputs, targets


def create_dataloader(
    text,
    sequence_length,
    batch_size,
    tokenizer=None
):
    if tokenizer is None:
        tokenizer = CharacterTokenizer(text)

    tokens = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long
    )

    dataset = LanguageModelDataset(
        tokens=tokens,
        sequence_length=sequence_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return tokenizer, dataloader


def create_optimizer(
    model,
    learning_rate=1e-3,
    weight_decay=0.01
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


def compute_loss(model, inputs, targets):
    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    return loss


def train_step(
    model,
    optimizer,
    inputs,
    targets,
    device
):
    inputs = inputs.to(device)
    targets = targets.to(device)

    model.train()

    optimizer.zero_grad(set_to_none=True)

    loss = compute_loss(
        model,
        inputs,
        targets
    )

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(
    model,
    dataloader,
    device,
    max_batches=50
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            loss = compute_loss(
                model,
                inputs,
                targets
            )

            total_loss += loss.item()
            total_batches += 1

            if total_batches >= max_batches:
                break

    return total_loss / total_batches


def train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    steps,
    device,
    eval_interval=500,
    eval_batches=50
):
    model.train()

    data_iterator = iter(train_dataloader)

    losses = []
    history = []

    best_val_loss = float("inf")
    best_state_dict = None

    for step in range(steps):
        try:
            inputs, targets = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_dataloader)
            inputs, targets = next(data_iterator)

        loss = train_step(
            model,
            optimizer,
            inputs,
            targets,
            device
        )

        losses.append(loss)

        if (step + 1) % eval_interval == 0 or step == steps - 1:
            window_size = min(eval_interval, len(losses))

            train_loss = sum(
                losses[-window_size:]
            ) / window_size

            val_loss = evaluate(
                model,
                val_dataloader,
                device,
                eval_batches
            )

            print(
                f"step {step + 1} | "
                f"train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f}"
            )

            history.append({
                "step": step + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_state_dict = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }

    return best_state_dict, history