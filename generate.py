import argparse

import torch

from model import Transformer


CHECKPOINT_PATH = "checkpoints/best_model.pt"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def generate(
    model,
    prompt,
    char_to_id,
    id_to_char,
    max_sequence_length,
    max_new_tokens,
    temperature=0.8,
    top_k=20
):
    token_ids = [
        char_to_id[char]
        for char in prompt
    ]

    for _ in range(max_new_tokens):
        context = token_ids[-max_sequence_length:]

        inputs = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            logits = model(inputs)

        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            threshold = values[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < threshold,
                torch.full_like(
                    logits,
                    -torch.inf
                ),
                logits
            )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        token_ids.append(next_token)

    return "".join(
        id_to_char[token_id]
        for token_id in token_ids
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "prompt",
    type=str
)

parser.add_argument(
    "--tokens",
    type=int,
    default=500
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.8
)

parser.add_argument(
    "--top-k",
    type=int,
    default=20
)

args = parser.parse_args()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model = Transformer(
    vocab_size=checkpoint["vocab_size"],
    max_sequence_length=checkpoint["max_sequence_length"],
    d_model=checkpoint["d_model"],
    num_heads=checkpoint["num_heads"],
    d_ff=checkpoint["d_ff"],
    num_layers=checkpoint["num_layers"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

text = generate(
    model=model,
    prompt=args.prompt,
    char_to_id=checkpoint["char_to_id"],
    id_to_char=checkpoint["id_to_char"],
    max_sequence_length=checkpoint["max_sequence_length"],
    max_new_tokens=args.tokens,
    temperature=args.temperature,
    top_k=args.top_k
)

print(text)