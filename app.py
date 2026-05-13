from pathlib import Path
import torch
from tokenizers import Tokenizer

from model import Build_Transformer
from config import get_config, latest_weights_file_path


# ---------------- MASK ----------------
def causal_mask(size):
    mask = torch.triu(
        torch.ones((1, size, size)),
        diagonal=1
    ).type(torch.int)

    return mask == 0


# ---------------- DEVICE ----------------
#device = torch.device(
#    "cuda" if torch.cuda.is_available() else "cpu"
#)

# Editing the cuda from gpt to cpu dependent so it will deploy on render 
device = torch.device("cpu")

print("Using device:", device)


# ---------------- CONFIG ----------------
config = get_config()

seq_len = config['seq_len']


# ---------------- TOKENIZERS ----------------
tokenizer_src = Tokenizer.from_file(
    str(Path(
        config['tokenizer_file'].format(config['lang_src'])
    ))
)

tokenizer_tgt = Tokenizer.from_file(
    str(Path(
        config['tokenizer_file'].format(config['lang_tgt'])
    ))
)


# ---------------- MODEL ----------------
model = Build_Transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    seq_len,
    seq_len,
    d_model=config['d_model']
).to(device)


# ---------------- LOAD WEIGHTS ----------------
model_filename = "weights/best_model.pt"

print(f"Loading model: {model_filename}")

state_dict = torch.load(
    model_filename,
    map_location=device
)

model.load_state_dict(state_dict)

# IMPORTANT
model.eval()


# ---------------- TRANSLATE FUNCTION ----------------
def translate(sentence: str):

    with torch.no_grad():

        source = tokenizer_src.encode(sentence)

        if len(source.ids) > seq_len - 2:
            raise ValueError("Sentence too long")

        source = torch.cat([

            torch.tensor(
                [tokenizer_src.token_to_id('[SOS]')],
                dtype=torch.int64
            ),

            torch.tensor(
                source.ids,
                dtype=torch.int64
            ),

            torch.tensor(
                [tokenizer_src.token_to_id('[EOS]')],
                dtype=torch.int64
            ),

            torch.tensor(
                [tokenizer_src.token_to_id('[PAD]')] * (
                    seq_len - len(source.ids) - 2
                ),
                dtype=torch.int64
            )

        ], dim=0)

        source = source.unsqueeze(0).to(device)

        source_mask = (
            source != tokenizer_src.token_to_id('[PAD]')
        ).unsqueeze(1).unsqueeze(2).to(device)

        encoder_output = model.encode(
            source,
            source_mask
        )

        decoder_input = torch.tensor(
            [[tokenizer_tgt.token_to_id('[SOS]')]],
            dtype=torch.int64
        ).to(device)

        while decoder_input.size(1) < seq_len:

            decoder_mask = causal_mask(
                decoder_input.size(1)
            ).to(device)

            out = model.decode(
                encoder_output,
                source_mask,
                decoder_input,
                decoder_mask
            )

            prob = model.project(out[:, -1])

            _, next_word = torch.max(prob, dim=1)

            next_word_id = next_word.item()

            if next_word_id == tokenizer_tgt.token_to_id('[EOS]'):
                break

            decoder_input = torch.cat(
                [
                    decoder_input,

                    torch.tensor(
                        [[next_word_id]],
                        dtype=torch.int64
                    ).to(device)
                ],
                dim=1
            )

        return tokenizer_tgt.decode(
            decoder_input[0].tolist()
        )
