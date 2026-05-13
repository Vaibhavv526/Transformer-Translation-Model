from pathlib import Path
from model import Build_Transformer
from config import get_config, latest_weights_file_path
from tokenizers import Tokenizer

import torch
import sys


def causal_mask(size):
    mask = torch.triu(
        torch.ones((1, size, size)),
        diagonal=1
    ).type(torch.int)

    return mask == 0


def translate(sentence: str):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Config
    config = get_config()

    # Tokenizers
    tokenizer_src = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_src'])))
    )

    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_tgt'])))
    )

    # Build model
    model = Build_Transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config['d_model']
    ).to(device)

    # Load weights
    model_filename = "weights/best_model.pt"

    print(f"Loading model: {model_filename}")

    state = torch.load(model_filename, map_location=device)

    model.load_state_dict(state['model_state_dict'])

    # Evaluation mode
    model.eval()

    seq_len = config['seq_len']

    with torch.no_grad():

        # Tokenize source sentence
        source = tokenizer_src.encode(sentence)

        # Check length
        if len(source.ids) > seq_len - 2:
            raise ValueError("Sentence is too long")

        # Create encoder input
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

        # Add batch dimension
        source = source.unsqueeze(0).to(device)

        # Encoder mask
        source_mask = (
            source != tokenizer_src.token_to_id('[PAD]')
        ).unsqueeze(1).unsqueeze(2).to(device)

        # Encoder output
        encoder_output = model.encode(source, source_mask)

        # Decoder starts with SOS
        decoder_input = torch.tensor(
            [[tokenizer_tgt.token_to_id('[SOS]')]],
            dtype=torch.int64
        ).to(device)

        print(f"\nSOURCE: {sentence}")
        print("PREDICTED:", end=" ")

        # Generate tokens one by one
        while decoder_input.size(1) < seq_len:

            # Decoder mask
            decoder_mask = causal_mask(
                decoder_input.size(1)
            ).to(device)

            # Decoder output
            out = model.decode(
                encoder_output,
                source_mask,
                decoder_input,
                decoder_mask
            )

            # Project to vocab
            prob = model.project(out[:, -1])

            # Get next token
            _, next_word = torch.max(prob, dim=1)

            next_word_id = next_word.item()

            # Stop if EOS
            if next_word_id == tokenizer_tgt.token_to_id('[EOS]'):
                break

            # Append token
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

            # Print token
            print(
                tokenizer_tgt.decode([next_word_id]),
                end=' '
            )

    print()

    # Convert generated ids to text
    return tokenizer_tgt.decode(
        decoder_input[0].tolist()
    )


# Run from terminal
if __name__ == "__main__":

    sentence = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I am a student"
    )

    result = translate(sentence)

    print("\nFINAL TRANSLATION:")
    print(result)


