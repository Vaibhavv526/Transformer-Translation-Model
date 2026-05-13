import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang,
        tgt_lang,
        seq_len
    ) -> None:

        super().__init__()

        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Source language special tokens
        self.sos_token_src = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")],
            dtype=torch.int64
        )

        self.eos_token_src = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")],
            dtype=torch.int64
        )

        self.pad_token_src = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")],
            dtype=torch.int64
        )

        # Target language special tokens
        self.sos_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")],
            dtype=torch.int64
        )

        self.eos_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")],
            dtype=torch.int64
        )

        self.pad_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")],
            dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):

        src_target_pair = self.ds[index]

        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Padding calculation
        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )

        dec_num_padding_tokens = (
            self.seq_len - len(dec_input_tokens) - 1
        )

        # Sentence too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            return self.__getitem__((index + 1) % len(self.ds))

        # Encoder input
        # [SOS] sentence [EOS] [PAD] ...
        encoder_input = torch.cat(
            [
                self.sos_token_src,

                torch.tensor(
                    enc_input_tokens,
                    dtype=torch.int64
                ),

                self.eos_token_src,

                torch.tensor(
                    [self.pad_token_src.item()] * enc_num_padding_tokens,
                    dtype=torch.int64
                )
            ]
        )

        # Decoder input
        # [SOS] sentence [PAD] ...
        decoder_input = torch.cat(
            [
                self.sos_token_tgt,

                torch.tensor(
                    dec_input_tokens,
                    dtype=torch.int64
                ),

                torch.tensor(
                    [self.pad_token_tgt.item()] * dec_num_padding_tokens,
                    dtype=torch.int64
                )
            ]
        )

        # Label
        # sentence [EOS] [PAD] ...
        label = torch.cat(
            [
                torch.tensor(
                    dec_input_tokens,
                    dtype=torch.int64
                ),

                self.eos_token_tgt,

                torch.tensor(
                    [self.pad_token_tgt.item()] * dec_num_padding_tokens,
                    dtype=torch.int64
                )
            ]
        )

        # Safety checks
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,

            "decoder_input": decoder_input,

            "encoder_mask":
                (encoder_input != self.pad_token_src)
                .unsqueeze(0)
                .unsqueeze(0),

            "decoder_mask":
                (
                    (decoder_input != self.pad_token_tgt)
                    .unsqueeze(0)
                    .unsqueeze(0)
                ) & causal_mask(decoder_input.size(0)),

            "label": label,

            "src_text": src_text,

            "tgt_text": tgt_text
        }


def causal_mask(size):

    mask = torch.triu(
        torch.ones((1, size, size)),
        diagonal=1
    ).type(torch.int)

    return mask == 0