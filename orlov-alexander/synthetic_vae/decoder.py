from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput


T5_MODEL = "google/t5-efficient-tiny"


def _choose_trainable_layers(n_samples: int) -> str:
    if n_samples < 500:
        return "projector_only"
    if n_samples < 2000:
        return "projector_and_last2"
    return "full_decoder"


def _choose_max_new_tokens(avg_word_len: float) -> int:
    if avg_word_len < 30:
        return 64
    if avg_word_len < 100:
        return 128
    return 256


def _choose_epochs(n_samples: int) -> int:
    if n_samples < 500:
        return 30
    if n_samples < 2000:
        return 20
    return 10


class _EmbeddingTextDataset(Dataset):
    def __init__(
        self,
        embeddings: torch.Tensor,
        token_ids: list[list[int]],
        pad_id: int,
    ) -> None:
        self.embeddings = embeddings
        self.token_ids  = token_ids
        self.pad_id     = pad_id

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.token_ids[idx]

    def collate_fn(self, batch):
        embs, seqs = zip(*batch)
        embs    = torch.stack(embs)
        max_len = max(len(s) for s in seqs)

        padded = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)

        labels = padded.clone()
        labels[labels == self.pad_id] = -100

        return embs, labels


class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim: int, t5_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, t5_dim),
            nn.LayerNorm(t5_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, input_dim) -> (batch, 1, t5_dim)
        return self.net(x).unsqueeze(1)


class EmbeddingToTextDecoder:
    def __init__(
        self,
        t5_model_name: str = T5_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        lr: float = 3e-4,
        verbose: bool = True,
    ) -> None:
        self.t5_model_name = t5_model_name
        self.device        = device
        self.batch_size    = batch_size
        self.lr            = lr
        self.verbose       = verbose

        self.projector_: EmbeddingProjector | None = None
        self.t5_: T5ForConditionalGeneration | None = None
        self.tokenizer_: T5Tokenizer | None = None
        self.max_new_tokens_: int = 64
        self._train_mode: str = "projector_only"
        self._t5_dim: int = 256


    def fit(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        epochs: int | None = None,
    ) -> "EmbeddingToTextDecoder":
        n_samples, emb_dim = embeddings.shape

        self._train_mode     = _choose_trainable_layers(n_samples)
        avg_word_len         = self._estimate_avg_word_len(texts)
        self.max_new_tokens_ = _choose_max_new_tokens(avg_word_len)
        n_epochs             = epochs if epochs is not None else _choose_epochs(n_samples)

        if self.verbose:
            print(f"  load {self.t5_model_name}...")
        self.tokenizer_ = T5Tokenizer.from_pretrained(self.t5_model_name)
        self.t5_        = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)
        self.t5_.to(self.device)
        self._t5_dim = self.t5_.config.d_model

        if self.verbose:
            n_params = sum(p.numel() for p in self.t5_.parameters()) / 1e6
            print(f"  model           : {self.t5_model_name} ({n_params:.1f}M param, d_model={self._t5_dim})")
            print(f"  mode training   : {self._train_mode}")
            print(f"  avg len         : {avg_word_len:.1f} words")
            print(f"  max_new_tokens  : {self.max_new_tokens_}")
            print(f"  epochs          : {n_epochs}")

        self.projector_ = EmbeddingProjector(input_dim=emb_dim, t5_dim=self._t5_dim).to(self.device)
        self._apply_freeze_strategy()

        trainable_params = list(self.projector_.parameters())
        if self._train_mode != "projector_only":
            trainable_params += [p for p in self.t5_.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        if self.verbose:
            print("  tokenization texts...")
        token_ids = self._tokenize_no_padding(texts)

        dataset = _EmbeddingTextDataset(
            torch.FloatTensor(embeddings),
            token_ids,
            self.tokenizer_.pad_token_id,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        if self.verbose:
            print("  train decoder...")
        self.t5_.train()
        self.projector_.train()

        for epoch in range(n_epochs):
            total_loss = 0.0
            for emb_batch, label_batch in loader:
                emb_batch   = emb_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                encoder_out = self.projector_(emb_batch)  # (B, 1, t5_dim)

                outputs = self.t5_(
                    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                    labels=label_batch,
                )

                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            if self.verbose:
                print(f"    epoch {epoch + 1:3d}/{n_epochs}  loss={total_loss / len(loader):.4f}")

        return self


    def decode(
        self,
        embeddings: np.ndarray,
        num_beams: int = 2,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> list[str]:
        self._check_fitted()
        self.t5_.eval()
        self.projector_.eval()

        pad_id = self.tokenizer_.pad_token_id
        texts: list[str] = []
        emb_tensor = torch.FloatTensor(embeddings).to(self.device)

        with torch.no_grad():
            for i in range(0, len(emb_tensor), self.batch_size):
                batch  = emb_tensor[i : i + self.batch_size]
                b_size = batch.size(0)
                encoder_out = self.projector_(batch)  # (B, 1, t5_dim)

                decoder_input_ids = torch.full(
                    (b_size, 1),
                    fill_value=pad_id,
                    dtype=torch.long,
                    device=self.device,
                )

                gen_ids = self.t5_.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=self.max_new_tokens_,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=do_sample,
                    early_stopping=True,
                )
                texts.extend(self.tokenizer_.batch_decode(gen_ids, skip_special_tokens=True))

        return texts


    def decode_single(self, embedding: np.ndarray, **decode_kwargs) -> str:
        return self.decode(embedding[np.newaxis, :], **decode_kwargs)[0]


    def _apply_freeze_strategy(self) -> None:
        for p in self.t5_.parameters():
            p.requires_grad = False

        if self._train_mode == "projector_and_last2":
            for block in self.t5_.decoder.block[-2:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.t5_.lm_head.parameters():
                p.requires_grad = True

        elif self._train_mode == "full_decoder":
            for p in self.t5_.decoder.parameters():
                p.requires_grad = True
            for p in self.t5_.lm_head.parameters():
                p.requires_grad = True

        if self.verbose:
            n_t = sum(p.numel() for p in self.t5_.parameters() if p.requires_grad)
            n_p = sum(p.numel() for p in self.projector_.parameters())
            print(f"  train params T5     : {n_t:,}")
            print(f"  projector params    : {n_p:,}")

    def _tokenize_no_padding(self, texts: list[str]) -> list[list[int]]:
        result = []
        for text in texts:
            enc = self.tokenizer_(
                text,
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors=None,
            )
            result.append(enc["input_ids"])
        return result

    @staticmethod
    def _estimate_avg_word_len(texts: list[str]) -> float:
        return float(np.mean([len(t.split()) for t in texts]))

    def _check_fitted(self) -> None:
        if self.projector_ is None or self.t5_ is None:
            raise RuntimeError("Сначала вызов .fit(embeddings, texts).")
