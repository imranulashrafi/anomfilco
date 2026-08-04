from typing import List

import numpy as np


def centrality_from_embeddings(emb: np.ndarray) -> np.ndarray:
    x = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    xn = x / norms
    S = xn @ xn.T
    n = S.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    row_sum = S.sum(axis=1) - np.diag(S)
    return row_sum / (n - 1)


class Encoder:

    def __init__(self, name: str, device: str = "cpu", max_seq_length: int = 512,
                 pooling: str = "mean"):
        self.name = name
        self.device = device
        self.max_seq_length = max_seq_length
        self.pooling = pooling
        self.kind = None
        self._model = None
        self._tok = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.name, device=self.device, trust_remote_code=True
            )
            self._model.max_seq_length = self.max_seq_length
            self.kind = "st"
            return
        except Exception:
            pass
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.name, trust_remote_code=True
        ).to(self.device).eval()
        self.kind = "hf"

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        if self.kind == "st":
            return np.asarray(
                self._model.encode(
                    texts, batch_size=batch_size,
                    normalize_embeddings=True, convert_to_numpy=True,
                ),
                dtype=np.float64,
            )
        import torch
        out = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = self._tok(
                chunk, padding=True, truncation=True, max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state
            if self.pooling == "cls":
                vec = hidden[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                vec = summed / counts
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.cpu().numpy())
        return np.asarray(np.concatenate(out, axis=0), dtype=np.float64)
