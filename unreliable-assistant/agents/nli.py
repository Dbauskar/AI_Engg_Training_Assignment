"""Natural-language-inference scorer for claim-to-chunk verification (premise = retrieved chunk, hypothesis = claim)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcp_servers.rag_pipeline import BATCH, _Onnx

DEFAULT_NLI_DIR = "models/nli-deberta-v3-xsmall"


class OnnxNli(_Onnx):
    model_id = "nli-deberta-v3-xsmall"

    def __init__(self, model_dir: str | Path = DEFAULT_NLI_DIR, threads: int | None = 2) -> None:
        super().__init__(model_dir, threads=threads)
        id2label = json.loads((Path(model_dir) / "config.json").read_text())["id2label"]
        order = {v.lower(): int(k) for k, v in id2label.items()}  # read from config: label order differs per model
        self._ent, self._con = order["entailment"], order["contradiction"]

    def probs(self, premises: list[str], hypotheses: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(entailment_prob, contradiction_prob) for each (premise, hypothesis) pair."""
        assert len(premises) == len(hypotheses)
        ent, con = [], []
        for i in range(0, len(premises), BATCH):
            enc = self.tok.encode_batch(list(zip(premises[i:i + BATCH], hypotheses[i:i + BATCH])))
            z = self._run(enc)
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            ent += p[:, self._ent].tolist()
            con += p[:, self._con].tolist()
        return np.array(ent, dtype=np.float32), np.array(con, dtype=np.float32)
