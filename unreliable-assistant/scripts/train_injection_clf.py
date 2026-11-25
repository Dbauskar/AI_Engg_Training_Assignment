"""Train the injection classifier (logistic regression on bge-small embeddings) and save app/injection_clf.npz.
Usage: uv run python scripts/train_injection_clf.py [--fp-target 0.005]
Training data: evals/injection_train.py + the earlier dev/test/red-team sets + app/injection_examples.py (all now seen, so all are TRAINING data).
Held-out: evals/injection_holdout.py is never read here. The decision threshold is picked from out-of-fold predictions so that the
benign false-positive rate is <= --fp-target, not from the held-out set."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.injection_examples import ATTACK as EX_A, BENIGN as EX_B  # noqa: E402
from evals import injection_set as S, injection_train as T, redteam_set as R  # noqa: E402
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, OnnxEmbedder  # noqa: E402


def corpus():
    atk = list(T.ATTACK) + list(EX_A) + list(R.DIRECT) + [x if isinstance(x, str) else x["text"] for x in R.INDIRECT]
    atk += [turns[-1] for turns in R.CRESCENDO]
    ben = list(T.BENIGN) + list(EX_B) + list(R.BENIGN)
    for name in ("train.json", "train2.json"):  # LLM-generated (scripts/gen_injection_data.py); holdout3.json is never read here
        gen = ROOT / "evals/injection_gen" / name
        if gen.exists():
            g = json.loads(gen.read_text()); atk += g["direct"] + g["indirect"]; ben += g["benign"]
    for text, kind in S.DEV + S.TEST:
        (ben if kind == "benign" else atk).append(text)
    return [t for t in dict.fromkeys(atk)], [t for t in dict.fromkeys(ben)]


def train_lexical(atk, ben, fp_target):
    """Word + char TF-IDF logistic regression -> app/injection_lex.npz (served by app.injection.LexicalClassifier)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    tx = atk + ben
    y = np.array([1] * len(atk) + [0] * len(ben))
    vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit(tx)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit(tx)
    X = hstack([vw.transform(tx), vc.transform(tx)]).tocsr()
    mk = lambda: LogisticRegression(C=16.0, class_weight={0: 2.0, 1: 1.0}, max_iter=3000)  # noqa: E731
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    b = np.sort(oof[y == 0])
    thr = float(max(0.5, b[int(np.ceil(len(b) * (1 - fp_target))) - 1] + 1e-6))
    print(f"lexical out-of-fold: threshold {thr:.3f} -> recall {(oof[y == 1] >= thr).mean():.3f}, FP {(oof[y == 0] >= thr).mean():.4f}")
    m = mk().fit(X, y)
    nw = len(vw.vocabulary_)
    coef = m.coef_[0].astype(np.float32)
    voc = lambda v: np.array(sorted(v.vocabulary_, key=v.vocabulary_.get))  # noqa: E731
    np.savez(ROOT / "app" / "injection_lex.npz", threshold=np.float32(thr), b=np.float32(m.intercept_[0]),
             w_vocab=voc(vw), w_idf=vw.idf_.astype(np.float32), w_coef=coef[:nw],
             c_vocab=voc(vc), c_idf=vc.idf_.astype(np.float32), c_coef=coef[nw:])
    print("saved app/injection_lex.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fp-target", type=float, default=0.005)
    ap.add_argument("--lexical-only", action="store_true", help="retrain only app/injection_lex.npz and leave the embedding probe untouched")
    a = ap.parse_args()
    atk, ben = corpus()
    if a.lexical_only:
        print(f"training texts: {len(atk)} attack, {len(ben)} benign")
        train_lexical(atk, ben, a.fp_target)
        raise SystemExit(0)
    emb = OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2)
    X = np.vstack([emb.encode(atk, False), emb.encode(ben, False)])
    y = np.array([1] * len(atk) + [0] * len(ben))
    print(f"training texts: {len(atk)} attack, {len(ben)} benign")
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = LogisticRegression(C=4.0, class_weight={0: 2.0, 1: 1.0}, max_iter=2000).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    b = np.sort(oof[y == 0])
    thr = float(max(0.5, b[int(np.ceil(len(b) * (1 - a.fp_target))) - 1] + 1e-6)) if len(b) else 0.5
    print(f"out-of-fold: threshold {thr:.3f} -> recall {(oof[y == 1] >= thr).mean():.3f}, FP {(oof[y == 0] >= thr).mean():.4f}")
    m = LogisticRegression(C=4.0, class_weight={0: 2.0, 1: 1.0}, max_iter=2000).fit(X, y)
    np.savez(ROOT / "app" / "injection_clf.npz", w=m.coef_[0].astype(np.float32), b=np.float32(m.intercept_[0]),
             threshold=np.float32(thr), dim=np.int32(X.shape[1]), embedder="bge-small-en-v1.5")
    print("saved app/injection_clf.npz")
    train_lexical(atk, ben, a.fp_target)
