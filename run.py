from __future__ import annotations
import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


try:
    import faiss  
except Exception as e:
    faiss = None

try:
    from sentence_transformers import sentence_transformers
except Exception as e:  
    SentenceTransformer = None


# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    # L2-normalize rows for cosine/IP search
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def gumbel(shape, rng: np.random.RandomState, scale: float) -> np.ndarray:
    # Draw i.i.d. Gumbel(0, scale)
    u = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=shape)
    return -scale * np.log(-np.log(u))


# ------------------------------
# Composition accounting
# ------------------------------

def max_steps_sequential(eps_total: float, delta_total: float, eps_tok: float, delta_tok: float) -> int:
    if eps_tok <= 0: 
        return 0
    t_eps = math.floor(eps_total / eps_tok)
    t_delta = math.inf if delta_tok == 0 else math.floor(delta_total / delta_tok)
    return int(min(t_eps, t_delta))


def max_steps_advanced(eps_total: float, delta_total: float, eps_tok: float, delta_tok: float) -> int:
    """Advanced composition (Dwork-Roth style) via binary search for max T s.t.
    eps_total >= sqrt(2 T ln(1/delta_hat)) * eps_tok + T * eps_tok * (e^{eps_tok} - 1)
    and delta_total >= T*delta_tok + delta_hat.
    We fix delta_hat = delta_total / 2.
    """
    if eps_tok <= 0:
        return 0
    delta_hat = max(1e-16, delta_total / 2.0)
    # Quick upper bound for search
    hi = 1
    def ok(T: int) -> bool:
        term1 = math.sqrt(2 * T * math.log(1.0 / delta_hat)) * eps_tok
        term2 = T * eps_tok * (math.exp(eps_tok) - 1.0)
        eps_bound = term1 + term2
        delta_bound = T * delta_tok + delta_hat
        return (eps_bound <= eps_total + 1e-12) and (delta_bound <= delta_total + 1e-18)
    # Exponentially increase hi
    while ok(hi):
        hi *= 2
        if hi > 10**7:
            break
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return int(lo)


# ------------------------------
# Retrieval index
# ------------------------------
@dataclass
class Retriever:
    encoder_name: str
    dim: int
    encoder: object
    index: object
    doc_ids: List[int]
    texts: List[str]

    @classmethod
    def build(
        cls,
        encoder_name: str,
        wiki_count: int,
        seed: int,
        use_gpu: bool = False,
    ) -> "Retriever":
        assert SentenceTransformer is not None, "Please install sentence-transformers"
        assert faiss is not None, "Please install faiss-cpu"

        encoder = SentenceTransformer(encoder_name)
        if use_gpu and torch.cuda.is_available():
            encoder = encoder.to(torch.device("cuda"))

        # Load a slice of Wikipedia. Newer versions of `datasets` expose dated configs.
        wiki_config = os.environ.get("WIKIPEDIA_CONFIG", "20231101.en")
        try:
            wiki = load_dataset("wikimedia/wikipedia", wiki_config, split="train")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the 'wikimedia/wikipedia' dataset with config "
                f"{wiki_config!r}. Ensure the machine can reach Hugging Face, "
                "pre-download the dataset, or export WIKIPEDIA_CONFIG with a valid snapshot."
                f" Original error: {exc}"
            ) from exc
        wiki = wiki.select(range(min(wiki_count, len(wiki))))
        texts = [rec["text"][:2000] for rec in wiki]  # truncate to keep it light
        doc_ids = list(range(len(texts)))

        # Embed deterministically
        set_seed(seed)
        all_vecs = []
        for i in range(0, len(texts), 256):
            chunk = texts[i:i+256]
            vecs = encoder.encode(chunk, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            all_vecs.append(vecs)
        emb = np.vstack(all_vecs).astype("float32")
        assert emb.shape[0] == len(texts)
        dim = emb.shape[1]

        # FAISS index (Cosine via inner product on normalized vectors)
        index = faiss.index_factory(dim, "IDMap,Flat")
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        index.add_with_ids(emb, np.array(doc_ids, dtype=np.int64))

        return cls(
            encoder_name=encoder_name,
            dim=dim,
            encoder=encoder,
            index=index,
            doc_ids=doc_ids,
            texts=texts,
        )

    def retrieve_topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        # Deterministic embedding and top-k
        q = self.encoder.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        q = q.astype("float32")
        scores, ids = self.index.search(q, k)
        pairs = list(zip(ids[0].tolist(), scores[0].tolist()))
        # Deterministic tie-break: sort by (-score, doc_id)
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs


# ------------------------------
# Token generation helpers
# ------------------------------
@dataclass
class LLM:
    name: str
    tok: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device

    @classmethod
    def load(cls, name: str, use_gpu: bool = False) -> "LLM":
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        load_kwargs = {"use_safetensors": True, "low_cpu_mem_usage": True}
        if device.type == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs).to(device)
        model.eval()
        return cls(name=name, tok=tok, model=model, device=device)

    @torch.no_grad()
    def next_token_logits(self, prompt: str) -> Tensor:
        b = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model(**b)
        logits = out.logits[0, -1, :]  # last-step logits
        return logits.float().cpu()

    @torch.no_grad()
    def topk_tokens(self, prompt: str, k: int) -> List[int]:
        logits = self.next_token_logits(prompt)
        k = min(k, logits.numel())
        topk = torch.topk(logits, k=k)
        # Deterministic tie-break: sort by (-logit, token_id)
        pairs = list(zip(topk.indices.tolist(), topk.values.tolist()))
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return [tid for tid, _ in pairs]

    @torch.no_grad()
    def greedy_next_token(self, prompt: str) -> int:
        logits = self.next_token_logits(prompt)
        # Deterministic greedy with tie-break by token id
        max_val = torch.max(logits)
        candidates = (logits == max_val).nonzero(as_tuple=False).flatten().tolist()
        return int(min(candidates))


# ------------------------------
# LimitedDomain & DP mechanisms
# ------------------------------

def limited_domain_argmax(
    histogram: Dict[int, int],
    domain: List[int],
    eps: float,
    rng: np.random.RandomState,
    null_token_id: Optional[int] = None,
) -> int:
    """Pick an argmax privately over a *limited* domain using Gumbel noise.
       Sensitivity = 1 for counts. Adding Gumbel(scale=1/eps) to each utility
       is equivalent to the exponential mechanism w.r.t. argmax.
       If domain is empty, return null_token_id (EOS) if provided, else the min token id.
    """
    if not domain:
        return null_token_id if null_token_id is not None else (min(histogram.keys()) if histogram else 0)
    util = np.array([histogram.get(t, 0) for t in domain], dtype=np.float64)
    noisy = util + gumbel(util.shape, rng, scale=(1.0 / max(eps, 1e-12)))
    # Deterministic tie-break via stable sort on (-noisy, token_id)
    order = np.lexsort((np.array(domain), -noisy))
    return int(domain[int(order[0])])


def laplace(rng: np.random.RandomState, scale: float) -> float:
    u = rng.uniform(-0.5, 0.5)
    return -scale * np.sign(u) * math.log(1 - 2 * abs(u) + 1e-12)


# ------------------------------
# Prompts
# ------------------------------

def build_prompt(question: str, docs: List[str], prefix: str) -> str:
    context = "\n\n".join([d.strip() for d in docs if d and d.strip()])
    return (
        f"Question: {question}\n"
        f"Context:\n{context}\n\n"
        f"Answer so far: {prefix}"
    )


def build_prompt_nonrag(question: str, prefix: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Answer so far: {prefix}"
    )


# ------------------------------
# Main algorithms
# ------------------------------
@dataclass
class Config:
    algo: str
    m: int
    k: int
    kbar: int
    tau_frac: float
    eps_total: float
    delta_total: float
    eps_token: float
    delta_token: float
    max_steps_cap: int
    seed: int


def dpvote_rag(
    llm: LLM,
    retriever: Retriever,
    question: str,
    m: int,
    k: int,
    kbar: int,
    eps_total: float,
    delta_total: float,
    eps_token: float,
    delta_token: float,
    max_steps_cap: int,
    rng: np.random.RandomState,
) -> str:
    # 1) Retrieve mk docs once
    mk = m * k
    top = retriever.retrieve_topk(question, mk)
    docs = [retriever.texts[doc_id] for doc_id, _ in top]

    # 2) Uniform random partition into m subsets of k docs (public randomness via fixed seed)
    idxs = list(range(len(docs)))
    rng.shuffle(idxs)
    parts = [idxs[i*k:(i+1)*k] for i in range(m)]
    subsets = [[docs[j] for j in part] for part in parts]

    # 3) Compute Tmax from composition
    t_seq = max_steps_sequential(eps_total, delta_total, eps_token, delta_token)
    t_adv = max_steps_advanced(eps_total, delta_total, eps_token, delta_token)
    Tmax = int(min(max(t_seq, t_adv), max_steps_cap))

    # 4) Token loop
    y_tokens: List[int] = []
    prefix = ""
    eos_id = llm.tok.eos_token_id if llm.tok.eos_token_id is not None else 0

    for t in range(Tmax):
        # Gather one token per voter
        hist: Dict[int, int] = {}
        for i in range(m):
            prompt_i = build_prompt(question, subsets[i], prefix)
            tok = llm.greedy_next_token(prompt_i)
            hist[tok] = hist.get(tok, 0) + 1

        # LimitedDomain: reduce to top-\bar{k} of non-RAG logits
        instr_top = llm.topk_tokens(build_prompt_nonrag(question, prefix), kbar)
        yt = limited_domain_argmax(hist, instr_top, eps_token, rng, null_token_id=eos_id)

        # Append token and maybe stop
        y_tokens.append(yt)
        prefix += llm.tok.decode([yt], skip_special_tokens=False)
        if yt == eos_id:
            break

    return llm.tok.decode(y_tokens, skip_special_tokens=True).strip()


def dpsparsevote_rag(
    llm: LLM,
    retriever: Retriever,
    question: str,
    m: int,
    k: int,
    kbar: int,
    tau_frac: float,
    eps_total: float,
    delta_total: float,
    eps_token: float,
    delta_token: float,
    max_steps_cap: int,
    rng: np.random.RandomState,
) -> str:
    # Privacy split per Algorithm 2
    eps_token_rag = eps_token / 2.0
    eps_token_lap = eps_token / 2.0

    # 1) Retrieve mk docs once
    mk = m * k
    top = retriever.retrieve_topk(question, mk)
    docs = [retriever.texts[doc_id] for doc_id, _ in top]

    # 2) Uniform random partition
    idxs = list(range(len(docs)))
    rng.shuffle(idxs)
    parts = [idxs[i*k:(i+1)*k] for i in range(m)]
    subsets = [[docs[j] for j in part] for part in parts]

    # 3) cmax from composition (advanced/seq; same routine as Tmax but interpreted as number of *private* votes)
    c_seq = max_steps_sequential(eps_total, delta_total, eps_token, delta_token)
    c_adv = max_steps_advanced(eps_total, delta_total, eps_token, delta_token)
    cmax = int(max(c_seq, c_adv))
    c = cmax

    # 4) Threshold init (noisy)
    tau = max(1, int(round(tau_frac * m)))
    tau_hat = tau + laplace(rng, scale=2.0 / max(eps_token_lap, 1e-12))

    y_tokens: List[int] = []
    prefix = ""
    eos_id = llm.tok.eos_token_id if llm.tok.eos_token_id is not None else 0

    for step in range(min(max_steps_cap, 4 * cmax + 8)):
        # Non-RAG next token (public)
        y_nonrag = llm.greedy_next_token(build_prompt_nonrag(question, prefix))

        # Token histogram from voters
        hist: Dict[int, int] = {}
        for i in range(m):
            prompt_i = build_prompt(question, subsets[i], prefix)
            tok = llm.greedy_next_token(prompt_i)
            hist[tok] = hist.get(tok, 0) + 1

        # Count of y_nonrag in the histogram + noisy thresholding
        a_t = hist.get(y_nonrag, 0)
        check = a_t + laplace(rng, scale=4.0 / max(eps_token_lap, 1e-12))

        if (check <= tau_hat) and (c > 0):
            # Use private LimitedDomain on reduced domain
            instr_top = llm.topk_tokens(build_prompt_nonrag(question, prefix), kbar)
            yt = limited_domain_argmax(hist, instr_top, eps_token_rag, rng, null_token_id=eos_id)
            c -= 1
            tau_hat = tau + laplace(rng, scale=2.0 / max(eps_token_lap, 1e-12))
        else:
            # Fall back to non-RAG token without consuming DP budget
            yt = y_nonrag

        y_tokens.append(yt)
        prefix += llm.tok.decode([yt], skip_special_tokens=False)

        if yt == eos_id or c <= 0:
            break

    return llm.tok.decode(y_tokens, skip_special_tokens=True).strip()


# ------------------------------
# Evaluation: Match accuracy
# ------------------------------

def contains_any_answer(pred: str, answers: List[str]) -> bool:
    p = pred.lower().strip()
    for a in answers:
        if a and a.lower().strip() in p:
            return True
    return False


# ------------------------------
# Data loading (Natural Questions)
# ------------------------------

def _normalize_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "value", "answer", "title"):
            val = value.get(key)
            if isinstance(val, str):
                return val
        parts = [v for v in value.values() if isinstance(v, str)]
        if parts:
            return " ".join(parts)
    if isinstance(value, (list, tuple)):
        parts = [v for v in value if isinstance(v, str)]
        if parts:
            return " ".join(parts)
    if value is None:
        return ""
    return str(value)


def load_nq_pairs(n: int, seed: int) -> List[Tuple[str, List[str]]]:
    """Return a list of (question, [answers...]) pairs.
    We try `natural_questions` first; if unavailable, fall back to `natural_questions_open`.
    """
    ds = None
    try:
        ds = load_dataset("natural_questions", split="validation")
    except Exception:
        try:
            ds = load_dataset("natural_questions_open", split="validation")
        except Exception:
            raise RuntimeError(
                "Could not load Natural Questions. Try installing datasets>=2.14 and check internet access."
            )
    # Build pairs; different configs have different fields. Try common ones.
    pairs = []
    for rec in ds:
        q_raw = rec.get("question", rec.get("question_text", ""))
        q = _normalize_text(q_raw).strip()
        ans: List[str] = []
        if "answers" in rec and isinstance(rec["answers"], dict):
            raw_vals = rec["answers"].get("text", [])
            if not isinstance(raw_vals, (list, tuple)):
                raw_vals = [raw_vals]
            for a in raw_vals:
                norm = _normalize_text(a).strip()
                if norm:
                    ans.append(norm)
        elif "answers" in rec and isinstance(rec["answers"], list):
            for a in rec["answers"]:
                norm = _normalize_text(a).strip()
                if norm:
                    ans.append(norm)
        elif "answer" in rec:
            a = rec["answer"]
            items = a if isinstance(a, (list, tuple)) else [a]
            for item in items:
                norm = _normalize_text(item).strip()
                if norm:
                    ans.append(norm)
        elif "annotations" in rec:
            # some NQ variants
            ann = rec["annotations"][0] if isinstance(rec["annotations"], list) and rec["annotations"] else {}
            short = ann.get("short_answers", [])
            for s in short:
                norm = _normalize_text(s).strip()
                if norm:
                    ans.append(norm)
            long = ann.get("long_answer", {})
            norm_long = _normalize_text(long).strip()
            if norm_long:
                ans.append(norm_long)
        if q and ans:
            unique_ans = list(dict.fromkeys(ans))
            pairs.append((q, unique_ans))
        if len(pairs) >= n:
            break
    if not pairs:
        raise RuntimeError("No question/answer pairs found in NQ split.")
    # Deterministic subset
    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)
    return pairs[:n]


# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dpvote", "dpsparse"], default="dpsparse")
    ap.add_argument("--nq_count", type=int, default=100)
    ap.add_argument("--wiki_count", type=int, default=100000)
    ap.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", type=str, default="gpt2")
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--kbar", type=int, default=30)
    ap.add_argument("--tau_frac", type=float, default=0.5)
    ap.add_argument("--eps_total", type=float, default=20.0)
    ap.add_argument("--delta_total", type=float, default=1e-4)
    ap.add_argument("--eps_token", type=float, default=2.0)
    ap.add_argument("--delta_token", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=64, help="cap on total generated tokens")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load models
    print("[Load LLM]", args.llm)
    llm = LLM.load(args.llm, use_gpu=args.use_gpu)

    print("[Build Retriever]", args.encoder, "| wikipedia count:", args.wiki_count)
    retr = Retriever.build(args.encoder, wiki_count=args.wiki_count, seed=args.seed, use_gpu=args.use_gpu)

    # Load NQ
    print("[Load Natural Questions] count:", args.nq_count)
    qa_pairs = load_nq_pairs(args.nq_count, args.seed)

    # Run
    rng = np.random.RandomState(args.seed)
    correct = 0
    outputs = []

    for idx, (q, answers) in enumerate(qa_pairs, 1):
        if args.algo == "dpvote":
            pred = dpvote_rag(
                llm, retr, q, args.m, args.k, args.kbar,
                args.eps_total, args.delta_total,
                args.eps_token, args.delta_token,
                args.max_steps, rng,
            )
        else:
            pred = dpsparsevote_rag(
                llm, retr, q, args.m, args.k, args.kbar, args.tau_frac,
                args.eps_total, args.delta_total,
                args.eps_token, args.delta_token,
                args.max_steps, rng,
            )
        ok = contains_any_answer(pred, answers)
        correct += int(ok)
        outputs.append({"question": q, "pred": pred, "answers": answers, "match": bool(ok)})
        print(f"[{idx:03d}] match={ok} | pred={pred[:120]!r}")

    acc = correct / len(qa_pairs)
    print(f"\nAverage match accuracy over {len(qa_pairs)} questions: {acc:.3f}")

    # Save jsonl
    out_path = f"dp_rag_{args.algo}_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
