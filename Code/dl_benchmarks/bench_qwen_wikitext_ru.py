"""
Benchmark the scalable custom PyTorch optimizers on Qwen/Qwen2.5-0.5B for
Russian-language Supervised Fine-Tuning (SFT) only.

Stage
-----
This script runs ONLY the SFT (next-token cross-entropy) stage on raw
Russian text.  No instruction tuning, RLHF, or preference optimisation is
performed.

Dataset
-------
Russian WikiText-2-style corpus is loaded directly from HuggingFace
``datasets`` (default: a slice of ``wikimedia/wikipedia`` Russian dump,
mirroring the WikiText-2 chunking protocol).  Alternative HF datasets can
be selected with ``--hf-dataset``/``--hf-config``/``--hf-split``.
A local text file (one paragraph per line) is also supported via
``--data-file``.

Optimizers
----------
Qwen2.5-0.5B has ~500 M parameters, so Hessian-based methods are excluded.
We benchmark first-order and limited-memory methods only.

Demo
----
Before training begins we print a generation sample from the *untrained*
(pre-finetune) Qwen2.5-0.5B so you can compare it qualitatively to the
post-training generations.

Run:
    cd Code
    python -m dl_benchmarks.bench_qwen_wikitext_ru --max-steps 200 --only Adam
"""

from __future__ import annotations

import os

# GPU configuration -- must be set before torch is imported.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import json
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from dl_benchmarks.runner import benchmark_suite, OptimizationLogger, build_optimizer  # noqa: E402
from utils.logging.optimizer_logger import MultiRunLogger  # noqa: E402


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEMO_PROMPT_RU = "Россия — это страна, в которой"


# ---------------------------------------------------------------------------
# Data: Russian WikiText-2 style stream
# ---------------------------------------------------------------------------

class CausalLMTextDataset(Dataset):
    """
    Tokenises a list of paragraphs into fixed-length blocks for causal LM.

    The standard WikiText-2 protocol is mirrored: concatenate all text, then
    chunk into ``block_size`` token windows.  ``labels`` are identical to
    ``input_ids`` since the HuggingFace ``forward(..., labels=...)`` API
    shifts internally.
    """

    def __init__(self, paragraphs, tokenizer, block_size: int = 512):
        text = "\n\n".join(paragraphs)
        # Drop trailing extra-long encodings via truncation iteratively
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        # Chunk
        n_blocks = len(ids) // block_size
        if n_blocks == 0:
            raise ValueError(
                f"Corpus too short ({len(ids)} tokens) for block_size={block_size}"
            )
        ids = ids[: n_blocks * block_size]
        self.input_ids = torch.tensor(ids, dtype=torch.long).view(n_blocks, block_size)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int):
        x = self.input_ids[idx]
        return x, x.clone()  # labels == input_ids


def load_paragraphs(
    data_file: Optional[str],
    hf_dataset: str = "wikimedia/wikipedia",
    hf_config: Optional[str] = "20231101.ru",
    hf_split: str = "train",
    text_column: str = "text",
    n_articles: int = 500,
    cache_dir: Optional[str] = None,
):
    """
    Load a list of paragraphs for SFT.

    Priority:
      1. ``--data-file <path>`` (one paragraph per line) if provided.
      2. HuggingFace ``datasets`` (streaming) -- defaults to a small slice
         of the Russian Wikipedia dump.
    """
    if data_file is not None:
        with open(data_file, encoding="utf-8") as f:
            paragraphs = [line.strip() for line in f if line.strip()]
        return paragraphs

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install `datasets` (`pip install datasets`) or pass --data-file."
        ) from e

    print(f"Loading {hf_dataset}"
          f"{f' ({hf_config})' if hf_config else ''}"
          f" split={hf_split} from HuggingFace...")
    if hf_config:
        ds = load_dataset(hf_dataset, hf_config, split=hf_split,
                          streaming=True, cache_dir=cache_dir)
    else:
        ds = load_dataset(hf_dataset, split=hf_split,
                          streaming=True, cache_dir=cache_dir)
    paragraphs = []
    for i, row in enumerate(ds):
        if i >= n_articles:
            break
        text = (row.get(text_column) or "").strip()
        if not text:
            continue
        # Cap each entry to keep memory bounded; chunking happens later.
        paragraphs.append(text[:2000])
    return paragraphs


# ---------------------------------------------------------------------------
# Demo generation from the untrained model
# ---------------------------------------------------------------------------

def demo_generation(model, tokenizer, device, prompt: str = DEMO_PROMPT_RU,
                    max_new_tokens: int = 60):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    model.train()
    return text


# ---------------------------------------------------------------------------
# Custom training loop -- HF model returns a dict, so we wrap accordingly
# ---------------------------------------------------------------------------

def train_qwen_with_optimizer(
    model,
    tokenizer,
    train_loader: DataLoader,
    optimizer_name: str,
    optimizer_kwargs,
    *,
    max_steps: int,
    device: torch.device,
    log_every: int = 10,
    grad_clip: float = 1.0,
    eval_prompt: Optional[str] = None,
) -> OptimizationLogger:
    """Run one optimizer end-to-end on Qwen and return its logger."""
    from dl_benchmarks.runner import cycle, peak_memory_tracker, grad_norm

    model.train()
    optimizer = build_optimizer(optimizer_name, model.parameters(), **optimizer_kwargs)
    logger = OptimizationLogger(optimizer_name=optimizer_name)
    data_iter = cycle(train_loader)

    with peak_memory_tracker(device) as get_peak_kb:
        for step in range(max_steps):
            input_ids, labels = next(data_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % log_every == 0 or step == max_steps - 1:
                logger.log(step, {
                    "loss": float(loss.detach().item()),
                    "ppl": float(torch.exp(loss.detach()).item()),
                    "grad_norm": grad_norm(model.parameters()),
                })

        logger.heap_peak_kb = float(get_peak_kb())

    if eval_prompt is not None:
        gen = demo_generation(model, tokenizer, device, eval_prompt, max_new_tokens=60)
        print(f"\n[{optimizer_name}] post-training generation:")
        print(gen)

    return logger


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

def build_suite():
    """Only scalable methods -- Qwen has ~500 M params."""
    return [
        ("SGD",             dict(lr=1e-4)),
        ("MomentumSGD",     dict(lr=1e-4, momentum=0.9)),
        ("Nesterov",        dict(lr=1e-4, momentum=0.9)),
        ("AdaGrad",         dict(lr=1e-3)),
        ("RMSProp",         dict(lr=5e-5)),
        ("Adam",            dict(lr=5e-5)),
        # L-BFGS family is feasible (m*n memory) but very slow / fragile on LLMs.
        # Included for completeness; expect slower wall-clock per step.
        ("StochasticLBFGS", dict(lr=1e-4, history_size=5,
                                  batch_size=4, n_samples=10_000)),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-0.5B SFT-only optimizer benchmark on Russian WikiText"
    )
    parser.add_argument("--data-file", default=None,
                        help="Optional local text file (one paragraph per line). "
                             "If omitted, the HF dataset is used.")
    parser.add_argument("--hf-dataset", default="wikimedia/wikipedia",
                        help="HuggingFace dataset name (default: Russian Wikipedia)")
    parser.add_argument("--hf-config", default="20231101.ru",
                        help="HuggingFace dataset config / language")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace datasets cache directory")
    parser.add_argument("--n-articles", default=500, type=int,
                        help="Number of paragraphs/articles to stream from HF")
    parser.add_argument("--block-size", default=256, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--max-steps", default=200, type=int)
    parser.add_argument("--log-every", default=10, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--save", default="dl_benchmarks/results_qwen_wikitext_ru.json")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only this subset of optimizer short names")
    parser.add_argument("--skip-untrained-demo", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    print(f"Device: {device} | dtype: {dtype}")

    # -----------------------------------------------------------------------
    # Tokenizer + dataset
    # -----------------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    paragraphs = load_paragraphs(
        args.data_file,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        text_column=args.text_column,
        n_articles=args.n_articles,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded {len(paragraphs)} paragraphs.")
    dataset = CausalLMTextDataset(paragraphs, tokenizer, block_size=args.block_size)
    print(f"Built {len(dataset)} blocks of {args.block_size} tokens each.")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # -----------------------------------------------------------------------
    # Untrained demo
    # -----------------------------------------------------------------------
    if not args.skip_untrained_demo:
        print("\n=== Demo: generation from the UNTRAINED (pre-finetune) model ===")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
        untrained_text = demo_generation(base_model, tokenizer, device, DEMO_PROMPT_RU, max_new_tokens=80)
        print(f"Prompt:  {DEMO_PROMPT_RU}")
        print(f"Output:  {untrained_text}")
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Suite
    # -----------------------------------------------------------------------
    suite = build_suite()
    if args.only:
        suite = [(n, kw) for n, kw in suite if n in args.only]
        if not suite:
            raise SystemExit(f"--only filtered out all suites. Valid names: "
                             f"{[s[0] for s in build_suite()]}")

    multi = MultiRunLogger()
    for name, kwargs in suite:
        print(f"\n=== Fine-tuning Qwen2.5-0.5B with {name} (kwargs={kwargs}) ===")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
        if device.type == "cuda":
            model.gradient_checkpointing_enable()

        logger = train_qwen_with_optimizer(
            model, tokenizer, train_loader,
            optimizer_name=name, optimizer_kwargs=kwargs,
            max_steps=args.max_steps, device=device,
            log_every=args.log_every, grad_clip=1.0,
            eval_prompt=DEMO_PROMPT_RU,
        )
        multi._runs[name] = logger  # type: ignore[attr-defined]
        print(f"   -> {logger}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Final comparison ===")
    multi.print_comparison()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    payload = {
        "device": str(device),
        "dtype": args.dtype,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "runs": {
            name: {
                "summary": multi.get_run(name).summary(),
                "history": multi.get_run(name).history,
            }
            for name in multi.run_names
        },
    }
    with open(args.save, "w") as f:
        json.dump(payload, f, default=float, indent=2)
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
