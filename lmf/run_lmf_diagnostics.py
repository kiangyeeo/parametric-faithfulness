"""Small sidecar diagnostics runner for LMF+KL.

Default scope is intentionally small: first 5 shuffled training examples,
all CoT steps for those examples, 20 held out as in the main split, 5 epochs.
"""

import argparse, csv, gc, json, os, random, sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro import config as cfg


METHOD = "lmf_KL"
DIAG_DIR = "lmf/diagnostics"


def diag_path(args):
    name = (
        f"{METHOD}_{cfg.STRATEGY}_s={cfg.STEPWISE}_lr={args.lr}{args.log_suffix or ''}"
        f"_rs={args.seed}_pos={not args.no_pos}_ff2={not args.no_ff2}.diagnostics.jsonl"
    )
    return Path(args.diagnostics_dir) / args.dataset / args.short_model / name


def load_done(path):
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done.add(f"{row['id']}_{row['step_idx']}")
    return done


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def scalar(x):
    return float(x.detach().cpu().float().item())


def masked_mean(values, mask):
    mask = mask.float()
    return (values * mask).sum() / mask.sum().clamp(min=1)


def masked_summary(values, mask):
    selected = values.masked_select(mask).detach().float()
    if selected.numel() == 0:
        return 0, None, None
    return int(selected.numel()), float(selected.mean().cpu().item()), float(selected.max().cpu().item())


def compute_diagnostics(model, oracle, batch):
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    oracle.eval()
    try:
        with torch.no_grad():
            (forget_ids, forget_labels, forget_mask_ids), retain_inputs = batch
            out = model(forget_ids, labels=forget_labels, attention_mask=forget_mask_ids)
            logits = out.logits[:, :-1, :].contiguous().float()
            fmask = forget_labels[:, 1:].contiguous().ne(-100)

            margin = logits.max(dim=-1).values - logits.mean(dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            max_prob = probs.max(dim=-1).values
            entropy = -(probs * log_probs).sum(dim=-1)

            retain_ids, retain_labels, retain_attention = retain_inputs
            oracle_out = oracle(retain_ids, labels=retain_labels, attention_mask=retain_attention)
            current_out = model(retain_ids, labels=retain_labels, attention_mask=retain_attention)
            oracle_logits = oracle_out.logits[:, :-1, :].contiguous().float()
            current_logits = current_out.logits[:, :-1, :].contiguous().float()
            retain_y = retain_labels[:, 1:].contiguous()
            rmask = retain_y.ne(-100)

            oracle_lp = F.log_softmax(oracle_logits, dim=-1)
            current_lp = F.log_softmax(current_logits, dim=-1)
            retain_kl = F.kl_div(current_lp, oracle_lp, reduction="none", log_target=True).sum(dim=-1)
            vocab = current_logits.size(-1)
            current_ce = F.cross_entropy(
                current_logits.view(-1, vocab), retain_y.view(-1), ignore_index=-100, reduction="none"
            ).view(retain_y.shape)
            oracle_ce = F.cross_entropy(
                oracle_logits.view(-1, vocab), retain_y.view(-1), ignore_index=-100, reduction="none"
            ).view(retain_y.shape)

            n_forget, margin_mean, margin_max = masked_summary(margin, fmask)
            _, max_prob_mean, max_prob_max = masked_summary(max_prob, fmask)
            _, entropy_mean, entropy_max = masked_summary(entropy, fmask)
            return {
                "forget_valid_tokens": n_forget,
                "forget_lmf_loss": scalar(masked_mean(margin.square(), fmask)),
                "forget_margin_mean": margin_mean,
                "forget_margin_max": margin_max,
                "forget_max_softmax_prob_mean": max_prob_mean,
                "forget_max_softmax_prob_max": max_prob_max,
                "forget_entropy_mean": entropy_mean,
                "forget_entropy_max": entropy_max,
                "retain_valid_tokens": int(rmask.sum().detach().cpu().item()),
                "retain_kl": scalar(masked_mean(retain_kl, rmask)),
                "retain_ce_current": scalar(masked_mean(current_ce, rmask)),
                "retain_ce_oracle": scalar(masked_mean(oracle_ce, rmask)),
                "retain_ce_delta_current_minus_oracle": scalar(masked_mean(current_ce - oracle_ce, rmask)),
            }
    finally:
        model.train(was_training)


def train_one_target(model_id, tokenizer, args, target, step_idx, cots_train):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM as CLM
    import unlearn
    from data import FRCollator, cot_to_otfd

    model = CLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=False, device_map="auto")
    oracle = CLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=False, device_map="auto")
    collator = FRCollator(tokenizer, device=model.device)
    dataset = cot_to_otfd(
        target, cots_train, tokenizer, strategy=cfg.STRATEGY,
        stepwise=cfg.STEPWISE, step_idx=step_idx, pos=not args.no_pos
    )

    n_targets = dataset.num_targets()
    if n_targets <= 2:
        print(f"skip {target['id']}_{step_idx}: too few targets ({n_targets})")
        del collator, dataset, model, oracle
        gc.collect()
        torch.cuda.empty_cache()
        return None

    if not args.no_ff2:
        for name, param in model.named_parameters():
            param.requires_grad = "mlp.down_proj.weight" in name

    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = unlearn.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=args.epochs * len(dataset)
    )

    diagnostics = {0: compute_diagnostics(model, oracle, collator([dataset[0]]))}
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        for batch in loader:
            loss = unlearn.compute_loss(model, oracle, batch, loss_type=METHOD, KL_coeff=args.rt_lambda)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        diagnostics[epoch + 1] = compute_diagnostics(model, oracle, collator([dataset[0]]))

    del loader, optimizer, scheduler, collator, dataset, model, oracle
    gc.collect()
    torch.cuda.empty_cache()
    return diagnostics


def run(args):
    import dataload
    from huggingface_hub import login
    from transformers import AutoTokenizer as TOK
    from repro.local_datasets import LOCAL_DATASETS

    dataload.DATASETS.update(LOCAL_DATASETS)
    from data import load_or_generate_dataset_cots
    from util import set_random_seed

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
    else:
        print("[warn] HF_TOKEN is not set; gated models must be cached locally.")

    set_random_seed(args.seed)
    model_id = cfg.MODELS[args.short_model]
    tokenizer = TOK.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.unk_token if "Phi" in model_id else tokenizer.eos_token
    cot_data = load_or_generate_dataset_cots(
        model_id=model_id, tokenizer=tokenizer, dataset_id=args.dataset,
        force_generate=args.new_cot, sentencize=(cfg.STRATEGY == "sentencize"),
        temperature=cfg.TEMPERATURE, seed=args.seed, atomic=False
    )
    random.shuffle(cot_data)
    cots_train = cot_data[:-args.n_verify]
    out = diag_path(args)
    done = load_done(out)
    print(f"diagnostics path: {out}")
    print(f"n_unlearn={args.n_unlearn}, n_verify={args.n_verify}, done={len(done)}")

    for target in cots_train[:args.n_unlearn]:
        for step_idx in range(len(target["segmented_cot"]) if cfg.STEPWISE else 1):
            key = f"{target['id']}_{step_idx}"
            if key in done:
                continue
            diagnostics = train_one_target(model_id, tokenizer, args, target, step_idx, cots_train)
            if diagnostics is None:
                continue
            append_jsonl(out, {
                "id": target["id"], "question": target["question"], "step_idx": step_idx,
                "dataset": args.dataset, "short_model": args.short_model, "method": METHOD,
                "lr": args.lr, "rt_lambda": args.rt_lambda, "pos": not args.no_pos,
                "ff2": not args.no_ff2, "cot_step": target["segmented_cot"][step_idx],
                "diagnostics_results": diagnostics,
            })
            done.add(key)
    summarize(out)


def summarize(jsonl_path, csv_path=None):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"No diagnostics file found: {jsonl_path}")
        return None
    csv_path = Path(csv_path) if csv_path else jsonl_path.with_suffix(".summary.csv")
    groups = {}
    metrics = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for epoch, vals in row["diagnostics_results"].items():
                key = (row["dataset"], row["short_model"], row["lr"], row["rt_lambda"], int(epoch))
                group = groups.setdefault(key, {"n": 0, "sum": defaultdict(float), "cnt": defaultdict(int)})
                group["n"] += 1
                for name, value in vals.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics.add(name)
                        group["sum"][name] += float(value)
                        group["cnt"][name] += 1
    metric_cols = sorted(metrics)
    fields = ["dataset", "short_model", "lr", "rt_lambda", "epoch", "n_records"] + metric_cols
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups, key=lambda x: (x[0], x[1], x[4])):
            dataset, short_model, lr, rt_lambda, epoch = key
            group = groups[key]
            row = {"dataset": dataset, "short_model": short_model, "lr": lr,
                    "rt_lambda": rt_lambda, "epoch": epoch, "n_records": group["n"]}
            row.update({m: group["sum"][m] / group["cnt"][m] if group["cnt"].get(m) else "" for m in metric_cols})
            writer.writerow(row)
    print(f"Wrote diagnostics summary: {csv_path}")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    for name, kwargs in [
        ("short_model", {"choices": list(cfg.MODELS)}), ("dataset", {"choices": cfg.DATASETS}),
        ("lr", {"type": float}), ("rt_lambda", {"type": float, "default": 1.0}),
        ("n_unlearn", {"type": int, "default": 5}), ("n_verify", {"type": int, "default": cfg.N_VERIFY}),
        ("epochs", {"type": int, "default": cfg.EPOCHS}), ("seed", {"type": int, "default": cfg.SEED}),
        ("diagnostics_dir", {"default": DIAG_DIR}), ("log_suffix", {}), ("summarize", {}), ("summary_csv", {}),
    ]:
        parser.add_argument(f"--{name}", **kwargs)
    for flag in ("new_cot", "no_pos", "no_ff2", "dry_run"):
        parser.add_argument(f"--{flag}", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.summarize:
        summarize(args.summarize, args.summary_csv)
        return
    missing = [x for x in ("short_model", "dataset", "lr") if getattr(args, x) is None]
    if missing:
        raise SystemExit(f"Missing required run arguments: {', '.join('--' + m for m in missing)}")
    if args.dry_run:
        path = diag_path(args)
        print(f"diagnostics path: {path}")
        print(f"summary path: {path.with_suffix('.summary.csv')}")
        print(f"default size: n_unlearn={args.n_unlearn}, n_verify={args.n_verify}, epochs={args.epochs}")
        return
    run(args)


if __name__ == "__main__":
    main()
