# NPO residual-confidence diagnostics. Writes sidecar JSONL + summary CSV only.

import argparse, csv, gc, json, os, random, sys
from collections import defaultdict; from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from repro import config as cfg

METHOD, DIAG_DIR = "npo_KL", "new_lmf/npo_residual_diagnostics"


def diag_path(a):
    name = (f"{METHOD}_{cfg.STRATEGY}_s={cfg.STEPWISE}_lr={a.lr}_beta={a.beta:g}"
            f"_rs={a.seed}_pos={not a.no_pos}_ff2={not a.no_ff2}.diagnostics.jsonl")
    return Path(a.diagnostics_dir) / a.dataset / a.short_model / name


def load_done(path):
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(f"{row['id']}_{row['step_idx']}")
    return done


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def scalar(x): return float(x.detach().cpu().float().item())


def masked_mean(x, mask):
    mask = mask.float(); return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_max(x, mask):
    vals = x.masked_select(mask).detach().float()
    return None if vals.numel() == 0 else float(vals.max().cpu().item())


def positive_delta_stats(token_delta, mask):
    import torch
    vals = token_delta.masked_select(mask).detach().float()
    pos = vals[vals > 0]
    if pos.numel() == 0:
        return 0.0, 0.0, 0.0
    pos = torch.sort(pos, descending=True).values
    total, k = pos.sum().clamp(min=1e-12), max(1, int(0.2 * pos.numel()))
    return scalar(pos[0] / total), scalar(pos[:k].sum() / total), float(pos.numel() / max(1, vals.numel()))


def compute_diagnostics(model, oracle, batch, beta, grad_thr, prob_thr):
    import torch
    import torch.nn.functional as F
    was_training = model.training
    model.eval(); oracle.eval()
    try:
        with torch.no_grad():
            (ids, labels, attn), retain = batch
            cur = model(ids, labels=labels, attention_mask=attn)
            ref = oracle(ids, labels=labels, attention_mask=attn)
            cur_logits = cur.logits[:, :-1, :].contiguous().float()
            ref_logits = ref.logits[:, :-1, :].contiguous().float()
            y = labels[:, 1:].contiguous()
            mask = y.ne(-100)
            safe_y = y.masked_fill(~mask, 0)

            cur_lp, ref_lp = F.log_softmax(cur_logits, -1), F.log_softmax(ref_logits, -1)
            cur_nll = -cur_lp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
            ref_nll = -ref_lp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
            token_delta = cur_nll - ref_nll
            delta = (token_delta * mask.float()).sum()
            grad = 2.0 * torch.sigmoid(-beta * delta)
            probs = cur_lp.exp()
            margin = cur_logits.max(-1).values - cur_logits.mean(-1)
            max_prob = probs.max(-1).values
            entropy = -(probs * cur_lp).sum(-1)
            target_prob = probs.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)

            rids, rlabels, rattn = retain
            rref = oracle(rids, labels=rlabels, attention_mask=rattn)
            rcur = model(rids, labels=rlabels, attention_mask=rattn)
            rref_full_lp = F.log_softmax(rref.logits.contiguous().float(), -1)
            rcur_full_lp = F.log_softmax(rcur.logits.contiguous().float(), -1)
            retain_kl_train_style = F.kl_div(
                rcur_full_lp.view(-1, rcur_full_lp.shape[-1]),
                rref_full_lp.view(-1, rref_full_lp.shape[-1]),
                reduction="batchmean",
                log_target=True,
            )
            rref_lp = rref_full_lp[:, :-1, :]
            rcur_lp = rcur_full_lp[:, :-1, :]
            rmask = rlabels[:, 1:].contiguous().ne(-100)
            retain_kl_masked = F.kl_div(rcur_lp, rref_lp, reduction="none", log_target=True).sum(-1)

            top1, top20, pos_frac = positive_delta_stats(token_delta, mask)
            grad_s, max_prob_m = scalar(grad), scalar(masked_mean(max_prob, mask))
            saturated, high_conf = grad_s < grad_thr, max_prob_m > prob_thr
            return {"forget_valid_tokens": int(mask.sum().detach().cpu().item()), "npo_delta": scalar(delta),
                    "npo_success_sigmoid": scalar(torch.sigmoid(beta * delta)), "npo_grad_factor": grad_s,
                    "npo_saturated": int(saturated), "residual_high_conf": int(high_conf),
                    "saturated_and_high_conf": int(saturated and high_conf),
                    "forget_margin_mean": scalar(masked_mean(margin, mask)), "forget_margin_max": masked_max(margin, mask),
                    "forget_max_softmax_prob_mean": max_prob_m, "forget_max_softmax_prob_max": masked_max(max_prob, mask),
                    "forget_entropy_mean": scalar(masked_mean(entropy, mask)),
                    "target_token_prob_mean": scalar(masked_mean(target_prob, mask)),
                    "target_token_nll_mean": scalar(masked_mean(cur_nll, mask)),
                    "token_delta_mean": scalar(masked_mean(token_delta, mask)),
                    "positive_token_delta_fraction": pos_frac, "positive_delta_top1_share": top1,
                    "positive_delta_top20pct_share": top20, "retain_valid_tokens": int(rmask.sum().detach().cpu().item()),
                    "retain_kl": scalar(retain_kl_train_style), "retain_kl_train_style": scalar(retain_kl_train_style),
                    "retain_kl_masked": scalar(masked_mean(retain_kl_masked, rmask))}
    finally:
        model.train(was_training)


def train_one_target(model_id, tokenizer, a, target, step_idx, cots_train):
    import torch, unlearn; from data import FRCollator, cot_to_otfd
    from torch.utils.data import DataLoader; from transformers import AutoModelForCausalLM as CLM
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    model, oracle = CLM.from_pretrained(model_id, **kw), CLM.from_pretrained(model_id, **kw)
    collator = FRCollator(tokenizer, device=model.device)
    dataset = cot_to_otfd(target, cots_train, tokenizer, strategy=cfg.STRATEGY, stepwise=cfg.STEPWISE, step_idx=step_idx, pos=not a.no_pos)
    n_targets = dataset.num_targets()
    if n_targets <= 2:
        print(f"skip {target['id']}_{step_idx}: too few targets ({n_targets})")
        del collator, dataset, model, oracle; gc.collect(); torch.cuda.empty_cache(); return None
    if not a.no_ff2:
        for name, param in model.named_parameters(): param.requires_grad = "mlp.down_proj.weight" in name
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sch = unlearn.get_linear_schedule_with_warmup(opt, 0, a.epochs * len(dataset))
    diag_args = (a.beta, a.grad_threshold, a.max_prob_threshold)
    diagnostics = {0: compute_diagnostics(model, oracle, collator([dataset[0]]), *diag_args)}
    for epoch in range(a.epochs):
        model.train(); opt.zero_grad()
        for batch in loader:
            loss = unlearn.compute_loss(model, oracle, batch, loss_type=METHOD, beta=a.beta, KL_coeff=a.rt_lambda)
            loss.backward(); opt.step(); sch.step(); opt.zero_grad()
        diagnostics[epoch + 1] = compute_diagnostics(model, oracle, collator([dataset[0]]), *diag_args)
    del loader, opt, sch, collator, dataset, model, oracle
    gc.collect(); torch.cuda.empty_cache()
    return diagnostics


def run(a):
    import dataload; from huggingface_hub import login; from transformers import AutoTokenizer as TOK
    from repro.local_datasets import LOCAL_DATASETS
    dataload.DATASETS.update(LOCAL_DATASETS)
    from data import load_or_generate_dataset_cots
    from util import set_random_seed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    login(os.environ["HF_TOKEN"]) if os.environ.get("HF_TOKEN") else print("[warn] HF_TOKEN is not set; gated models must be cached locally.")
    set_random_seed(a.seed)
    model_id = cfg.MODELS[a.short_model]
    tok = TOK.from_pretrained(model_id)
    tok.pad_token = tok.unk_token if "Phi" in model_id else tok.eos_token
    cots = load_or_generate_dataset_cots(model_id, tok, a.dataset, a.seed, cfg.TEMPERATURE,
                                        force_generate=a.new_cot, sentencize=(cfg.STRATEGY == "sentencize"), atomic=False)
    random.shuffle(cots)
    cots_train, out = cots[:-a.n_verify], diag_path(a)
    done = load_done(out)
    print(f"diagnostics path: {out}")
    print(f"n_unlearn={a.n_unlearn}, n_verify={a.n_verify}, epochs={a.epochs}, done={len(done)}")
    for target in cots_train[:a.n_unlearn]:
        for step_idx in range(len(target["segmented_cot"]) if cfg.STEPWISE else 1):
            key = f"{target['id']}_{step_idx}"
            if key in done:
                continue
            diagnostics = train_one_target(model_id, tok, a, target, step_idx, cots_train)
            if diagnostics is None:
                continue
            append_jsonl(out, {"id": target["id"], "question": target["question"], "step_idx": step_idx,
                                "dataset": a.dataset, "short_model": a.short_model, "method": METHOD, "lr": a.lr,
                                "beta": a.beta, "rt_lambda": a.rt_lambda, "grad_threshold": a.grad_threshold,
                                "max_prob_threshold": a.max_prob_threshold, "pos": not a.no_pos, "ff2": not a.no_ff2,
                                "cot_step": target["segmented_cot"][step_idx], "diagnostics_results": diagnostics})
            done.add(key)
    summarize(out)


def summarize(jsonl_path, csv_path=None):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"No diagnostics file found: {jsonl_path}"); return None
    groups, metrics = {}, set()
    for line in jsonl_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for epoch, vals in row["diagnostics_results"].items():
            key = (row["dataset"], row["short_model"], row["lr"], row["beta"], row["rt_lambda"], int(epoch))
            group = groups.setdefault(key, {"n": 0, "sum": defaultdict(float), "cnt": defaultdict(int)})
            group["n"] += 1
            for name, value in vals.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics.add(name); group["sum"][name] += float(value); group["cnt"][name] += 1
    metric_cols = sorted(metrics)
    fields = ["dataset", "short_model", "lr", "beta", "rt_lambda", "epoch", "n_records"] + metric_cols
    csv_path = Path(csv_path) if csv_path else jsonl_path.with_suffix(".summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for key in sorted(groups, key=lambda x: (x[0], x[1], x[5])):
            dataset, short_model, lr, beta, rt_lambda, epoch = key
            g = groups[key]; row = {"dataset": dataset, "short_model": short_model, "lr": lr, "beta": beta, "rt_lambda": rt_lambda, "epoch": epoch, "n_records": g["n"]}
            row.update({m: g["sum"][m] / g["cnt"][m] if g["cnt"].get(m) else "" for m in metric_cols})
            writer.writerow(row)
    print(f"Wrote diagnostics summary: {csv_path}")
    return csv_path


def parse_args():
    p = argparse.ArgumentParser()
    for name, kwargs in [
        ("short_model", {"choices": list(cfg.MODELS)}), ("dataset", {"choices": cfg.DATASETS}),
        ("lr", {"type": float}), ("beta", {"type": float, "default": 0.1}),
        ("rt_lambda", {"type": float, "default": 1.0}), ("n_unlearn", {"type": int, "default": 5}),
        ("n_verify", {"type": int, "default": cfg.N_VERIFY}), ("epochs", {"type": int, "default": cfg.EPOCHS}),
        ("seed", {"type": int, "default": cfg.SEED}), ("grad_threshold", {"type": float, "default": 0.05}),
        ("max_prob_threshold", {"type": float, "default": 0.5}), ("diagnostics_dir", {"default": DIAG_DIR}),
        ("summarize", {}), ("summary_csv", {}),
    ]:
        p.add_argument(f"--{name}", **kwargs)
    for flag in ("new_cot", "no_pos", "no_ff2", "dry_run"): p.add_argument(f"--{flag}", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.summarize: summarize(a.summarize, a.summary_csv); return
    missing = [x for x in ("short_model", "dataset", "lr") if getattr(a, x) is None]
    if missing: raise SystemExit(f"Missing required run arguments: {', '.join('--' + m for m in missing)}")
    if not a.dry_run: run(a); return
    path = diag_path(a)
    print(f"diagnostics path: {path}"); print(f"summary path: {path.with_suffix('.summary.csv')}")
    print(f"default size: n_unlearn={a.n_unlearn}, n_verify={a.n_verify}, epochs={a.epochs}, beta={a.beta}")


if __name__ == "__main__":
    main()
