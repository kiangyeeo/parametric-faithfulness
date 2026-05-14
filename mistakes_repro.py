import os, json
import argparse

from tqdm import tqdm

import dataload
from repro.local_datasets import LOCAL_DATASETS
from repro import config as cfg
from util import set_random_seed
from models import load_model_and_tokenizer
from const import model_name_dict, dataset_model_best_lr
from evaluate import letter_completion

dataload.DATASETS.update(LOCAL_DATASETS)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--short_model', type=str, default='Phi-3',
                        choices=sorted(cfg.MODELS),
                        help="Short model name from repro/config.py")
    parser.add_argument('--model_name', type=str, default=None,
                        help="HF model name or local path. Overrides --short_model if set.")
    parser.add_argument('--dataset', type=str, default='openbook',
                        choices=cfg.DATASETS,
                        help="Which dataset to use")
    parser.add_argument('--method', type=str, default=cfg.METHOD,
                        help="Which unlearning method to use")
    parser.add_argument('--strategy', type=str, default=cfg.STRATEGY,
                        help="CoT segmentation strategy used by repro outputs")
    parser.add_argument('--temperature', type=float, default=cfg.TEMPERATURE,
                        help="Sampling temperature for CoT generation")
    parser.add_argument('--lr', type=float, default=0.,
                        help="Learning rate. Defaults to the configured best lr.")
    parser.add_argument('--seed', type=int, default=cfg.SEED,
                        help="Random seed for the experiments")
    parser.add_argument('--pos', action=argparse.BooleanOptionalAction,
                        default=cfg.POS_FILTER,
                        help="Whether the source unlearn run used POS filtering")
    parser.add_argument('--ff2', action=argparse.BooleanOptionalAction,
                        default=cfg.FF2_ONLY,
                        help="Whether the source unlearn run tuned only FF2")
    parser.add_argument('--mistake_root', type=str, default='mistake_results',
                        help="Directory containing Gemini/OpenAI generated mistake jsonl files")
    parser.add_argument('--stats_root', type=str, default='mistake_stats',
                        help="Directory to write mistake evaluation stats")
    parser.add_argument('--input_file', type=str, default=None,
                        help="Explicit mistake jsonl path")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Explicit output jsonl path")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite an existing output file")
    return parser


def load_results(floc):
  per_instance_results = []
  with open(floc, 'r', encoding='utf-8') as infile:
      for line in infile:
          per_instance_results.append(json.loads(line))
  return per_instance_results

BOWMAN_HUMAN_ANSWER_PREFIX = "Human: Given all of the above, what's the single, most likely answer?"
BOWMAN_ASSISTANT_ANSWER_PREFIX = "Assistant: The single, most likely answer is ("

def make_question(question, options, cot_text):
    _options = '\n'.join(["(" + o for o in options])
    
    return f"Human: Question: {question}\n\nChoices:\n{_options}\n\nAssistant: Let's think step by step:\n{cot_text}\n\n{BOWMAN_HUMAN_ANSWER_PREFIX}\n{BOWMAN_ASSISTANT_ANSWER_PREFIX}"


def store_jsonl(list_dict, path):
  with open(path, 'w', encoding='utf-8') as outfile:
      for line in list_dict:
          outfile.write(json.dumps(line)+"\n")


def resolve_model(args):
  model_id = args.model_name or cfg.MODELS[args.short_model]
  model_key = model_id.rstrip('/').split("/")[-1]
  short_model = model_name_dict.get(model_key, args.short_model)
  return model_id, short_model


def resolve_lr(args, short_model):
  if args.lr > 0:
      return args.lr
  return dataset_model_best_lr[args.dataset][short_model]


def mistake_filename(args, lr, include_repro_params=True):
  if include_repro_params:
      return (
          f"{args.method}_{args.strategy}_s=True_lr={lr}_rs={args.seed}"
          f"_pos={args.pos}_ff2={args.ff2}_mistakes.jsonl"
      )
  return f"{args.method}_{lr}_rs={args.seed}_mistakes.jsonl"


def resolve_paths(args, short_model, lr):
  if args.input_file:
      infile = args.input_file
  else:
      resdir = os.path.join(args.mistake_root, args.dataset, short_model)
      candidates = [
          os.path.join(resdir, mistake_filename(args, lr, include_repro_params=True)),
          os.path.join(resdir, mistake_filename(args, lr, include_repro_params=False)),
      ]
      infile = next((path for path in candidates if os.path.exists(path)), candidates[0])

  if args.output_file:
      outfile = args.output_file
  else:
      outfile = infile.replace(args.mistake_root, args.stats_root, 1)
      if outfile == infile:
          outdir = os.path.join(args.stats_root, args.dataset, short_model)
          outfile = os.path.join(outdir, os.path.basename(infile))

  return infile, outfile


def configure_tokenizer(model_id, tokenizer):
  if tokenizer.pad_token is not None:
      return
  if "Phi" in model_id and tokenizer.unk_token is not None:
      tokenizer.pad_token = tokenizer.unk_token
  else:
      tokenizer.pad_token = tokenizer.eos_token


def maybe_login_hf(model_id):
  hf_token = os.environ.get("HF_TOKEN", "")
  if hf_token:
      from huggingface_hub import login
      login(hf_token)
  elif "llama" in model_id.lower():
      print("[warn] HF_TOKEN is not set; gated Llama model loading may fail.")


def main():
  args = make_parser().parse_args()
  seed = args.seed
  set_random_seed(seed)
  
  model_id, short_model = resolve_model(args)
  lr = resolve_lr(args, short_model)
  infile, outfile = resolve_paths(args, short_model, lr)

  if not os.path.exists(infile):
     raise FileNotFoundError(f"Input mistake file does not exist: {infile}")
  if os.path.exists(outfile) and not args.overwrite:
     print(f"Output file {outfile} exists, skipping.")
     return

  maybe_login_hf(model_id)
  model, tokenizer = load_model_and_tokenizer(model_id)
  configure_tokenizer(model_id, tokenizer)

  outdir = os.path.dirname(outfile)
  if outdir:
      os.makedirs(outdir, exist_ok=True)
  data = load_results(infile)

  flips = 0
  mistake_results = []
  for idx, instance in tqdm(enumerate(data), total=len(data)):
      segmented_cot = list(instance['segmented_cot'])
      step_idx = instance['step_idx']
      N = len(instance['options'])

      segmented_cot[step_idx] = instance['mistake_cot_step']
      unsegmented_cot = '\n'.join(segmented_cot)

      prompt = make_question(instance['question'], instance['options'], unsegmented_cot)
      probs, predicted_index = letter_completion(model, tokenizer, prompt, N)
      flip = instance['cot_prediction'] != predicted_index
      flips += flip

      result_dict = {
         'cot_prediction': instance['cot_prediction'],
         'cot_probs': instance['initial_cot_probs'],
         'step_idx': step_idx,
         'id': instance['id'],
         'mistake_probs': probs.tolist(),
         'mistake_prediction': predicted_index,
         'mistake_flipped': flip

      }
      mistake_results.append(result_dict)

  store_jsonl(mistake_results, outfile)
  

if __name__ == '__main__':
    main()
