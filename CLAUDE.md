# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase (MSc thesis work) for **evaluating the explanatory properties of LLM-generated explanations**. It is a collection of standalone scripts, not a package: there is no `setup.py`/`pyproject.toml`, no test suite, no linter config, and nothing is importable as an installed module. Everything is run as `python <script>.py`, and configuration is done by **editing constants at the top of the script** rather than by flags (except for the few argparse-based scripts listed below).

The work has two explanation *generators* and three families of *evaluators*:

| Stage | Where | Produces |
|---|---|---|
| Dataset conversion (QA → binary claims) | `generators/convertors/` | `generators/*_claims_dataset.json` |
| CoT explanation generation (via local Ollama HTTP) | `generators/cot_generator_*.py` | `results/generation/cot_generation/{dataset}_{model}/*.jsonl` |
| ArgLLM generation (via HuggingFace, builds argumentation graphs) | `generators/argLLM_implementation/argumentative-llms_/argllm_generator_*.py` | `results/generation/argLLM_generation/{dataset}_{model}/argllm_outputs_ollama.jsonl` |
| CoT structural metrics (NLI graph → redundancy / relevance) | `evaluation/cot_explanation_evaluation_calc_{qa,cv}.py`, driven by `experiments/cot_metrics_summary_{qa,cv}.py` | `results/cot_method/` |
| ArgLLM metrics (structural + semantic acceptability/circularity) | `evaluation/run_{structural,semantic}_arg_eval.py` | `results/argLLM_metrics_{structural,semantic}/` |
| SHAKE faithfulness test (attention perturbation → label flip) | `evaluation/run_shake_test.py`, `evaluation/expl_token_perturbation/run_expl_shake_test.py` | `evaluation/shake_test_results/`, CSV+HTML |
| AKF (argument-knowledge-faithfulness, paraphrase-based) | `evaluation/akf/run_akf.py` | `results/akf_test/` |

`evaluation/Evaluating_Explanations/` and `generators/argLLM_implementation/argumentative-llms_/Uncertainpy/` are **vendored third-party code** (the AAAI-25 ArgLLM paper's repo and the Uncertainpy gradual-argumentation library). Treat them as upstream: prefer not to modify them.

## The naming convention is load-bearing

Almost every summarizer discovers its inputs by globbing on `{dataset}_{model}`:

- dataset ∈ `truthfulqa | medqa | strategyqa | commonsenseqa` (QA) and `truthfulclaim | medclaim | strategyclaim | commonsenseclaim` (claim verification)
- model ∈ `llama | mistral | qwen`

Directory names, CSV filenames (`shake_test_results_{dataset}_{model}.csv`, `results_{dataset}_{model}.csv`, `{dataset}_{model}.csv`) and checkpoint names (`checkpoint_{dataset}_{model}.json`) all follow it. Renaming an output file silently breaks the aggregation scripts (`summarize_akf.py`, `cot_prediction_label_accuracy.py`, `argllm_prediction_table_accuracy.py`, `semantic_arg_eval_analysis.py`).

The "claim" datasets are the QA datasets rewritten into single self-contained assertions with a boolean `label`, so the whole pipeline can reduce to a TRUE/FALSE decision. Claim JSON schemas are *not* uniform — `strategy_claims_dataset.json` uses `qid` and `label/answer` where the others use `id` and `label`.

## Two disjoint runtime environments

1. **Generation via Ollama** (`cot_generator_*.py`, the convertors): POSTs to `http://localhost:11434/api/generate`. Requires a running Ollama server with the model pulled (`llama3:latest`, `mistral`, `qwen3:4b`). No GPU needed.
2. **Everything else**: `torch` + `transformers` loading 7B checkpoints in fp16 with `device_map="auto"` and **`attn_implementation="eager"`** (required — the SHAKE perturbation subclasses the attention module and needs real attention probabilities, which SDPA/flash do not return). Needs a GPU and `HF_TOKEN` in the environment for the gated Llama-2 weights. Pinned deps for the ArgLLM half are in `generators/argLLM_implementation/argumentative-llms_/requirements.txt`; the evaluators additionally need `sentence-transformers`, `pandas`, `seaborn`, `scikit-learn`.

## Running things

Scripts that use `from <sibling> import ...` **must be run from their own directory**:

```bash
# SHAKE, rationale-token variant (edit MODEL_NAME in shake_pipeline.py first)
cd evaluation && python run_shake_test.py

# SHAKE, explanation-ablation variant (the only fully flag-driven runner)
cd evaluation/expl_token_perturbation && python run_expl_shake_test.py \
  --data ../../generators/truthful_claims_dataset.json \
  --csv out.csv --html out.html --last-n-layers 12 --max-kw 32 --sample-mode random --sample-size 500

# ArgLLM generation (edit MODEL_NAME / dataset / SAVE_PATH at the top)
cd generators/argLLM_implementation/argumentative-llms_ && python argllm_generator_claim_verification.py

# AKF (edit MODEL_NAME, INPUT_JSON_PATH, CSV_PATH at the top)
cd evaluation/akf && python run_akf.py
```

Argparse-driven scripts, runnable from the repo root:

```bash
# CoT metrics over all {dataset}_{model} folders, with per-pair checkpointing
python experiments/cot_metrics_summary_qa.py --base-dir results/generation/cot_generation \
  --datasets truthfulqa strategyqa medqa --models mistral llama qwen --batch-size 16
python experiments/cot_metrics_summary_cv.py --base-dir results/generation/cot_generation ...

# ArgLLM explanation metrics
python evaluation/run_structural_arg_eval.py --input <argllm_outputs.jsonl> --out_csv <out.csv> --bags base estimated
python evaluation/run_semantic_arg_eval.py  --input <argllm_outputs.jsonl> --out_csv <out.csv> --tau-circ 0.80 --tau-neutral 0.70

# Aggregation
python evaluation/summarize_akf.py --inputs results/akf_test --outdir results/akf_test/summary_analysis
python evaluation/semantic_arg_eval_analysis.py
```

The `cot_metrics_summary_*.py` runners write `checkpoint_{dataset}_{model}.json` per pair, so an interrupted run resumes rather than recomputing — don't delete those while iterating.

## Known breakages and traps

These are pre-existing; fix them deliberately, don't be surprised by them.

- **Hard-coded absolute paths.** Many scripts embed `/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/...` for inputs and outputs (`run_shake_test.py`, `akf/run_akf.py`, `derive_*.py`, `*_prediction_table_accuracy.py`, `faithfulness_label_flip_analysis*.py`, `semantic_arg_eval_analysis.py`). Several also set `os.environ["HF_HOME"] = "/vol/bitbucket/rc1124/hf_cache"` **before** importing transformers. Any run outside that cluster requires editing these.
- **Broken imports in `experiments/`.** `cot_metrics_summary_{qa,cv}.py` import `evaluation.explanation_evaluation_calc_{qa,cv}` and `aggregate_cot_metrics_table.py` imports `evaluation.explanation_evaluation_calc`; the modules were renamed to `evaluation/cot_explanation_evaluation_calc_{qa,cv}.py` and never updated. `aggregate_cot_metrics_table.py` additionally does `os.path.abspath("...")`, which is not a valid parent-directory path — it is superseded by the `cot_metrics_summary_*` scripts.
- **Generator output paths are stale.** `cot_generator_claim_verification.py` writes to `results/generation/{dataset}_{model}/`, but the committed tree has since been reorganized into `results/generation/cot_generation/` and `results/generation/argLLM_generation/`. Pass `--base-dir` to the summarizers accordingly.
- **Per-model duplication instead of parameterization.** `attention_perturbation_{llama,mistral,qwen}.py` are near-identical copies differing only in which attention class they subclass (`LlamaAttention` / `MistralAttention` / `Qwen2Attention`); `cot_explanation_evaluation_calc_qa.py` and `..._cv.py` differ only in a few prompt-boilerplate phrases. `run_shake_test.py` imports the llama variant by name, and switching models means editing both the import and the commented-out `MODEL_NAME` block in `shake_pipeline.py`. A change to one variant almost always needs mirroring into the others.
- **Dataset selection by comment.** `cot_generator_qa.py` and `dataset_probe.py` select a dataset by uncommenting one of four `load_dataset(...)` blocks.

## Domain concepts worth knowing before editing

- **SHAKE / BFS.** The faithfulness test asks the model for a TRUE/FALSE label, extracts the rationale tokens it claims to rely on, then zeroes those tokens' attention *columns* in the last N layers and renormalizes the rows (`CustomLlamaAttention.forward`, `AttentionPatcher.install/restore`). BFS ("Binary Flip Score") is 1.0 if the label flipped, 0.0 otherwise; `UNKNOWN` labels are filtered out before computing flip rates. Two variants exist: perturbing tokens of the *claim* (`evaluation/`) versus tokens of the model's own *explanation* (`evaluation/expl_token_perturbation/`, which also reads soft `p_true` off the last-token logits rather than only decoding text).
- **ArgLLM outputs** contain two "bags" per record — `base` and `estimated` — each a bipolar argumentation graph of named arguments with `initial_weight` and gradual-semantics `strength`. The root/decision argument is `db0`; supports/attacks are named `Sdb0←d1b1` / `Adb0←...`. Predictions come from thresholding the root strength at 0.5 (`derive_argllm_predictions.py`).
- **CoT structural metrics** work by splitting the explanation into propositions, sampling down to ≤15–20 of them (`IntelligentCoTSampler`, scoring by position, length, and discourse markers), running pairwise NLI with `textattack/bert-base-uncased-MNLI` to build an entailment adjacency matrix, and feeding that matrix to the vendored `compute_redundancy` / `compute_strong_relevance` / `compute_weak_relevance`. Redundancy is best at 0; the relevance metrics are best at 1.
- **Semantic vs structural ArgLLM eval.** Structural runs circularity/acceptability on the graph topology alone; semantic (`run_semantic_arg_eval.py`) first embeds argument texts with `all-mpnet-base-v2` and uses similarity thresholds (`--tau-circ`, `--tau-neutral`) so paraphrased-but-distinct arguments count as circular.

## Conventions in the first-party code

Docstrings follow a consistent `@params:` / `@returns:` style (not Google or NumPy format). Scripts log progress with prefixed `print(..., flush=True)` tags like `[AKF-Lite]`, `[PATCH]`, `[WARN]`. Result tables are emitted as a CSV plus a self-contained interactive HTML table built by a per-script `build_html()` that pulls jQuery/DataTables from a CDN — that helper is copy-pasted across `run_shake_test.py`, `run_expl_shake_test.py`, `run_akf.py`, and `summarize_akf.py`.
