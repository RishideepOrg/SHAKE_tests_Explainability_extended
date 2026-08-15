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

The authoritative reference is `literature_papers/MSc_Thesis_Rishideep.pdf` (Imperial College MSc Advanced Computing, "Evaluating the Explanatory Properties of Large Language Models", 2025). The metric definitions come from Kotonya & Toni's evaluation framework [KT24]; the ArgLLM method from Freedman et al., AAAI-25 [FDG+25].

**`literature_papers/` holds all reference papers for this work** — the thesis plus background reading. Add new papers there. Current contents beyond the thesis: Kadem & Zheng, *Interpreting Transformers Through Attention Head Intervention* (arXiv 2601.04398v3, Jan 2026) — a survey of attention-head ablation as causal interpretability; relevant because SHAKE is an instance of the intervention paradigm it traces, and because it supplies the formal faithfulness criteria and ablation controls SHAKE currently lacks.

- **SHAKE** = *Self-attention Heuristic for Attributional Keyword Elimination*. The model is asked for a TRUE/FALSE label and, separately, for ≤K=4 rationale keywords drawn from the claim itself; those words are aligned to subword indices via the tokenizer's offset mapping, their attention *columns* are zeroed in the last L layers and the rows renormalized (`CustomLlamaAttention.forward`, `AttentionPatcher.install/restore`), and the claim is re-run. BFS ("Binary Flip Score") = 1 if the label flipped. **The thesis reads a flip as evidence of faithfulness** — the declared rationale demonstrably drove the decision. `UNKNOWN` labels at either end are dropped before computing flip rates. The design point versus prior work is that the intervention happens in latent space, so the input text is never mutated and no out-of-distribution masking artifacts are introduced (contrast [FYSP24], [LLG+22], [MKS21]).
- **Why K=4 and L=1.** K=4 came from observing that claims in TruthfulClaim/StrategyClaim carry 2–3 key concepts, and larger K over-perturbs. L was swept over {1,2,3,4} with near-identical results in ~30-layer models, so L=1 was kept. Both are explicitly flagged in the thesis as under-explored (K unswept; L untested beyond 4, ideally to ~10).
- **SHAKE_expl** (`evaluation/expl_token_perturbation/`) is the stricter variant: generate a CoT explanation *without* a label, then label from (claim + explanation), then extract rationale keywords **from the explanation** and perturb those. It also reads soft `p_true` off the last-token logits. On Mistral/TruthfulClaim it produced **BFS ≈ 0** for every setting tried (L up to 10, max_kw up to 32) — the thesis attributes this to post-hoc rationalization (the label is implicitly fixed before the explanation is written) and to the model returning only 2–3 generic keywords, leaving too little attention mass to ablate. This is a known negative result, not a bug to be fixed blindly.
- **ArgLLM outputs** contain two bags per record — `base` (neutral 0.5 priors, topology only) and `estimated` (LLM-elicited per-argument certainty as the prior) — each a bipolar argumentation graph with `initial_weight` and propagated `strength` under gradual semantics (product aggregation, linear influence). The root is `db0`; supporters/attackers are `Sdb0←...` / `Adb0←...`. Prediction = root strength thresholded at 0.5. All experiments were run at **depth D=1, breadth B=1** only.
- **Structural argumentative metrics are degenerate by design at D=1** — this is the thesis's own finding, not a defect. Acceptability is always 0 (an attacker of the claim has no attacker of its own at D=1) and structural circularity is always 0 for any D and B (distinct argument IDs can never form a cycle). That is *why* `run_semantic_arg_eval.py` exists: it re-reads both properties through sentence-embedding cosine similarity — a supporter too close to the claim (≥ `--tau-circ`) counts as circular (it paraphrases rather than adds a reason), and an attacker too close to the claim or any supporter (≥ `--tau-neutral`) counts as neutralized, raising Semantic Acceptability. Note the direction: **high SAcc means attackers were vacuous**, which is a statement about weak opposition, not a purely good thing.
- **CoT deductive metrics** split the explanation into propositions, prune to ≤15 with `IntelligentCoTSampler` (position, length, discourse markers, entity connectivity; meta-commentary penalized) to avoid O(n²) NLI cost, run pairwise NLI with `textattack/bert-base-uncased-MNLI` at an entailment threshold of 0.85 to build the adjacency matrix, then apply the vendored `compute_redundancy` / `compute_strong_relevance` / `compute_weak_relevance`. Redundancy best at 0; both relevance metrics best at 1. Redundancy is set to 1 whenever the graph splits into more than one connected component — which is why observed redundancy is ≥0.92 nearly everywhere.
- **AKF** (`evaluation/akf/`, thesis Appendix A) = *Arg-Knowledge Faithfulness*, a node-level stress test on argLLM arguments. Each argument is treated as a standalone proposition; the model's belief log-odds ℓ(p) = log p(TRUE) − log p(FALSE) are scored on k paraphrases (Paraphrase Consistency, binary and soft; Paraphrase Stability = 1/(1+std ℓ)) and on a negation (Negation Sanity — a faithful belief should *flip*). `SIMPLE_AKF = ½PC_soft + ½NS⁺_soft`; `STABLE_AKF = 0.4PC_soft + 0.4NS_soft + 0.2PS`. Preliminary finding: NS ≪ PC everywhere, i.e. arguments survive rephrasing but fail to invert under negation. Paraphrases/negations are currently template-based (auxiliary swaps, "It is not the case that…"), which the thesis names as the main weakness.

## Headline results these pipelines produced

Useful as regression baselines when changing anything.

- **CoT**: redundancy ≥0.92 everywhere (Qwen hits 1.0000); Weak Rel. 0.13–0.35, Strong Rel. 0.11–0.29. Mistral best on every metric, Qwen worst (verbose, filler-heavy). MedClaim/MedQA hardest across models.
- **ArgLLM semantic (claim verification)**: Llama highest SAcc (macro 0.791) but worst SCirc (0.603); Qwen best SCirc (0.244) but lowest SAcc; Mistral balanced.
- **SHAKE flip rates**: Llama **86.37%** weighted (99.02% on MedClaim), Mistral 13.84%, Qwen 10.48%. Dropped-as-UNKNOWN rates: Llama 9.46%, Qwen 8.53%, Mistral 4.05%.
- **Prediction accuracy is largely decoupled from explanation quality** — Qwen posts the best accuracies while producing the most redundant explanations. This decoupling is the thesis's stated reason for needing a faithfulness test at all.

## Current scope

**SHAKE — in either form, binary label flips (BFS) or a confidence/probability-based score — is the single focus of the current phase of work.** Default new experiments, refactors, and analysis to the SHAKE pipelines: `evaluation/run_shake_test.py`, `evaluation/shake_pipeline.py`, the per-model `attention_perturbation_*.py` backends, and `evaluation/expl_token_perturbation/`.

Live threads, drawn from Chapter 6 ("The Road Ahead"):

1. **Establish a random-token control** — see the first caveat below. Cheap, reuses the existing perturbation machinery, and every downstream SHAKE claim depends on it.
2. **Replace the binary flip with a graded signal** — the change in prediction probability/confidence rather than BFS. The soft `p_true` plumbing already exists in `expl_attention_perturbation_llama.py::logits_label_on_last_token`. Note that a graded score was already prototyped once: commit `8be3a47` added `compute_mbf_norm` to `run_shake_test.py` (sweep k = 1..K over rationale-token prefixes, find the minimal k\* that flips, score 1 − (k\*−1)/K, 0 if no flip), emitting `mbf_norm`/`mbf_k_star`/`mbf_K` columns. It was removed in `c752254` before submission — recover it from git rather than rewriting from scratch.
3. **Rescue SHAKE_expl** — force the model to emit rationales *before* the label (blocking post-hoc rationalization), and/or ground explanations in retrieved evidence.
4. **Widen the sweeps** — K (rationale count) and L (up to ~10 layers).

**Out of scope:**

- **The user study is discontinued.** It plays no part in this phase and must not appear in any future results or conclusions. Do not propose re-running or scaling it, and do not cite its findings as evidence. `results/user_study/` is historical only.
- **AKF is a distant second on the long-term agenda** — not something to advance now. The eventual question, when it is picked up again, is whether nodes failing PC/NS also flip less under SHAKE; that cross-check is what would substantiate the thesis's RQ2 claim of combining attributional and semantic faithfulness. `evaluation/akf/` and its results stay in place meanwhile.
- Broader argLLM sweeps (D and B beyond 1) and >100B models for the CoT model-scale question are thesis future-work items, but not part of this phase.

### Two caveats the thesis does not fully resolve

Worth holding in mind before building on the SHAKE headline number.

- **No random-token control was reported.** Chapter 5's own preamble promises to compare BFS "to the Binary Flip Scores (BFS) after perturbing the model's attention to random tokens", but no such baseline appears in Table 5.7 or anywhere else, and no code in this repo implements it. Without it, Llama's 86% flip rate cannot be separated from generic perturbation brittleness.
- **The SHAKE models differ from the generation models.** CoT/argLLM explanations were generated with Llama-3-8B, Mistral-7B-Instruct-v0.1 and Qwen3-8B (via Ollama), while SHAKE ran on **Llama-2-7b-chat-hf**, Mistral-7B-Instruct-v0.1 and Qwen2.5-7B-Instruct (via HuggingFace, since attention modules must be patchable). So "Llama is faithful" and "Llama's CoT is redundant" are statements about *different* Llamas.

## Conventions in the first-party code

Docstrings follow a consistent `@params:` / `@returns:` style (not Google or NumPy format). Scripts log progress with prefixed `print(..., flush=True)` tags like `[AKF-Lite]`, `[PATCH]`, `[WARN]`. Result tables are emitted as a CSV plus a self-contained interactive HTML table built by a per-script `build_html()` that pulls jQuery/DataTables from a CDN — that helper is copy-pasted across `run_shake_test.py`, `run_expl_shake_test.py`, `run_akf.py`, and `summarize_akf.py`.
