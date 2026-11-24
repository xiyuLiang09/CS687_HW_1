# Sycophancy Evaluation on LLM: Harmful Flips of LLMs under User Pushback

Implementation of  CS687 Homework 1. 

The goal of these experiments is to measure LLM sycophancy, in which a model abandons a correct, verifiable answer after unsupported user pushback, on Olympiad-level mathematics problems with ground-truth answers. This sycophancy failure mode is commonly called a **harmful flip**.

**An intuitive example** (full conversation with DeepSeek, for Q(4)-a):
<https://chat.deepseek.com/share/9q0dsk6bsdjaml8ik0>

## Experiment Results

**Settings**: 36 Olympiad problems (Omni-MATH, difficulty 6–7) x 2 conditions x 5 fresh sessions; main condition asserts a wrong answer at T3, control condition C1 asks for a neutral re-check instead. HFR = P(final answer wrong | initial answer correct), with 95% Wilson CIs.

| Model (thinking disabled) | Initial acc (main) | HFR (main) | HFR (control) | Adopted distractor |
|---|---|---|---|---|
| deepseek-v4.1-flash | 78.9% | **44.4%** [36.4%, 52.6%] | 3.1% [1.2%, 7.6%] | 21.8% |
| deepseek-v4-pro | 48.3%* | **26.4%** [18.3%, 36.6%] | 9.2% [4.7%, 17.1%] | 21.8% |

\* The 36 problems were selected based on the smaller model’s unchallenged reliability (pass@5 >= 1). Therefore, the base model (the smaller one) can answer each problem correctly at least once, while the scaled model (the larger one) cannot always solve them correctly. 

HFR conditions on initially-correct conversations per model. On the **27 problems** where both models had initially-correct conversations, HFR was **43.2% (flash)** vs **26.4% (pro)**.

**Takeaway**: scaling roughly halves the flip rate but does not eliminate the failure; when the larger model flips, it fails in exactly the same way (identical distractor-adoption rate of 21.8%).

## Repo Structure

```
syco_pipeline.py            # 5-stage pipeline: prepare -> filter -> distract -> challenge -> report
make_omni_jsonl.py          # Omni-MATH subset sampler (difficulty quotas, answer/problem form filters)
probe_no_think.py           # probe which API convention disables "thinking" on an endpoint
omni_math_d6x20_d7x30.jsonl # the 50 sampled problems (difficulty 6 x 20, difficulty 7 x 30, seed 0)
example.md  # the conversation of Deepseek(final turn verbatim + link)
syco_runs_omni/             # full artifacts for deepseek-v4.1-flash (smaller model, thinking disabled)
syco_runs_omni_pro/         # full artifacts for deepseek-v4-pro (larger model, thinking disabled)
```

Each `syco_runs_*/` folder contains: `problems.jsonl` (input), `filter_runs.jsonl` (pass@k unchallenged solves), `challenge_set.jsonl`, `challenge_set_distractors.jsonl`, `challenge_runs.jsonl` (graded 3-turn conversations), `report.json` (aggregate stats), `summary.csv` (flat export).

## Reproduce

```bash
pip install openai tqdm math_verify datasets pandas pyarrow

# Need API to run the model
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_API_KEY="sk-..."

# Sample the 50-problem set (or use the included omni_math_d6x20_d7x30.jsonl)
python3 make_omni_jsonl.py --from-file Omni-Math.jsonl \
    --quotas "6:20,7:30" --seed 0 --out omni_math_d6x20_d7x30.jsonl

# Take `flash` for script example
python3 syco_pipeline.py prepare   --source jsonl \
    --source-path omni_math_d6x20_d7x30.jsonl --out syco_runs_omni --max-problems 50
python3 syco_pipeline.py filter    --model deepseek-flash --out syco_runs_omni \
    --min-correct 1 --no-thinking --no-think-via thinking_flag
python3 syco_pipeline.py distract  --model deepseek-flash --out syco_runs_omni \
    --no-thinking --no-think-via thinking_flag
python3 syco_pipeline.py challenge --model deepseek-flash --out syco_runs_omni \
    --with-c1 --runs 5 --no-thinking --no-think-via thinking_flag
python3 syco_pipeline.py report    --out syco_runs_omni
```

For the larger model, repeat with `--model deepseek-v4-pro` and a separate `--out` directory, reusing the same `problems.jsonl` and `challenge_set_distractors.jsonl` so problems, distractors and prompts are identical across models.

## Notes

- Prompts and generation settings are frozen across models for comparability (T=0.7, max_tokens=16384, thinking disabled via the official
  `thinking: {"type": "disabled"}` API flag).
- Answers are graded by `math_verify` against the ground-truth final answer; two problems whose harvested distractors were grading artifacts (semantically equal to the gold answer) were removed (38 -> 36).
- Problems are sampled from [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) (Gao et al., 2024, arXiv:2410.07985); the sampled subset is included for reproducibility.
