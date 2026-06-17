# OR-Bench for Inspect AI

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of **OR-Bench: An
Over-Refusal Benchmark for Large Language Models** (Cui et al., ICML 2025).

- Paper: <https://arxiv.org/abs/2405.20947>
- Dataset: <https://huggingface.co/datasets/bench-llm/or-bench>
- Reference implementation: <https://github.com/justincui03/or-bench>

## What it measures

*Over-refusal* — when a safety-aligned model rejects a benign request. A model can
look "safe" while being useless by refusing too much; OR-Bench quantifies that.

Scoring is faithful to the official `response_checker`: an LLM judge classifies each
response as `direct_answer`, `direct_refusal`, or `indirect_refusal` (a refusal is
either refusal class). The reported metric is the **rejection rate** (mean of
`refused`).

| Task | Prompts | Direction |
| --- | --- | --- |
| `or_bench_hard_1k` | hard seemingly-toxic-but-safe | **lower** rejection rate is better |
| `or_bench_80k` | full safe set | **lower** rejection rate is better |
| `or_bench_toxic` | genuinely toxic control | **higher** rejection rate is better |

A well-aligned model rejects most toxic prompts and few safe ones.

## Install

```bash
pip install -e .          # or: uv pip install -e .
```

## Usage

```bash
# by file path
inspect eval src/or_bench/or_bench.py@or_bench_hard_1k --model openai/gpt-4o --limit 20

# or via the registered package
inspect eval or_bench/or_bench_toxic --model openai/gpt-4o --limit 20
```

The judge defaults to the eval model. Override it with the `grader` model role:

```bash
inspect eval or_bench/or_bench_hard_1k \
  --model openai/gpt-4o \
  --model-roles grader=openai/gpt-4o-mini
```

Any OpenAI-compatible endpoint works (set `OPENAI_API_BASE` / `OPENAI_API_KEY`).

## Tests

```bash
pytest          # unit tests for the judge-label parser (no model required)
```

## Notes

- The judge prompt and 3-class scheme mirror the official OR-Bench `response_checker`;
  the paper additionally uses a keyword pre-filter for the full 80k set and a
  three-judge ensemble for dataset construction.
- Rejection rate is reported per task; cross-task comparison (toxic vs. safe) follows
  the paper's two-axis presentation.

## Citation

```bibtex
@inproceedings{cui2025orbench,
  title={OR-Bench: An Over-Refusal Benchmark for Large Language Models},
  author={Cui, Justin and Chiang, Wei-Lin and Stoica, Ion and Hsieh, Cho-Jui},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
