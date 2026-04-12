# Valid survey simulations with limited human data

[Stefan Krsteski](https://github.com/Stefanstud)<sup>1</sup>, Giuseppe Russo<sup>1,2</sup>, Serina Chang<sup>3</sup>, Robert West<sup>1</sup>, Kristina Gligorić<sup>4</sup>

<sup>1</sup>EPFL &nbsp; <sup>2</sup>Stanford University &nbsp; <sup>3</sup>UC Berkeley &nbsp; <sup>4</sup>Johns Hopkins University

[![arXiv](https://img.shields.io/badge/arXiv-2510.11408-red.svg)](https://arxiv.org/abs/2510.11408)
[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue.svg)](https://2026.aclweb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TL;DR: We study how to combine LLM-generated survey responses with a small set of real human responses to produce valid population estimates. We answer the question of whether and how LLMs can be useful for survey research.**

## Getting started

```bash
git clone <repo>
cd <repo>
uv sync # or use conda if preferred
```

### Reproducibility

```bash
uv run python reproduce.py
```

This evaluates all three datasets (NHANES, ATP Q1, ATP Q2) across four synthesis methods and two rectification strategies, then prints the main table from the paper. Numbers may vary slightly from the paper due to bootstrap averaging over random labeled splits. A subset of simulations were regenerated after a cluster storage loss; since all models use non-zero sampling temperature (τ=0.7), regenerated responses may differ slightly from the originals. The overall patterns and conclusions remain consistent.

### Evaluate a single dataset

```bash
cd datasets/atp1
python evaluate.py --trial_ns 100 50 150 200
```

### Simulate responses

```bash
cd datasets/atp1
python simulate.py --config configs/simulate_gpt.yaml
```

Requires a `.env` file with `OPENAI_API_KEY=...` for GPT models.

### Fine-tune a model

```bash
pip install axolotl==0.12.2
axolotl train datasets/atp1/configs/finetune_qwen.yaml
```

For SubPOP fine-tuning, see the [SubPOP repository](https://github.com/JosephJeesungSuh/subpop).


## Citation

```bibtex
@article{krsteski2025valid,
  title={Valid survey simulations with limited human data: The roles of prompting, fine-tuning, and rectification},
  author={Krsteski, Stefan and Russo, Giuseppe and Chang, Serina and West, Robert and Gligori{\'c}, Kristina},
  journal={arXiv preprint arXiv:2510.11408},
  year={2025}
}
```
