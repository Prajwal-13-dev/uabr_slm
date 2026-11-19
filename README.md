# Instruction Tuning Disrupts a Small Model's Retrieval Policy: Two Testbeds and Teacher-Based Decision Anchoring


> **Team Straw Hats (PES University, AFML)**  
> **Authors:** Pradeep T, Prajwal A, Nikitha Uppar, Sakshi Kiran  
> **Project:** UABR-SLM (Utility-Aware Boundary Replay & Decision Anchoring on ~40M Small Language Models)


## Summary

Retrieval-Augmented Generation (RAG) models must decide for each question whether to answer from their own weights (`<ANSWER>`) or call an external retriever (`<CALL>`).

In this project, we train a **39.7M-parameter decoder-only LLaMA model from scratch** on **1.0B Wikipedia tokens** and investigate what happens to this routing decision when the model undergoes standard Supervised Fine-Tuning (SFT) on 6.0M tokens of Dolly-15k and Alpaca.

### Key Discoveries:
1. **SFT Disrupts Routing:** Instruction tuning severely damages the retrieval policy. In Track S, AUROC drops from **0.684 → ~0.50** (chance). In Track P, AUROC drops from **0.828 → ~0.70**, and closed-book exact match drops from 11.3% to ~1.7%.
2. **UABR Does Not Beat Random Replay:** Prioritizing boundary points ($q_i = \exp(-m_i/\tau) \cdot |\Delta U_i|$) underperforms uniform random replay. Boundary examples suffer from label noise and narrow coverage. In Track S, utility-derived targets skewed the buffer to 65% `<CALL>`, causing the model to over-retrieve.
3. **The Breakthrough — Decision-Slot Anchoring:** Probing reveals that internal representations still retain the routing information, but the output unembedding head drifts during SFT (**readout misalignment**). A teacher-based decision-slot anchoring loss ($A_1$ / $A_2$) restores test AUROC to **0.829–0.830** (pre-SFT: 0.828) while retaining **99.4%** of instruction gains, without requiring ground-truth routing labels.

## Dual-Track Experimental Design

Our study evaluates the phenomenon across two complementary testbeds:

| Dimension | Track S (Synthetic Benchmark) | Track P (PopQA Benchmark) |
| :--- | :--- | :--- |
| **Directory** | [`track_s/`](track_s/) | Root (`src/uabr/`, `notebooks/`) |
| **Dataset** | 2,400 invented entities $\times$ 3 relations (7,200 facts) | 14,267 PopQA questions (1,689 ambiguous excluded) |
| **Exposure Control**| Exact exposures: $0\times, 40\times, 120\times, 300\times$ | Forced subject/object pretraining + reading adaptation |
| **Evidence** | Top-3 keyword search with 20% look-alike distractors | BM25 over ~19.8M 60-word Wikipedia windows |
| **Routing Target** | Knowledge-based ($b_{\text{net}} > 0$) vs Utility-derived | Counterfactual benefit ($b_{\text{net}} = \bar{F}_1(C) - \bar{F}_1(A) - \lambda$) |
| **Primary Metric** | Exact-match accuracy at 10% call budget | Sealed-test routing AUROC (paired clustered bootstrap) |
| **Paper Reference**| **Table 2** | **Table 3, Table 4, and Figure 1** |

## Key Results

### 1. Track S Results (Table 2 in Paper)
Evaluated on 2,880 test prompts across Seed 11 and Seed 29:

| Method | Acc @ 10% Budget | AUROC | Call Rate (0× Facts) | Call Rate (300× Facts) |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-SFT Model** | **0.704** | **0.684** | 55.7% | 19.5% |
| **Standard SFT** | 0.324 / 0.331 | 0.511 / 0.493 | 29.4% / 23.7% | 20.1% / 14.9% |
| **Random Replay (5%)** | **0.479 / 0.478** | 0.594 / 0.587 | 65.3% / 59.2% | 44.8% / 43.0% |
| **BPR (Boundary-only)** | 0.453 / 0.453 | 0.596 / 0.603 | 64.1% / 61.8% | 41.6% / 42.3% |
| **UABR** | 0.419 / 0.427 | 0.600 / 0.598 | 95.4% / 94.7% | 88.8% / 87.9% |

*UABR suffered severe target imbalance (65% `<CALL>`), causing the model to retrieve indiscriminately.*

### 2. Track P Sealed Test Results (Table 3 in Paper)
Pre-SFT Router Baseline: **AUROC 0.828 [0.804, 0.849]**, Call Rate 13.9%.

| Model / Strategy | Test AUROC (s11 / s29) | Call Rate (%) | Verdict vs Plain SFT |
| :--- | :---: | :---: | :---: |
| **A2 Question Anchoring** | **0.829 / 0.830** | 11.3 / 14.1 | **+0.137 / +0.130 (Best)** |
| **A1 SFT-Prompt Anchoring** | **0.817 / 0.815** | 9.7 / 14.6 | **+0.124 / +0.115 (Zero extra data)** |
| **OPR-inspired Replay** | 0.808 / 0.823 | 12.1 / 15.1 | +0.115 / +0.123 |
| **WiSE-FT Interpolation ($\alpha=0.5$)** | 0.810 / 0.811 | 15.2 / 13.0 | +0.117 / +0.111 |
| **Random Replay (5%)** | 0.800 / 0.795 | 15.9 / 15.6 | +0.107 / +0.095 |
| **Benefit-only Replay** | 0.784 / 0.800 | 12.4 / 12.9 | +0.091 / +0.100 |
| **CASR-inspired Replay** | 0.787 / 0.784 | 54.5 / 55.9 | +0.094 / +0.084 |
| **UABR (Boundary $\times$ Benefit)** | 0.709 / 0.740 | 6.5 / 6.6 | +0.016 / +0.040 |
| **BPR (Boundary-only)** | 0.665 / 0.710 | 3.2 / 3.7 | -0.028 / +0.010 (Weakest) |
| **Plain SFT** | 0.693 / 0.700 | 28.6 / 21.2 | Baseline failure |


## Repository Structure

```text
uabr_slm/
├── report/                     # Paper materials
│   ├── Report_FINAL.pdf        # Complete IEEE research report
│   ├── make_figures.py         # Generates Figure 1 & result charts
│   └── report.md               # Extended technical notes
│
├── track_s/                    # Track S: Synthetic Facts Exposure Testbed
│   ├── configs/                # Hyperparameters for synthetic runs
│   ├── notebooks/              # Phases 1 to 7 notebooks
│   ├── src/                    # Synthetic data generator & model code
│   └── scripts/                # Execution pipeline scripts
│
├── notebooks/                  # Track P: PopQA Benchmark Pipeline
│   ├── 00_popqa_splits.ipynb   # 70/15/15 Union-Find subject-clustered splits
│   ├── 01_corpus_tokenizer.ipynb# 12k BPE tokenizer & Wiki chunking
│   ├── 02_bm25_retrieval.ipynb # BM25 retrieval over 60-word passages
│   ├── 03_pretrain.ipynb       # 39.7M LLaMA pretraining on 1B tokens
│   ├── 03b_reinject.ipynb      # Dose ladder fact injection (0/16/64/256)
│   ├── 04_g1_pilot.ipynb       # Closed-book knowledge probe & Gate 01
│   ├── 05_mixed_training.ipynb # Q&A + RAFT reading adaptation
│   ├── 06_routing.ipynb        # Counterfactual benefit labeling (lambda=0.1875)
│   ├── 07_sft_pilot.ipynb      # SFT interference verification
│   ├── 08_replay.ipynb         # Replay arms (Random, BPR, UABR, OPR, CASR)
│   ├── 09_test.ipynb           # Sealed test evaluation & paired bootstrap
│   ├── 10_selfrag.ipynb        # Critique token baseline
│   ├── 11_posthoc.ipynb        # Mechanistic linear probing (Figure 1)
│   └── 12_methods.ipynb        # Decision-slot anchoring (A1, A2) & readout repair
│
├── src/uabr/                   # Track P source library
│   ├── data/                   # PopQA loaders, formatting, tokenizer
│   ├── model/                  # 39.7M decoder architecture (RoPE, SwiGLU, RMSNorm)
│   ├── retrieval/              # BM25 indexing and document scoring
│   ├── routing/                # Counterfactual benefit & metric calculations
│   ├── replay/                 # Selection strategies (UABR, Random, BPR, OPR, CASR)
│   ├── train/                  # SFT trainer, Decision-Slot Anchoring (A1/A2)
│   └── eval/                   # Paired clustered bootstrap statistics
│
├── configs/                    # YAML configuration files
└── docs/                       # Specifications and Gate verification logs
```

## How to Run & Reproduce

### 1. Environment Setup
```bash
git clone https://github.com/Prajwal-13-dev/uabr_slm.git
cd uabr_slm
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Running Track S (Synthetic Benchmark)
```bash
cd track_s
python scripts/run_pipeline.py --phases 1-7
```

### 3. Running Track P (PopQA Benchmark)
The pipeline is designed to execute cleanly on Kaggle (T4 x2 GPUs) or local workstations:
* **Phase 00–03:** Pretraining and BM25 index creation (`notebooks/00_popqa_splits.ipynb` through `03_pretrain.ipynb`).
* **Phase 05–06:** Mixed reading adaptation and counterfactual label generation (`notebooks/05_mixed_training.ipynb`, `notebooks/06_routing.ipynb`).
* **Phase 08:** SFT with Replay comparison (`notebooks/08_replay.ipynb`).
* **Phase 11–12:** Probing analysis and Decision-Slot Anchoring ($A_1$ / $A_2$) (`notebooks/11_posthoc.ipynb`, `notebooks/12_methods.ipynb`).
* **Phase 09:** Final evaluation on the sealed test split (`notebooks/09_test.ipynb`).


## Official Dataset Links

All datasets used in this work are publicly available:
1. **PopQA**: Official dataset from Mallen et al. [ACL 2023]  
   * Source: [https://github.com/AlexTMallen/PopQA](https://github.com/AlexTMallen/PopQA)
2. **Wikipedia Pretraining Corpus**: English Wikipedia dump processed into 60-word passage windows.  
   * Source: [Wikimedia Dumps](https://dumps.wikimedia.org/) / [Hugging Face `wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
3. **Instruction Tuning Datasets (SFT)**:
   * **Databricks Dolly-15k**: [https://huggingface.co/datasets/databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
   * **Cleaned Stanford Alpaca**: [https://huggingface.co/datasets/yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
4. **Track S Synthetic Data**: Deterministically generated via random seed scripts in `track_s/scripts/generate_synthetic.py`.

