# UABR-SLM

Utility-Aware Boundary Replay for retaining an ANSWER/CALL routing policy through SFT in a ~40M decoder-only model.
Team Straw Hats (PES University, AFML): Pradeep T, Prajwal A, Nikitha Uppar, Sakshi Kiran.

**The full plan is in [docs/PLAN.md](docs/PLAN.md)** (research question, definitions, arms, gates, amendments A1–A11).
Old v1 code (synthetic facts) is on branch `archive/v1`, tag `v1-final`.

## Layout
```
configs/            YAML configs (model, retrieval, pilot, routing, sft, replay)
src/uabr/data/      PopQA loader, splits, aliases, relation-majority baseline
src/uabr/retrieval/ Wikipedia short-window chunking + BM25 index
src/uabr/model/     LLaMA-style 40M decoder (8x512, SwiGLU, RoPE, RMSNorm, tied emb.)
src/uabr/train/     pretrain / reading (RAFT-style) / routing (action-token only) / SFT
src/uabr/eval/      alias-F1/EM, sampled benefit (A1), AUROC, utility@budget, bootstrap
src/uabr/replay/    random, Hybrid-CASR-adapted, OPR-adapted, BPR, benefit-only, UABR
scripts/            entry points (one per pipeline step)
tests/              unit tests
docs/               PLAN.md, gate reports
data/               local only (git-ignored) except split manifests
```

## Next step (PLAN §12.2, decides everything)
`popqa_loader.py` + BM25 over the Wikipedia shards (top-2/3 ~60-word windows) +
500-question dev pilot with relation-majority baseline and the 4-way outcome table → gate G1.
