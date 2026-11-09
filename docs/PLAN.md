PROJECT HANDOVER — UABR-SLM (course project, team "Straw Hats", PES University AFML)
Team: Pradeep T, Prajwal A, Nikitha Uppar, Sakshi Kiran

========================================
1. RESEARCH QUESTION (FROZEN — do not change)
========================================
Does an answer-versus-call policy learned BEFORE instruction tuning keep its decision quality AFTER supervised fine-tuning (SFT), and can Utility-Aware Boundary Replay (UABR) protect that policy from forgetting better than random, Hybrid-CASR-style, on-policy (OPR), boundary-only and benefit-only replay at equal replay and call budgets?

- Contribution A: policy retention (does SFT damage a learned ANSWER/CALL policy?)
- Contribution B: UABR as a policy-retention mechanism during SFT (NOT a way to make the router learn faster)
- UABR priority: s_i = exp(-m_i/tau) * |b_i^net|
  - m_i = |z_CALL - z_ANSWER| from the pre-SFT routing checkpoint (normalized; fixed during SFT for the first experiment)
  - |b| = replay importance; sign(b) = correct action (both CALL and ANSWER examples can be high priority)
- Claims limited to a ~40M scratch-trained model; negative results (e.g. "SFT does not hurt routing at 40M") are valid outcomes.

========================================
2. HISTORY / WHAT FAILED
========================================
Three code folders:
- uabr_slm: original v1 (synthetic facts, statements only, exposure tiers 40/120/300, CALL = exact-ID lookup with R(call)=1.0). Phase 4 failed: memorization (train loss 0.0003), routing AUROC ~0.49–0.60.
- uabr-master: v2 theory redesign (Phase 3.5 format adaptation, in-memory BM25, graded reward add_graded_reward, groundedness filter, preflight_gate.py --project). Missing src/data/.
- uabr-slm-main: ACTIVE MAINLINE. data_v2 with ~30% Q&A injected into pretraining, precomputed keyword search with distractors, CALL reading practice (copy_example, loss only on answer tokens), diagnose_routing.py, full Phase 5 pilot harness (run_sft.py, sft_batches.py, pilot_eval.py, docs/pilot_decision.md). Preflights (Nov 7 2026): "question recall 0.25→1.018" (METRIC DEFINITION MUST BE AUDITED — recall can't exceed 1), reading accuracy 0.498→0.797.

Diagnosed root causes of the Phase 4 failure:
1. Routing SFT supervised the full sequence → taught the answers → pre-computed labels went stale → memorization.
2. Exposure tiers 40/120/300 were above saturation (89% recall at 40) → no graded known/unknown signal.
3. Familiarity was per entity, not per fact → router could learn name recognition.
4. Never checked that a routing signal exists (probe/confidence) before training the router.
5. Binary EM makes |b| ∈ {0,1} → UABR collapses to BPR.

Decision: do NOT rebuild. Continue in uabr-slm-main. Port from uabr-master: add_graded_reward, groundedness filter, preflight_gate --project. Make a clean git checkpoint first and reproduce the preflights.

========================================
3. DATASETS (no synthetic facts)
========================================
- PRIMARY: PopQA (Mallen et al., ACL 2023). 14,267 questions, 16 Wikidata relations, fields: subj, subj_id, prop, prop_id, obj, obj_id, s_aliases, o_aliases, s_pop, o_pop, s_wiki_title, o_wiki_title, question, possible_answers. Questions are template-generated from real Wikidata facts (not human-written — state this). Does NOT include passages → build our own retrieval corpus.
- Retrieval corpus: our existing ~1B-token filtered Wikipedia, fixed snapshot, split into short passage windows.
- Supplementary: SimpleQuestions-Wikidata (human-written questions) for overlapping relations → wording-transfer test only.
- SFT data: existing Dolly/Alpaca pipeline; audit overlap with PopQA test facts.
- NOT core: Natural Questions, HotpotQA (too hard for 40M).
- If power is too low: add more real Wikidata triples for the same 16 relations using PopQA's templates (same construction as PopQA itself), or EntityQuestions.
- Note: an earlier advisor claimed arXiv 2603.11513 ("Can Small Language Models Use What They Retrieve?") has a correction notice on GitHub — UNVERIFIED (arXiv shows only v1). Check its README before citing its numbers.

========================================
4. MODEL ARCHITECTURE (keep existing)
========================================
- Decoder-only, LLaMA-style, ~40M params: 8 layers, d_model 512, 8 heads (head dim 64), SwiGLU FFN, RoPE, pre-norm RMSNorm, no biases, tied embeddings.
- Tokenizer: byte-level BPE 12k (keep; 16k max). Special tokens: <BOS> <USER> <ASSISTANT> <ANSWER> <CALL> <EVIDENCE> </EVIDENCE>.
- Context 512. Extend to 1024 ONLY if the pilot shows top-2/3 short windows don't fit/hit; extend by brief continued pretraining (~5% tokens, RoPE interpolation).
- Why decoder-only: routing = next-token decision; routing, answering, reading and instruction following share weights (necessary for SFT interference to exist); replay = extra sequences in SFT batches; matches Self-RAG/Toolformer/LaCy.
- Why 40M: Chinchilla ~20 tokens/param → ~0.8B tokens ≈ our 1B; ~5 h pretrain on Kaggle T4 (54k tok/s); knowledge capacity ~1–2 bits/param (Allen-Zhu & Li, Physics of LMs 3.3) → real known/unknown mix.
- Scale to ~100–125M (12×768) only if the 40M pilot fails; that changes scope → ask instructor.

========================================
5. PIPELINE ORDER
========================================
General pretraining → real factual exposure (measured; optional authentic-Wikipedia-passage re-injection at controlled frequencies) → evidence-reading preparation (RAFT-style on PopQA TRAIN questions + real retrieved passages with distractors) → freeze θ0 → counterfactual benefit → routing training (action-token only) → θ_route,preSFT → SFT (7 arms) → retention evaluation.

========================================
6. KEY DEFINITIONS
========================================
Graded benefit (frozen θ0, canonical answer string for scoring, aliases for correctness):
  p̄_A = exp(mean_t log P(a_t | q, a_<t))
  p̄_C = exp(mean_t log P(a_t | q, D, a_<t))   D = BM25 top-2/3 short windows
  b_quality = p̄_C − p̄_A ∈ [−1,1]
  b_net = b_quality − λ c_i  (c_i normalized to [0,1]; λ chosen on validation)
  label: CALL if b_net > 0 else ANSWER
- MUST validate b against actual EM/F1 change on validation; if it doesn't track, use alias-aware F1 difference.
- Report P(|b|≤0.01/0.05/0.10) and the b distribution by popularity, relation and exposure.
- Report the 4-way outcome table: both correct / only ANSWER / only CALL / neither. Analyze "neither" separately (it inflates the ANSWER class).
- Oracle evidence = passage that actually SUPPORTS the relation (not just contains the answer string); upper-bound analysis only. Realistic BM25 is used for the main labels.
- Measure retrieval damage (correct closed-book → wrong with evidence).

Routing loss: cross-entropy on the <CALL>/<ANSWER> logits at the action position only; all other labels = −100. The router decides BEFORE seeing evidence.

Replay loss: L_total = L_SFT + β L_replay; replay examples are action-token-only; β tuned on validation; identical replay budget across arms.

========================================
7. KNOWN CONFOUNDS TO CONTROL
========================================
1. Relation-prior guessing: some relations (sport, country, religion, genre, occupation) have dominant answers. Report a majority-answer baseline per relation; "known" = beats the relation prior; report per-relation AUROC.
2. Context budget: top-5 × 100 words doesn't fit 512 tokens → use top-2/3 ~60-word windows; measure hit rate at that size.
3. Statistical power: ~2.1k test questions × ~5% known ≈ 100 known examples → estimate power from the pilot; expand with Wikidata triples if needed.
4. Exposure ledger: relation-supported counting is hard → use sentence-level subject+object alias co-occurrence + relation keywords; hand-audit ~200, report precision; call it observational. Injected passages expose other facts too → log them.
5. Popularity (s_pop) ≠ corpus exposure ≠ action utility — keep the three signals separate.
6. Reading must be revalidated on real Wikipedia text (0.797 was on templated records).

========================================
8. SPLITS
========================================
- Group by subj_id; stratify by relation and popularity: 70 train / 15 validation / 15 test.
- Extra OOD: 2–3 held-out relations; SimpleQuestions-Wikidata wording transfer.
- Test never enters routing training or the replay buffer. Thresholds, λ, τ, β, ε are tuned only on validation.

========================================
9. EXPERIMENT ARMS (all from the same θ_route,preSFT; identical SFT data/steps/optimizer/seeds)
========================================
E1 SFT only | E2 uniform random replay | E3 Hybrid-CASR-style (uncertainty + ANSWER/CALL class balancing; "adapted") | E4 OPR (current model's routing decisions on old prompts, kept if they pass the frozen-utility filter) | E5 boundary-only (BPR) | E6 benefit-only | E7 UABR.

========================================
10. EVALUATION
========================================
Primary: net utility at a fixed 10% call budget (also 5%, 20%) using ACTUAL alias-aware correctness:
  U(π) = mean[ R_i(π(q_i)) − λ c_i 1[π=CALL] ]
plus ΔAUROC = AUROC_postSFT − AUROC_preSFT on the same held-out questions.
Also: AUPRC, regret = mean[max(U_A,U_C) − U_π], CALL frequency, Brier, ECE, answer-only EM/F1, evidence-conditioned EM/F1, SFT validation performance.
Routing baselines (deployable, pre-retrieval): generated-answer confidence threshold, generated-answer token entropy, hidden-state linear probe, PopQA s_pop threshold, learned token router. Gold-answer log-prob is for offline labels ONLY.
Stats: ≥2–3 seeds, paired bootstrap 95% CIs, per-relation / per-popularity / per-exposure breakdowns, macro and overall averages, paired comparisons UABR vs each baseline.

========================================
11. DECISION GATES
========================================
G0 Reproduce: git checkpoint, reproduce preflights, define the "1.018" metric.
G1 Pilot (500 PopQA questions, stratified by popularity/relation, existing base checkpoint): closed-book EM roughly ≥5–10% ABOVE the relation-majority baseline, accuracy rises with popularity/exposure, b distribution not almost all positive, some retrieval damage exists. Fail → re-inject authentic Wikipedia text and repeat → still fail → discuss a larger/pretrained model with the instructor.
G2 Retrieval: BM25 top-2/3 windows actually support the relation for a useful share (manual audit of 100–200).
G3 Reading: evidence-conditioned accuracy on held-out real passages is acceptable.
G4 Graded benefit tracks the EM/F1 change.
G5 Routing signal: probe/confidence AUROC clearly > 0.5 on held-out (a diagnostic, not a hard upper bound).
G6 Pre-SFT policy works on held-out → run the Phase 5 SFT-only pilot (existing GO / RECALIBRATION_ONLY / NO_CHANGE protocol).
G7 GO → full 7-arm study. NO_CHANGE → report it. All replay fails / UABR ≤ Hybrid-CASR or OPR → report honestly.

========================================
12. IMPLEMENTATION ORDER (minimum viable path)
========================================
1. Freeze/audit uabr-slm-main, port the 3 components.
2. popqa_loader.py + BM25 over the existing Wikipedia shards (short windows) + 500-question pilot with relation-majority baseline and the 4-way table. ← THIS DECIDES EVERYTHING
3. graded_reward.py (p̄_C − p̄_A, λ) + validation against EM/F1.
4. Reading data on real passages; reading gate.
5. routing_train.py (action-token only) + baselines + diagnose_routing.py.
6. Phase 5 SFT-only pilot.
7. On GO: hybrid_casr.py, opr.py, uabr_priority.py, replay_train.py, retention_eval.py, statistical_analysis.py.
8. Last, if time allows: full exposure ledger, authentic-text injection, SimpleQuestions-Wikidata transfer.
Keep data, checkpoints and indexes out of git; version code, configs, split manifests and experiment metadata.

========================================
========================================
13. AMENDMENTS (after review, Nov 8 2026)
========================================
A1 Benefit (replaces the §6 primary): sample k=8 answers per condition (T=0.7, fixed seed) from θ0.
   b_quality = mean alias-F1(CALL samples) − mean alias-F1(ANSWER samples) ∈ [−1,1], graded in 1/k steps.
   b_net = b_quality − λc. The likelihood proxy p̄_C − p̄_A is kept as a secondary analysis and must be correlated against b_quality.
A2 Add E0 = frozen θ_route,preSFT (reference, not an arm).
A3 Report separately: (a) POLICY RETENTION vs the frozen pre-SFT labels/decisions;
   (b) POST-SFT UTILITY using fresh ANSWER/CALL rewards from each post-SFT checkpoint.
A4 Mechanism check in the Phase 5 pilot: flip rate of pre-SFT decisions binned by margin m_i (E1). If low margin does not predict flips, the boundary term is unjustified — report it.
   Also report ANSWER→CALL and CALL→ANSWER flip counts per arm.
A5 Action-slot collapse: report (i) action-token emission rate under free generation, (ii) two-way AUROC from z_CALL−z_ANSWER only, (iii) metrics after re-tuning the threshold on validation to a 10% call rate.
   Declare the SFT format: PRIMARY = plain SFT with no action tokens (format interference); optional = SFT responses prefixed with <ANSWER>.
A6 Architecture: record the exact d_ff from the existing checkpoint config; print the exact parameter count (d_ff=2048 → ~39.7M). Measure real throughput before planning the timeline.
A7 Fairness: equal replay TOKEN budget; controlled/reported CALL:ANSWER ratio in each replay buffer; same decoding and retrieval budget everywhere.
A8 OPR-adapted: state whether the buffer is built once or refreshed every N steps (default: refreshed every N steps = true on-policy). Label E3 "Hybrid-CASR-adapted".
A9 Stats: subject-clustered paired bootstrap. The 500 pilot questions are dev-only and never in test. G1 thresholds stated in percentage points over the relation-majority baseline, with CI.
A10 Claims: no "first" claims; position against OPR, Hybrid-CASR, Prioritized Experience Replay and "Retaining by Doing".
   Hypothesis: at an equal replay-token budget, boundary × utility priority preserves the pre-SFT routing policy better than random, boundary-only, benefit-only, Hybrid-CASR-adapted and OPR-adapted replay.
A11 arXiv 2603.11513: the author's GitHub reportedly withdraws the v1 oracle results (oracle passages were chosen by answer-string containment). Do not cite the v1 numbers; use it only as motivation for requiring relation-supporting evidence.

========================================
14. RESTART NOTE (Nov 8 2026)
========================================
The repo was restarted from scratch on Nov 8 2026. This supersedes the §2 decision "do NOT rebuild".
Old v1 code is preserved on branch archive/v1 (tag v1-final). Components to re-implement fresh
(not port): graded reward, groundedness filter, preflight gate. G0 now = verify the base checkpoint,
tokenizer and Wikipedia shards load in the new code, and record d_ff / exact param count (A6).

========================================
15. TEAM DECISIONS (Nov 9 2026, before notebook 01)
========================================
D1 Splits: custom (PopQA ships one 14,267-row file). Grouped by subject + identical question text,
   stratified relation x popularity, 70/15/15, seed 1234; OOD relations religion, mother, capital of.
D2 Ambiguous questions (same text, different same-named subjects; 1,689 rows = 11.8%): kept in data,
   in_main = False -> excluded from routing labels, replay buffers and primary metrics; reported separately.
   Relation prior is fitted and scored on in_main rows.
D3 Pilot (500) re-drawn from in_main validation questions only (splits unchanged).
D4 Pretrain the 40M model from scratch in the new code (old uabr-slm-main checkpoint not reused).
D5 Corpus: HF wikimedia/wikipedia 20231101.en, revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa.
   All PopQA subject + object articles (forced) + random articles to 1.0B natural tokens.
   The same natural articles are the BM25 retrieval corpus.
D6 Exposure: every PopQA subject article found is assigned (stratified by primary relation x popularity bin,
   25% each) to Control / Low / Medium / High = +0 / +1 / +4 / +16 ADDITIONAL copies of its first
   1,024 tokens. Copies are ON TOP of the 1B natural tokens, shuffled uniformly into the stream.
D7 Tokenizer: byte-level BPE 12k; 9 special tokens with fixed ids
   <PAD>=0 <BOS>=1 <EOS>=2 <USER>=3 <ASSISTANT>=4 <ANSWER>=5 <CALL>=6 <EVIDENCE>=7 </EVIDENCE>=8.
   Pretraining documents stored as <BOS> text <EOS>; no Q&A-formatted text in pretraining.
OPEN (decide before notebook 04): prompt format for the closed-book G1 pilot, since the base model
   never sees Q&A format in pretraining (cloze-style templates vs few-shot vs small format adaptation).
