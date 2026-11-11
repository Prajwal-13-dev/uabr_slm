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
D8 Model: d_ff = 2048 -> exactly 39,707,136 parameters (A6). Pretraining on Kaggle 2x T4 with DDP,
   one exact pass over train.bin (fixed block permutation) so every exposure copy is seen exactly once.
   Recipe defaults: 131k tokens/step, AdamW (0.9, 0.95), wd 0.1, lr 1e-3 -> 1e-4 cosine, 500 warm-up, fp16.
D9 Retrieval windows: 60 words, stride 30, each prefixed with its article title ("<title>: ...").
   BM25 (k1 0.9, b 0.4), hashed bag of words, stop words removed, no stemming; top-20 cached per question.
   RETRIEVAL CORPUS = the 1B-token pretraining subset from notebook 01 (PopQA forced + random fill, each
   article once), NOT full Wikipedia. Articles outside the subset are not retrievable; distractors come only
   from the subset. Full-Wikipedia retrieval (~110-125M windows) is a different, unimplemented configuration.
   Notebook 02 states this, records it in meta.json ("retrieval_corpus": "pretraining_subset") and checks it in its gate.
D10 Evidence overflow rule: use top-3 windows if question + windows fit 512 tokens, else top-2, else top-1;
    the number of windows used is logged per question.
D11 G1 prompt format = ASSUMPTION (team did not answer): relation cloze template + 3 same-relation TRAIN facts
    as demonstrations (LAMA-style); evidence condition = windows before the same prompt. Change in notebook 04 if needed.
    G1 criteria as implemented: (a) pilot closed-book EM - relation prior >= 5 pp with clustered-CI lower bound > 0;
    (b) EM rises with popularity (top - bottom bin) or exposure (High - Control), CI lower bound > 0 on all in-main val;
    (c) P(b_quality > 0) < 90%; (d) >= 1 retrieval-damage case.
D12 Evidence selection: walk the cached top-20 in rank order, skip any window overlapping an already chosen
    window of the same article (start offsets < 60 words apart), take up to 3; then apply D10 (3 -> 2 -> 1).

========================================
16. G1 RESULT AND OPTION A (Nov 10 2026)
========================================
G1 v1 FAILED: pilot closed-book EM 1.2% vs relation prior 9.6% (-8.4 pp [-11.4, -5.4]).
Diagnosis: (i) cloze+3-demo prompt -> demonstration copying (59% exact copies; one answer per relation for >=94%
of questions); (ii) randomized +0/+1/+4/+16 article-copy exposure had NO effect on gold-answer log-prob
(High-Control -0.15 nats [-0.57, +0.29]); (iii) evidence raises gold log-prob +0.66 nats [0.56, 0.77].
Also a bug: generation stopped at any '.', cutting names at initials ("John H").
Decision (team): Option A = authentic fact-sentence re-injection by continued pretraining (notebook 03b):
  A1 4,000 subjects (all splits, in_main), stratified relation x popularity, 1,000 each at 0/16/64/256 injected
     sentences per fact; doses REPLACE the old exposure groups for all analyses.
  A2 Repeat unit: all authentic corpus sentences naming the subject and containing a gold alias (subject article,
     object article, top-20 BM25 window articles), cycled across copies. String-match proxy; 100-row audit exported.
  A3 Continued pretraining from model_final.pt: ~25% fact sentences / ~75% replayed pretraining blocks;
     fresh AdamW, lr 2e-4 -> 2e-5 cosine, 50 warm-up (defaults); forgetting gate: val loss rise <= 0.05 nats.
Notebook 04 v2: zero-shot cloze (no demonstrations), stop only at newline, fixed answer cleaning; ranking probe
(gold vs 30 same-relation distractors, with vs without subject) and verbatim probe; G1b = dose 256 - dose 0.
If G1 still fails: verbatim rises but ranking/EM flat = memorised, not extractable; nothing rises = capacity ->
instructor decision (larger model / pretrained model).

========================================
17. G1 v2 RESULT AND MIXED TRAINING (Nov 10 2026)
========================================
03b gate PASS: val loss 2.8968 -> 2.9341 (+0.037); verbatim answer log-prob d256-d0 +3.40 nats.
04 v2: verbatim completion 13% -> 96% at d256 (+82.8 pp [+77.8, +87.7]); ranking acc@1 +11.4 pp [+3.4, +19.5] BUT
+7.7 pp [+1.8, +13.3] of it also appears without the subject (answer-frequency boost); subject-specific part
+3.7 pp [-4.0, +11.6] (n.s.). => memorised, not extractable. Correction: the v2 G1b PASS was confounded.
Decision (team): notebook 05 = mixed training.
  M1 In-main train split into half A (closed-book Q&A + RAFT evidence reading) and half B (never Q&A; routing
     labels come from half B). Grouped by subject, stratified relation x dose. Val/test untouched.
  M2 Formats: <USER> q <ASSISTANT><ANSWER> a<EOS> and <USER> q <ASSISTANT><CALL><EVIDENCE> windows</EVIDENCE> a<EOS>;
     loss on answer + <EOS> only (not question, evidence or action token).
  M3 Two arms with identical Q&A data: M = base + 03b injection stream + half-A Q&A mixed; S = 03b model + half-A
     Q&A + replay (3x). theta0 = arm with higher subject-specific val EM (rule fixed in advance; tie -> M).
  M4 G1 v3 (theta0, val facts never seen as Q&A): (a) dosed val EM - prior >= 5 pp, CI lo > 0, and
     EM - EM(subject replaced by "the subject") CI lo > 0; (b) subject-specific EM or ranking d256-d0 CI lo > 0;
     (c) P(b>0) < 90%; (d) >= 1 retrieval damage.
  Defaults: 8 passes over half A; RAFT oracle swap p=0.5; lr 2e-4 -> 2e-5, 50 warm-up.

========================================
18. NOTEBOOK 05 RESULT, theta0 SWITCH, NOTEBOOK 06 DESIGN (Nov 10 2026)
========================================
05 result: G1 v3 FAIL formally (theta0=S by pre-set rule: dosed val EM-prior +3.98 pp [+1.57, +6.50]; S lost accuracy
on d256 facts, relation-adjusted -7.7 pp [-13.4, -1.6]). Knowledge concentrated in country (+31 pp) and capital
(+29 pp); none for person/work relations; no within-relation popularity trend (-0.6 pp [-4.5, +3.2]).
Arm M: dosed EM-prior +5.53 [+3.05, +8.10]; dosed subject-specific +7.30 [+4.79, +9.87]; all-val EM-prior +3.48 [+2.26, +4.77].
DEVIATION (team decision, Option 2): theta0 = arm M instead of the pre-set rule's arm S, because S's d256 errors would
leak into half-B routing labels. Dose-based exposure claims are dropped.
Notebook 06 design (team decisions):
  R1 Labels (A1) with frozen theta0=M on half B (training), val (selection/gates), test (sealed, for later retention eval).
  R2 b = 0 questions kept; label CALL iff b_quality - lambda > 0. Lambda from a 1/16 grid chosen on val as the value whose
     labels best agree (balanced accuracy) with the realised greedy outcome "evidence strictly helps" (my rule).
  R3 Both router formats trained on half B (action: <ANSWER> vs <CALL> at <ASSISTANT>; answer_slot: <CALL> vs the model's
     own first answer token); theta_route,preSFT = higher val AUROC.
  R4 G6: val AUROC >= 0.65 GO; 0.58-0.65 continue with warning; < 0.58 STOP. Relation-only baseline reported, not required.
06 run 1 (Nov 10): G1c/G1d (arm M) PASS; lambda 0.1875 (19% CALL labels); G4 Spearman 0.579 [0.541, 0.616];
G5 hidden probe AUROC 0.802 [0.775, 0.823]; router [action] val AUROC 0.836 [0.815, 0.856] (GO), answer_slot 0.819,
relation-only 0.759. Outcomes: neither 75%, only-call 14%, both 9%, only-answer 2% -> the router mainly learns when
retrieval helps. PROBLEM: router training cut closed-book EM 12.4% -> 3.2% (labels stale).
Team fix (06 v2): R5 frozen tied embedding matrix + half-A Q&A replay (answer-token loss) during router training;
checkpoints qualify only if closed EM within 1 pp of theta0 (300-question val subset); hard gate G6b: closed AND
evidence EM of the routed model within 1 pp of theta0 (500 val). R6 2-logit loss kept and declared: the policy compares
z_CALL vs z_ANSWER at <ASSISTANT> (free-generation emission 0%, A5).

========================================
19. NOTEBOOK 06 v2 RESULT + NOTEBOOK 07 DESIGN (Phase 5 SFT-only pilot, Nov 10 2026)
========================================
06 v2: all gates PASS. lambda 0.1875; router [action] val AUROC 0.838 [0.815, 0.858] (GO); closed EM 12.4 -> 11.6 %,
evidence EM 21.6 -> 22.4 % (G6b preserved); free-generation emission 0 % (declared, A5).
07 protocol = uabr-slm-main docs/pilot_decision.md, unchanged: Dolly-15k + Alpaca-cleaned (pinned HF revisions),
plain SFT format <USER> instruction<ASSISTANT> response<EOS> (no action tokens, A5 primary), <= 512 tokens, ~6M tokens,
1 epoch, 500 held-out instruction examples, shuffle seed 2026; AdamW (0.9, 0.95), wd 0.01, LR 5e-5 constant after 30
warm-up, batch 32, clip 1, fp16, all weights train; seeds 11 and 29; checkpoints 25 / 50 / 100 %. Start = theta_route,preSFT
(06 router_action.pt) = E0 and the 0 % point. Evaluation on all labelled in_main validation questions (test sealed).
Decision rule unchanged (GO / RECALIBRATION_ONLY / NO_CHANGE on 100 % minus 0 %: AUROC, acc at re-picked 10 % budget,
acc at frozen 10 % threshold). Declared adaptations: (1) paired bootstrap resamples subject clusters (A9);
(2) AUROC is against the frozen theta0 labels (A3a), budget accuracy uses fresh greedy closed/evidence EM of each
checkpoint (A3b); (3) instruction rows stating a val/test PopQA fact (subject AND gold alias) are removed before the
token budget is filled (sft_overlap_audit.csv). A4: flips by pre-SFT margin quintile at boundary 0 (m_i = |z_C - z_A|)
and at the frozen 10 % threshold, AUROC(-m -> flip), A->C / C->A counts. A5: emission rate, P(<ANSWER>)+P(<CALL>),
action-token rank, two-way AUROC, re-tuned 10 % threshold.

========================================
20. NOTEBOOK 07 RESULT (Phase 5 pilot, Kaggle run Nov 11 2026, code 9262cb421a)
========================================
SFT: 34,884 examples, 6,000,100 tokens, 1,091 updates per seed (~7.2 min); 1,742 instruction rows removed by the overlap
audit (234 val/test facts). Instruction val loss 2.989 -> 2.383 / 2.385 (SFT learned). 0 % point = 06 router exactly.
100 % minus 0 % (1,754 val questions, clustered paired bootstrap):
  seed 11: AUROC -0.137 [-0.170, -0.108] | acc@10% re-picked -0.132 [-0.146, -0.115] | acc@10% frozen -0.132 [-0.148, -0.114]
  seed 29: AUROC -0.132 [-0.163, -0.101] | acc@10% re-picked -0.128 [-0.144, -0.113] | acc@10% frozen -0.128 [-0.145, -0.112]
DECISION: GO (AUROC alone is significant in both seeds). AUROC 0.838 -> 0.701 / 0.707.
CAVEAT: SFT also collapsed the answer paths (closed EM 11.3 -> 1.5 / 1.9 %, evidence EM 22.2 -> 2.3 / 3.4 %), so the
accuracy drops mostly measure answer damage, not routing; AUROC vs frozen labels is the clean retention measure.
A4 supported: AUROC(-m -> flip) at 100 % 0.741 [0.716, 0.766] / 0.765 [0.735, 0.792]; flips mostly ANSWER -> CALL
(333 vs 67; 234 vs 75). Default call rate 12.9 % -> 28.1 / 21.9 %.

========================================
21. PHASE 6 DESIGN (team decisions, Nov 11 2026, before any Phase 6 result)
========================================
07 check: post-SFT answers are sentences ("X is a Japanese actress and model"); of the 199 closed-book / 390 evidence
questions correct at 0 %, only 18.6 % / 23.1 % still contain the old answer at 100 % -> SFT really damages answering, not
only the format. Overlap audit (1,742 rows) is mostly generic subjects ("Day", "Boston", "Time"); removal kept.
P1 Replay examples are ACTION-TOKEN ONLY (2-logit CE at <ASSISTANT>, target = frozen theta0 label y).
P2 Primary endpoint = routing AUROC vs frozen theta0 labels (A3a) + decision agreement with E0; fresh-reward utility
   (EM and answer-containment) secondary (A3b).
P3 Budget as uabr-slm-main: select distinct examples worth 25 % of the pool tokens, cycle to 5 % of instruction tokens
   (spread evenly over the 1,091 updates). Pool = half-B questions, scored once with theta_route,preSFT.
P4 Same ANSWER:CALL token share for every arm (= pool share); methods rank within each class.
P5 E3 Hybrid-CASR-adapted uncertainty = theta0's closed-book first-token entropy (router entropy would equal BPR).
P6 E4 OPR-adapted: current model re-scores the pool every 100 updates; keep prompts whose current decision (boundary 0)
   equals y; random selection at the same budget/ratio.
P7 L = L_SFT + beta * L_replay; beta from {0.1, 0.3, 1, 3} on validation with E2 random seed 11: smallest beta within
   0.01 AUROC of the best; fixed for all arms.
P8 Notebook 08 = validation only (preliminary); notebook 09 opens test once for all arms with the claim rule fixed in advance.

========================================
22. NOTEBOOK 08 RESULT (Phase 6 validation, Kaggle run Nov 11 2026) — all validity checks PASS
========================================
Pool 4,098 half-B prompts (50,751 tokens; CALL 19.0 %); tau 2.002; each arm selects 12,687 tokens, replays 300,005-300,017.
Beta (E2 seed 11, val AUROC): 0.1 0.8167 | 0.3 0.8179 | 1 0.8226 | 3 0.8205 -> beta 0.1. Replay loss reaches ~0 by step 500
for every static arm (each of ~1,000 distinct prompts is replayed ~24 times).
Validation AUROC vs frozen labels (seed 11 / 29); E0 0.838, E1 0.701 / 0.707:
  E2 random 0.817 / 0.804 | E3 CASR 0.797 / 0.794 (default call 55 / 57 %) | E4 OPR 0.823 / 0.824
  E5 BPR 0.639 / 0.692 (default call 3.5 / 3.9 %) | E6 benefit 0.774 / 0.794 | E7 UABR 0.698 / 0.731 (default call 5.4 / 6.2 %)
Reading: every arm except BPR and UABR recovers most of the lost AUROC; the two margin-driven arms (BPR, UABR) recover
none or little and push the default decision toward ANSWER. Same direction as uabr-slm-main Phase 6 (UABR < random).
Closed / evidence EM stay collapsed in every arm (2.7-5.8 % / 4.2-6.9 %).

========================================
23. EXTENSION: SELF-RAG-ADAPTED CRITIQUE TOKEN (notebook 10; team decisions Nov 11 2026, before any result)
========================================
Kept: [Retrieve] (= existing router) and [IsRel] (new tokens <REL>/<IRR>, ids 12000/12001, appended; tokenizer file
unchanged); [IsSup] merged into IsRel; [IsUse] dropped. Critique slot = end of the single-window evidence prompt
(<USER> q<ASSISTANT><CALL><EVIDENCE> w</EVIDENCE>), scored as z_REL - z_IRR, never generated (answer format unchanged).
S1 Training questions: half B (not half A: theta0 memorised half-A answers). Non-overlapping top-10 windows; label =
   subject AND gold alias (auto proxy, G2 precision 79 % / recall 81 %); all positives + 3 negatives each (half hard).
S2 Trainable: transformer body + the 2 new rows; old embedding rows frozen; half-A Q&A replay + half-B router replay.
   Checkpoint = best critique AUROC on 300 val questions not in G2, among checkpoints with closed EM within 1 pp and
   routing AUROC within 0.01 of theta_route. Gate G10 on 500 other val questions (closed + evidence EM, full-val routing).
S3 RQ-S1 comparator = best ANSWER-FREE baseline (BM25 score/rank, subject mention, own article, lexical LR,
   theta_route hidden probe; chosen on the selection subset). The gold-using rule is a label-quality reference only.
   Rule: PASS iff critic - best baseline on G2 has clustered 95 % CI lower bound > 0.
S4 G2 labels are AI-drafted: used now; the team's 50-row blind check (Cohen's kappa) must be done before reporting.
RQ-S2 (per-window selection: a top-3 concat, b max P(REL) of top-5, b2 gated closed-book if max P(REL) < 0.5,
c baseline selection, c1 BM25 top-1, d best-of-5) computed always, interpreted only if RQ-S1 passes.
RQ-S3: 07 SFT recipe from the critic (seeds 11, 29); IsRel vs routing AUROC drop, flips by margin.
Notebook 09 (test) still waits for the 08b decision (methods A/B/C).
