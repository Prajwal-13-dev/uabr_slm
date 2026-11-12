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

========================================
24. NOTEBOOK 10 RESULT (Kaggle run Nov 12 2026, code 7e272d955b) — status FAIL (G10)
========================================
Data: 40,878 half-B windows (10.8 % relevant) -> 17,724 training windows; 1 window dropped (too long); G2 597 windows / 199
questions (100 relevant), text identical to the audit; auto rule vs G2: precision 79.4 %, recall 81.0 %.
Best answer-free baseline (selection subset): lexical LR 0.870 (subject mention 0.864, hidden probe 0.853, BM25 0.794).
Critic: best qualifying step 600 (selection AUROC 0.875); val auto AUROC 0.867.
G10 FAIL: routing AUROC 0.8384 -> 0.8190 (-0.019, tolerance 0.01); closed 11.8 -> 12.4 %, evidence 23.6 -> 25.2 % (fine).
The 300-question selection check (-0.0085) did not catch the full-validation drop; routing AUROC fell steadily with training.
RQ-S1 NOT SUPPORTED: critic - lexical LR on G2 +0.021 [-0.023, +0.066]; val auto +0.008 [-0.008, +0.025].
  Gold-using rule on G2 AUROC 0.884 (reference). Shortcut subset (114 answer-present windows, 32 not stating the fact):
  critic 0.761 vs lexical 0.733 (no CI). Uncertain G2 windows dropped: critic 0.896.
RQ-S2 (not interpreted per rule): critic selection - top-3 concat -3.92 pp [-5.83, -2.14]; critic - baseline selection
  +0.69 pp [-0.68, +2.08]; gated -6.05 pp [-7.92, -4.16] (closed-book chosen for 68 %).
RQ-S3: SFT runs done (1,091 updates each); table in rq_s3_retention.csv (not in the log).
RQ-S1 tables (G2 / val auto AUROC): critic 0.872 [0.830, 0.908] / 0.867; lexical LR 0.851 / 0.860; own article 0.842 / 0.825;
  subject mention 0.832 / 0.855; hidden probe 0.832 / 0.840; BM25 score 0.802 / 0.795; BM25 rank 0.556 / 0.601.
RQ-S2 EM: top-3 concat 23.2 % | critic select 19.3 % | gated 17.1 % | baseline select 18.6 % | BM25 top-1 16.8 % |
  best-of-5 27.2 % (headroom +4 pp not captured). Retrieval damage 1.3 % (concat) vs 4.3 % (critic select).
RQ-S3 (critic model, 100 % minus 0 %): IsRel G2 -0.109 [-0.156, -0.065] (s11), -0.122 [-0.168, -0.078] (s29);
  IsRel val -0.104 / -0.111; routing -0.137 [-0.171, -0.104] / -0.127 [-0.162, -0.094].
  Share of above-chance skill lost: IsRel 28-33 %, routing 40-43 %. After SFT the critic (0.75-0.76) falls BELOW the
  answer-free lexical baseline (0.85). Default positive rate: IsRel stable (14.6 -> 13 %), routing drifts 11.6 -> 39 / 30 %.
  Small margin predicts flips: IsRel AUROC 0.83-0.87, routing 0.67-0.71.

========================================
25. NOTEBOOK 11 DESIGN: post-hoc analyses, no training (team decision Nov 12 2026, before any result)
========================================
(1) Readout vs representation: hidden state h (post-RMSNorm) at the decision slot; token score = h.(e_CALL - e_ANSWER) is a
    fixed linear readout (tied embeddings). Compare token vs linear probe re-fitted on post-SFT h (half-B labels ->
    validation) vs the pre-SFT probe applied unchanged; label-budget curve 50..all. Same for IsRel (critic; half-B windows
    -> G2 / validation windows). Rule: recovered share = (probe_post - token_post)/(token_pre - token_post) >= 0.8
    readout-dominated, <= 0.2 representation-dominated, else mixed (n/a if damage < 0.02). Secondary: probe_pre - probe_post.
(2) Weight interpolation theta_route -> theta_SFT (alpha 0..1 step 0.1, per seed) and task vector theta_SFT + beta
    (theta_route - theta0), beta 0.5 / 1; critic interpolation alpha 0..1 step 0.2. Report alpha* (largest alpha with routing
    AUROC >= E0 - 0.02) with SFT gain kept, and routing AUROC at the smallest alpha keeping >= 90 % of the SFT loss gain.
(6) Threshold recalibration on n = 50/100/200/500 labelled half-B questions (5 draws; max balanced accuracy) vs default 0
    vs re-fitted probe with the same labels: balanced accuracy, call rate, agreement with E0, utility with theta0 rewards.

========================================
26. NOTEBOOK 11 RESULT, part 1 (Kaggle run Nov 12 2026) — status PASS (all consistency checks)
========================================
Interpolation theta_route -> E1 (seed 11 / 29; EM on 500 val questions; E0: AUROC 0.838, loss 2.989, closed 9.4 %, evidence 17.6 %):
  alpha 0.4: AUROC 0.829 / 0.828 | loss 2.605 (63 % of SFT gain) | closed 10.2 / 10.8 % | evidence 18.4 / 18.8 %
  alpha 0.5: AUROC 0.822 / 0.821 | loss 2.541 (74 %)               | closed 10.0 / 10.2 % | evidence 17.6 / 17.6 %   <- alpha*
  alpha 0.7: AUROC 0.790 / 0.789 | loss 2.446 (90 %)               | closed 5.8 / 6.0 %   | evidence 13.0 / 12.4 %
  alpha 0.8: AUROC 0.765 / 0.766 | loss 2.415 (95 %)               | closed 3.2 / 3.6 %
  alpha 1.0: AUROC 0.701 / 0.707 | loss 2.383 / 2.385              | closed 0.6 / 1.0 %   | evidence 1.0 / 2.0 %
  SFT gain is front-loaded in alpha, damage back-loaded: answering and routing hold up to alpha ~0.5.
Task vector E1 + beta (theta_route - theta0): beta 1: AUROC 0.812 / 0.809, loss 2.401 / 2.402 (97 % of SFT gain), closed
  3.8 / 4.4 %; beta 0.5: AUROC 0.779 / 0.779. No training, close to random replay (08 E2: 0.817 / 0.804); answers not restored.
Critic interpolation (IsRel G2 / routing / loss; alpha 0 = 0.872 / 0.819 / 3.081): alpha 0.4: 0.866 / 0.818 / 2.619 (66 %);
  0.6: 0.845 / 0.798 / 2.494 (84 %); 1.0: 0.762 / 0.682 / 2.385 (s11).
Readout (1) and recalibration (6) tables: in the output CSVs (not in the log).
Part 2 — readout vs representation (1) and recalibration (6):
Routing (val AUROC): token 0.838 -> 0.701 / 0.707; probe re-fitted on half-B hidden states 0.788 (pre) -> 0.789 / 0.791 (post);
  pre-SFT probe applied unchanged after SFT 0.51 / 0.51 (0.42-0.43 at 25-50 %). Recovered share 0.64 / 0.64 -> "mixed" by the
  fixed rule. But probe_pre - probe_post = -0.001 / -0.003: everything a re-fitted linear probe can read before SFT is still
  there after SFT; the unrecovered part is the probe's lower ceiling (0.788 < token 0.838; probe trained on the router's own
  training questions). The decision information moved to a new direction (frozen probe ~chance) and the token readout did
  not follow. Label budget: post-SFT probe 0.67-0.68 with <= 200 labels, 0.73 with 500-1,000, 0.79 with all 4,098 (pre-SFT
  probe ~0.78 already at 50) -> recoverable, but not cheaply.
IsRel (G2 / val auto): token 0.872 / 0.867 -> 0.763-0.750 / 0.763-0.756; re-fitted probe 0.863 / 0.868 -> 0.833-0.837 /
  0.844; recovered 0.64-0.72 (G2), 0.78-0.79 (val) -> "mixed", mostly readout; small real loss (probe drop 0.026-0.030 G2,
  0.024 val). Post-SFT re-fitted probe still below the answer-free lexical baseline on G2 (0.851).
Recalibration (balanced-accuracy threshold): E1 balanced accuracy 0.633 / 0.628 (default 0) -> 0.633-0.650 with 50-500
  labels; oracle threshold on val only 0.653 / 0.660 (E0 default 0.652, E0 oracle 0.780). Utility with theta0 rewards gets
  WORSE (call rate jumps to 40-57 %). Re-fitted probe + threshold: 0.60-0.67. -> Threshold recalibration cannot repair the
  damage (independent confirmation of the Phase 5 GO). Caveat: the balanced-accuracy objective favours high call rates.

========================================
27. NOTEBOOK 12 DESIGN: new methods (team decision Nov 12 2026, before any result)
========================================
0  Check: 5-fold GroupKFold logistic probe on the validation questions' own hidden states (E0, E1 s11/s29); recovered share
   with this ceiling, notebook 11 rule.
A1 SFT-prompt anchoring: 07 SFT recipe + gamma * softBCE(sigmoid(d_student), sigmoid(d_teacher)) at <ASSISTANT> of every SFT
   example (d = z_CALL - z_ANSWER; teacher = frozen theta_route). No labels, no extra data.
A2 Question-pool anchoring: same loss on 16 random half-B question prompts per update (teacher scores precomputed). No labels.
A2-critic: A2 from the notebook 10 critic, anchoring routing (half-B questions) and IsRel (half-B windows) slots, 16 + 16.
B  Readout repair of the 07 SFT models: only <ANSWER>/<CALL> embedding rows + final RMSNorm gain train (2-logit CE, half-B
   labels, lr 1e-3, 3 epochs) + SFT replay; best selection-subset AUROC with instruction val loss <= E1 + 0.02.
gamma = 1 fixed (not tuned). Selection subset 300 val questions (B only); results on the other 1,454 val questions (EM on
500); references rebuilt on the same questions: E0, E1, interpolation alpha 0.5, task vector beta 1. Primary: routing AUROC
vs frozen labels; paired clustered bootstrap vs E1 / interpolation / task vector per seed; A2-critic vs critic-SFT on IsRel.

========================================
28. NOTEBOOK 12 RESULT, part 1 (Kaggle run Nov 12 2026) — status PASS
========================================
Routing on the 1,454 evaluation questions (AUROC [95% CI] | default call | instruction val loss | closed / evidence EM, 500 q):
  E0            0.834 [0.812, 0.857] | 12.9 % | 2.989 | 12.0 / 21.6 %
  E1 s11 / s29  0.689 / 0.696        | 28.1 / 21.8 % | 2.383 / 2.385 | 1.0-1.4 / 2.4-3.2 %
  A1 (SFT-prompt anchoring, no labels, no data)   0.827 / 0.827 | 9.1 / 13.3 % | 2.388 / 2.390 | 1.2-1.8 / 4.4-7.2 %
  A2 (question-pool anchoring, no labels)         0.832 / 0.836 | 10.9 / 13.6 % | 2.387 / 2.388 | 4.4-4.8 / 8.6-9.6 %
  B  (readout repair: 2 rows + norm gain)         0.804 / 0.810 | 10.5 / 11.4 % | 2.383 / 2.384 | 2.2-3.0 / 2.2-3.2 %
  interpolation alpha 0.5                         0.815 / 0.814 | 14.9 / 12.5 % | 2.541 / 2.542 | 11.2-11.4 / 21.2-21.4 %
  task vector beta 1                              0.807 / 0.805 | 17.5 / 15.1 % | 2.401 / 2.402 | 4.6 / 11.4-11.6 %
Reading: A2 keeps the pre-SFT routing AUROC (0.834) with the full SFT gain and no labels; A1 nearly does with no extra data
at all. Both beat interpolation / task vector / readout repair on routing, and are above 08 random replay (0.817 / 0.804,
full val) - soft teacher targets instead of hard labels. Answering stays collapsed under anchoring (only interpolation keeps it).
B selection: best steps 325 / 375 (selection AUROC 0.852 / 0.855), instruction loss unchanged.
CV-probe check, critic (A2c) results and paired CIs: in the output CSVs.
Part 2:
Paired clustered bootstrap (routing AUROC, evaluation questions; seed 11 | seed 29), all CIs exclude 0 unless marked n.s.:
  A1 - E1 +0.139 | +0.130 ; A1 - interp0.5 +0.012 [+0.001, +0.023] | +0.012 [+0.001, +0.023] ; A1 - taskvec +0.020 | +0.021
  A2 - E1 +0.143 | +0.140 ; A2 - interp0.5 +0.017 [+0.007, +0.025] | +0.021 [+0.012, +0.032] ; A2 - taskvec +0.024 | +0.031
  B  - E1 +0.116 | +0.114 ; B - interp0.5 -0.011 n.s. | -0.004 n.s. ; B - taskvec -0.003 n.s. | +0.005 n.s.
Agreement with E0 decisions: A2 0.95, A1 0.92-0.93, interp 0.93, taskvec 0.90-0.91, B 0.89-0.90, E1 0.77-0.83.
SFT gain kept: A2 99.4 %, A1 99.2 %, B 100 %, taskvec 97 %, interp 74 %.
Critic (IsRel G2 / IsRel val / routing / instr loss): critic 0.872 / 0.867 / 0.812 / 3.081; critic-SFT 0.763-0.750 / 0.763-0.756
  / 0.662-0.674 / 2.386; A2-critic 0.883 / 0.870 (s11), 0.881 / 0.874 (s29) / 0.827, 0.813 / 2.409. A2c - critic-SFT on IsRel
  G2 +0.121 [+0.072, +0.167] | +0.131 [+0.085, +0.181]; val +0.107 | +0.118. Both decision tokens kept at once.
CV-probe check: E0 0.770, E1 0.781 / 0.778 (no loss of linearly decodable information; drop -0.011 / -0.008). Recovered share
  0.58 / 0.54 -> "mixed" by the fixed rule, because the CV probe ceiling (0.77) is below the trained token (0.838). Supporting
  (not preregistered): in-model readout repair B recovers 79 % / 83 % of the AUROC drop.

========================================
29. TEST PRE-REGISTRATION (notebook 09; fixed Nov 12 2026, before any test number was seen)
========================================
Test set: in_main PopQA test questions; frozen theta0 labels y from notebook 06 (lambda 0.1875), unused until now.
Models (seeds 11, 29): E0, E1 (07); A1, A2 (12); replay E2-E7 (08); interpolation alpha 0.5 and task vector beta 1 (rebuilt
from checkpoints; included only as comparators required by the primary family); critic, critic-SFT (10), A2-critic (12).
Readout repair B is NOT scored on test (team choice).
Primary endpoint: routing AUROC vs frozen labels on test.
Primary family: A2 and A1 each vs E1, interpolation 0.5, task vector 1, E2 random, E7 UABR.
Original hypothesis (reported): E7 UABR vs E1, E2 random, E5 BPR. Descriptive: E2-E6 vs E1.
Critic family: A2-critic vs critic-SFT on IsRel AUROC (test windows: non-overlapping top-5, automatic labels).
Test: paired subject-clustered bootstrap, 2,000 resamples, 95 % CI, per seed. Claim rule: BETTER only if the CI excludes 0 in
favour in BOTH seeds; one seed = suggestive; excludes 0 against in both = WORSE; otherwise no difference detected. No
multiplicity correction (the both-seed rule is the guard).
Secondary (no claims): AUPRC, default call rate, agreement with E0, instruction val loss / SFT gain kept, utility at 10 %
budget (theta0 and fresh rewards), fresh closed / evidence EM.
Procedure: validation integrity checks (E0 = 06, E1 = 07, E7 = 08, A2 = 12, interp = 12) must pass before the test cells.
The notebook is run once; a re-run is allowed only after a crash and must be reported.
