
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
