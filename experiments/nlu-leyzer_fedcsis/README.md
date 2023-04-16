Experiment description
======================
In this experiment [leyzer-fedcsis](https://huggingface.co/datasets/cartesinus/leyzer-fedcsis) was translated with
[iva_mt_wslot-m2m100_418M-en-pl](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-pl) model. Two variants were created:
1. m2m100_418M-leyzer_fedcsis-translated-plain_text.tsv - src_utt column with plain text en-US was translated to tgt_utt. model occasionally generated tags (e.g., <a>), but there are no slots in the input.
2. m2m100_418M-leyzer_fedcsis-translated-w_slot.tsv - src_

Translation error analysis
==========================
To analyze translation errors, 10TC per intent (`cut -f3` from the above files) was selected, and translation correctness was done manually.
1. m2m100_418M-leyzer_fedcsis-translated-plain_text-error_analysis-10tc_per_intent.tsv - only text translation was evaluated
2. m2m100_418M-leyzer_fedcsis-translated-w_slot-error_analysis-10tc_per_intent.tsv - both text and slot translation were evaluated

Initial observations:
1. In cases when utterances consist of more than one slot, then during translation, slot names (<a>, <b>, etc) are very often incorrectly assigned
2. Even if slot values are not translated correctly, they are almost always in the right part of a sentence

Acceptance criteria for text (evaluation as made on plain text, slots were removed for evaluation):
1. PASS (1) if perfect or near-perfect translation. The near-perfect translation is when minor omissions are made or minimal errors while all essential parts of utterance are perfectly translated. In some cases, verb translation was not exactly "the one", but as long as they are correct translations and this is not changing intent meaning it should be accepted.
2. Name entities (slots) translations are very hard to define, but here are some rules used by the expert:
a) Music, location, and other similar entities cannot be translated
b) Message content (slack, email), their subject, channel names, calendar event names, and similar should be translated. However, they can be accepted without value translation because this model aims to translate NLU training examples. For that purpose, non-translated values will be ok (not ideal, but much better than a wrong translation).
c) Option names (e.g., phone type, email priority) should be translated BUT same as in b) no translation should be accepted
d) Dates should be translated
e) Numerals (ints) should not be translated, but if the number is written as a word (first, second, etc.), then the translation is expected
f) Filenames should not be translated
g) Contact names should not be translated
h) Tag names should not be translated
i) Cousine and restaurant types should be translated

Acceptance criteria for slot:
1. PASS (1) if the same part of the sentence are annotated AND if either slot value is not translated (slot value was copied from input) OR slot value was translated correctly. Also, most rules from text acceptance criteria (2) apply here.
3. PASS (1) if there is no slot in input and no slot in the output.
