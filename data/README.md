# Data policy

This folder must not contain harmful, abusive, or red-team prompt text.

- `fixtures/` holds only short benign examples used by the CPU smoke test.
- Raw and derived safety evaluation sets are written at runtime to
  `data/raw/` and `data/derived/`, both of which are gitignored.
- The evaluation set is built from public, license-checked benchmarks,
  starting with [McGill-NLP/tukabench](https://huggingface.co/datasets/McGill-NLP/tukabench)
  (CC-BY-NC 4.0). Cite TukaBench and JailbreakBench.
- Do not generate new harmful prompts. Code-mixed eval items, if used, must
  be deterministic mixes of already-published TukaBench pairs that share an
  `Index`, not newly authored attacks.
- Nigerian Pidgin has no public safety benchmark we can use today. Leave it
  out of the harmful-compliance set until a licensed public source exists.
