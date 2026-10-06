A logbook entry longer than its embedding model's input window is cut so its
start is embedded, instead of failing on every enhancement pass. Each model
under `ariel.enhancement_modules.text_embedding.models` states its window as
`max_input_tokens`; the presets set 2048 for `nomic-embed-text`, and a model
that states none is cut to 512 tokens. `osprey ariel reembed` applies the same
cut.
