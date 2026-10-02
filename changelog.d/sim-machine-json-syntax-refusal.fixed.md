A simulation `machine.json` that is not valid JSON is now refused as
`Machine file <path> is not valid JSON: …` with the decoder's line and column,
instead of a bare `JSONDecodeError` that named no file.
