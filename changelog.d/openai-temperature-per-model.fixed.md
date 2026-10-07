The `openai` provider sends a caller's temperature to OpenAI's chat models
(the `gpt-4o`, `gpt-4.1`, `gpt-4.5`, `gpt-4` and `gpt-3.5` families) and leaves
it out only for the reasoning models, on direct calls and through the
translation proxy alike. A deployment naming `gpt-4o` no longer samples at the
model default whatever temperature was asked for.
