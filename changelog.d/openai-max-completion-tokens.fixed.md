The OpenAI provider sends its output-token limit as `max_completion_tokens`,
the parameter OpenAI's API takes on every chat model, and sends no
temperature, which OpenAI's reasoning models refuse. Completions and health
checks on a GPT model that LiteLLM does not yet recognise, such as
`gpt-6-sol`, no longer fail with HTTP 400 `Use max_completion_tokens instead`,
and completions on `gpt-5.6-*` no longer fail on temperature 0. Claude Code
sessions on the `openai` provider send the same request shape through the
translation proxy.
