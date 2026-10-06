The web terminal's scaffold Preview now sanitises the markdown it renders.
Agent-writable `.claude/` files (rules, agents, hook docstrings) could carry
markup such as `<img onerror>` that ran in the terminal page when an operator
opened the file; the Preview now shares the chat log's DOMPurify path and shows
plain text when the sanitiser is unavailable.
