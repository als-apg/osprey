`osprey init` now seeds the gateway endpoint variable beside the API key when
the shell exports it. A provider that ships no default endpoint no longer needs
a hand edit of `.env` before the first `osprey up`.
