A connection block's `auth.token_env` or `auth.password_env` that ends in a newline is refused
as not naming an environment variable. It was accepted with the newline kept, so the connector
looked the secret up under a name no environment holds.
