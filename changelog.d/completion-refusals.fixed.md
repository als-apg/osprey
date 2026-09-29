`get_chat_completion` refuses a call with no model id for every provider
instead of handing the provider an empty one, and a `model_config` that names
no provider is refused as such rather than as `Unknown provider: None`.
