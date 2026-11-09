Provider adapters no longer declare `requires_model_id`, and a provider class
is no longer refused for leaving it out. Nothing read the flag: every chat
completion is refused without a model id, whatever the provider. A custom
provider that still sets it only carries an unused class attribute.
