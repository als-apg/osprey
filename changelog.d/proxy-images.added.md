The translation proxy for OpenAI-protocol providers sends images to routes that
take them, including images a tool returns. Content a route cannot take is
replaced by a note the model sees, and each kind is logged once per
conversation. Local model servers take no images unless their `providers.yml`
entry says `supports_images: true`.
