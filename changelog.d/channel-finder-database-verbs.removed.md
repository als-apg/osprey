`osprey channel-finder build-database` and `osprey channel-finder generate`, with
no replacement: the build writes the channel-finder indexes from the facility
description. The three `channel_finder.channel_name_generation.llm_model.*` keys
(`provider`, `model_id`, `max_tokens`) went with the CSV builder that read them.
