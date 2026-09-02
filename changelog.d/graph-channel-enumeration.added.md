The web channel-finder API's graph mode can now validate and enumerate
channels instead of refusing outright: `validate_channels` checks membership
against the facility's channel roster, and `get_channels` serves the
roster's addresses and total (`chunk_idx`, meaningful only for the
in-context paradigm, is rejected). When the graph corpus cannot be read,
both still answer with the same unavailable response, naming the config key
to check.
