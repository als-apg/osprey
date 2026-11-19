Members that only the tests called are gone. No framework release uses any of them:

- `osprey_connectors.connection.ConnectionSettings.credential_env_names`: read
  `login.env_names`, or `read_credential_env_names()` for a whole block.
- `osprey_connectors.control_system.ChannelWriteResult.observed_number`:
  `observed_value` holds the reading in the type the channel reports.
- `osprey_connectors.control_system.LimitsValidator.writable_addresses()`.
- `osprey_connectors.config.ConfigBuilder._require_config()` and
  `osprey_connectors.errors.ChannelWriteFailedError._VALID_REASONS`, both
  private. A write failure's reason is still its outcome word, uppercased.
