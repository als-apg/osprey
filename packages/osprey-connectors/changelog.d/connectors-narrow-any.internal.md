Parameters that validate what they are given take `object` instead of `Any`, so
mypy checks the narrowing in their bodies; callers pass the same values as
before. The `ControlSystemConnector` methods a connector overrides keep `Any`.
