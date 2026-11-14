Malformed input now fails with the error the code documents instead of a
stray `AttributeError`, `KeyError` or `TypeError`: a `control_system` or
`archiver` section, `connector` block or per-type block that is not a mapping
(an explicit `null` included) raises `ValueError` from the factory; a
`file_paths` that is not a mapping raises `TypeError` from `get_agent_dir`; an
Archiver Appliance reply of the wrong shape, a `retrieval_path` that is not a
string, and a mock archiver `sample_rate_hz` that is not a number raise
`ValueError`; and `model_status` on a connector whose simulator has no status
raises its `ValueError`. A connector-host frame whose `dict` pairs or `blobs`
list have the wrong shape is refused as a `FrameDecodeError`.
