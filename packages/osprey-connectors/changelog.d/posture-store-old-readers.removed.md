**Breaking change:** `osprey_connectors.posture_store` no longer exports the
old narrowing and store readers: `effective_writes`, `store_permits`,
`store_verdict`, `store_verdict_detail`, `record_verdict_detail`,
`StoreVerdict`, `StoreVerdictDetail`, `recorded_posture`, `target_posture`,
`launch_permits`, `launch_narrowed_target` and `parse_launch_posture`. Ask
`write_verdict(section, target)` instead — its `allowed` and `reason` replace
the bool and the verdict member, and a run whose launch pin refuses answers
`reason == "launch_pinned"`.
