`posture_store.owner_context(owner, target, baseline=...)` reads one owner's
control-context record out of the bound tree and its verdict for a target in a
single no-follow read. With no tree bound, or an owner that is not a name, it
answers `control_context_unavailable`. Existing posture reads are unchanged.
