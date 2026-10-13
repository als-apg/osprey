from osprey.services.optimizer.utils.job_keys import (
    all_meta_scan_pattern,
    config_key,
    job_scan_pattern,
    state_prefix,
)


def test_keys_are_inside_job_namespace() -> None:
    assert config_key("abc") == "job:abc:config"
    # Must be swept by the per-job scan (terminal cleanup / heartbeat).
    assert job_scan_pattern("abc") == "job:abc:*"
    assert all_meta_scan_pattern() == "job:*:meta"
    # Optimization and script jobs share the same ``job:{id}:state`` prefix — a
    # job is exactly one type and its id is unique, so they never collide.
    assert state_prefix("job-3") == "job:job-3:state"
