The agent record's service reads a test-only stopped-threshold override
(`OSPREY_AGENT_RECORD_STOPPED_AFTER_S`, unset in every deployment), and its
end-to-end cases watch several sessions for duplicates through one shared
window, so the lane no longer waits out five minutes and a run of 10 s windows.
