Headless dispatch runs go through the agent runner and read its event records,
so the dispatch worker imports nothing from the agent SDK.
run_query closes its message stream when its loop body raises.
