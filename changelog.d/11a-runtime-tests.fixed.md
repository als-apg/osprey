The isolated picture renderer's per-loop guard task is dropped without an
asyncio `Task was destroyed but it is pending!` log record when its event loop
is closed, or stops for good, before the task can finish.
