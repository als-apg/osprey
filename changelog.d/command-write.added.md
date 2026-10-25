`osprey.runtime.guarded_run.command_write` fires one value at a batch of
command setpoints (arm a monitor, request a shot) under the run lock. It
refuses the whole batch unless the lock's record, the account, the control
target and an enforcing limits validator all match.
