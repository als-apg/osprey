`dangerously_allow_bash: true` at the root of the deploy config waives the
refusal to deploy a launch-token persona whose settings allow `Bash` — for a
development box with one trusted operator. Every `osprey up` under it prints a
warning naming the personas it waved through; absent, nothing changes.
