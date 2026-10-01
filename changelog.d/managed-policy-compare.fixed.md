A Claude Code managed policy that sets a provider variable to the value the
deployment already launches with no longer stops `osprey chat`, the Web
Terminal or the dispatch worker from starting. A policy value that differs,
or that sets a variable the deployment leaves unset, still refuses the launch.
The refusal names the policy's value and the deployment's for each variable,
never prints a credential, and says which side to change.
