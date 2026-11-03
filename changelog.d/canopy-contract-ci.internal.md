The CANOPY contract CI job skips pull requests from forks, which receive no
secrets, and says so in the run summary; a maintainer runs the contract on
such a branch through a workflow dispatch. Same-repository pull requests still
fail closed without the CANOPY read credential. The contract's verdict line
names the platform the tests ran on.
