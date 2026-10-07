After a control-target switch, a notebook cell now reads from and writes to the
new target. Before, the kernel's next cell rebuilt its connector in the same
process, and Channel Access kept the server it had reached first: the connector
reported the new target while its reads (and any write) went to the old one.
On a deployment that can switch, the notebook kernel now takes its connector
from a connector-host child of its own for each target, so the kernel no longer
has to be restarted after a switch. (#1518)
