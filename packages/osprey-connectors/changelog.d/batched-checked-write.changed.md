`write_multiple_channels()` checks every write of a batch before it sends any,
on every connector: if the checks refuse one, none is sent, and the held-back
writes are refused with the new reason `BATCH_REFUSED`. A refusal the control
system gives only at the send can follow writes already sent; each result says
what happened to its own channel. A limits refusal in a batch is a `refused`
result, not a raised `ChannelLimitsViolationError`. A batch that names a
channel more than once is refused whole. The EPICS connector reads the stepped
channels once and sends the puts together. `raise_for_write_results()` raises
for the refused write of a batch, not for one the batch held back.
