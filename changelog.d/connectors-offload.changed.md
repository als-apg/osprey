The DOOCS and Tango connectors now run limits validation and the write itself in
one background thread. A `max_step` check reads the channel's present value from
the control system first, and that read no longer stalls everything else the
assistant is doing while it is in flight. Every outcome word is unchanged: a
write refused by validation still sends nothing, and a value the device did not
take is still reported as a failure.
