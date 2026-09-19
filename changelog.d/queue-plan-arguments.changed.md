An enqueue whose plan arguments are not a mapping of names to values is
refused with `invalid_item` and `HTTP 400`. Nothing is queued: the arguments
are no longer dropped to make room for the name of whoever queued the plan.
