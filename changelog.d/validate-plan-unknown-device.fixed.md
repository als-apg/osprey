Validating a session plan now refuses sample arguments that name a device the
queue worker does not have, with the same `unknown_device` answer an enqueue
gives, instead of passing a plan the queue later refuses.
