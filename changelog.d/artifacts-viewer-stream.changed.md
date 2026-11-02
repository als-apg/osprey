The artifacts viewer in the web terminal no longer opens its own event stream.
The hub's sidebar already follows that stream, and now tells the viewer when
the artifact it shows is updated or deleted. This frees one of the browser's
six connections to the hub, so tiles do not wait for a free connection.
