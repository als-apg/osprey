A dock op's `panel_op` frame goes only to the dock client's event stream, so
another page on the same session cannot answer the op in its place. A stalled
event stream is dropped after 1024 queued frames instead of growing without
bound.
