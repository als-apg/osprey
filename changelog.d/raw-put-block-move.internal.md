The readonly executor guard now lives in the stdlib-only `osprey.runtime.raw_put_block` module; the executor embeds its source and runs it in a private namespace, so behaviour is unchanged.
