`aiofiles` is no longer a dependency of `osprey-framework`, and the unused
`serialize_results_to_file_async` helper is gone; results are written by
`serialize_results_to_file`, as they already were. Install `aiofiles` yourself
if your own code imports it.
