A measurement file states a listed kind's own sub-block under
`kind_blocks.<kind>`; the build checks it against the pydantic model the kind
declares as its `measurement_schema` before the kind's `build_stops` hook runs.
