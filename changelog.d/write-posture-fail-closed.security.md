Connector writes fail closed on unknowns, and record writes are serialized
across processes. The residual: the serialization does not hold across a macOS
VM file share.
