A site build-arg value carrying a quote character no longer renders a compose
fragment that fails to parse. The value reaches the build exactly as it was
declared, so an index or proxy URL with credentials in it is usable.
