The notebook panel starts only the OSPREY kernel. A request for any other
kernelspec, including the interpreter's own `python3`, is refused, and a
notebook that names no kernel gets the OSPREY one, so every kernel runs with
the write gates and files its records under its own kernel name.
