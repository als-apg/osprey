A deployment with the archive recorder now builds and records channels whose
address carries a field name (`<record>.RBV`) or any other character the store
treats as syntax. Such channels are stored and read back under their own
address, and archives written before keep reading unchanged.
