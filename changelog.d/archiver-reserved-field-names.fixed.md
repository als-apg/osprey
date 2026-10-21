A channel whose address equals one of the archive's own document fields
(`date`, `_id`, `expireAt`, the densify marker `osprey_densified`, or a seed
manifest field such as `fingerprint`) is now recorded, seeded and read back
under its own address instead of overwriting or matching that field. Its first
character is stored escaped, so a channel named `date` lives under `%64ate`.
