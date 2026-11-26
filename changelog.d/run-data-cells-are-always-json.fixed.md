A run whose data holds a cell of bytes that are not text now loads: the cell
reads `refused: not text`. Every cell of a run's data is a number, text, an
array or a refusal, from the live buffer and from storage alike: a date reads as
ISO 8601 text in UTC (`2026-11-24T10:00:00.5Z`, or `2026-11-24` for a date
without a time), and a value with no such spelling (a complex number, a set)
reads `refused: not a number, text or array`.
A missing value of any kind, such as a date with no time recorded, reads as an
empty cell, like a missing number.
