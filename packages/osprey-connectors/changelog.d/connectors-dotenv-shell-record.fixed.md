The record of which shell exports the env chain overrode now compares each
export with the value actually applied. Before, a `.env` line with an inline
`# comment` or a `${VAR}` reference could be reported as overriding an export
it matched, and an export it did replace could go unreported.
