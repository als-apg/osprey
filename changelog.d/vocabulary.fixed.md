Operator- and agent-facing copy no longer assumes EPICS, one facility's device
vocabulary, or a particular logbook product. The visualization tools now
describe the packages their sandbox can actually import, the network guard and
the archiver's empty-window notice name the control system rather than a
protocol, the ARIEL entry form takes free-text shift and logbook names, and
`generic_json` ingestion keeps unrecognised top-level fields under `metadata`
instead of dropping them. The
`ariel.enhancement_modules.semantic_processor.prompt_template` key is now
documented as where a facility puts its own extraction vocabulary.
