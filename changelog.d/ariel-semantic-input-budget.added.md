How much of a logbook entry ARIEL's semantic processor sends for keywords and a
summary is now a config key,
`ariel.enhancement_modules.semantic_processor.max_input_chars` (default 8000,
the previous fixed slice). An entry longer than the budget is still cut, but
the cut is now logged naming the entry, so a summary that covers only an
opening says so.
