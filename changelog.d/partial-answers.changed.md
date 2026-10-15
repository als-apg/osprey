`channel_read` and `entries_by_ids` name what they did not answer under `missing`, a map from
each requested item to the source's own text (`{}` when everything answered); this replaces
`channels_failed` and the `requested`/`found` counts. A request nothing answered is refused with
the same map under `details.missing`: as the one error every item failed with, or as the new
`nothing_answered` error, which the agent's error-handling rule names with the class
"Unanswered". The contract is documented on the new Partial Answers reference page.
