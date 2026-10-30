A connector write result that is refused must name a known refusal reason; one
that does not is refused at construction. A result also cannot carry a reason
unless it is refused, or a message unless it is refused, failed or unconfirmed.
The reasons are `osprey_connectors.errors.RefusalReason`, each naming who
refuses and what to do next.
