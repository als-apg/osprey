The PV directory service serialises the connectors' service records as pydantic models, and
rebuilds a changed query, row, record or backend spec through `model_validate`.
