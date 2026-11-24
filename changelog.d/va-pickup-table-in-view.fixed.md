A Virtual Accelerator model whose pickup statement names a table now starts
from a rendered deployment. The facility build accepted the table, but the
render copied the deck alone into the simulator view, so the served model
failed with "table cannot be read". The view now carries each table beside
its copy of the deck, as `decks/<model>.<file name>`, and its settings name
that copy. An engine plug-in whose settings name files beside the deck states
`served_settings(settings, carry)` to have them carried.
