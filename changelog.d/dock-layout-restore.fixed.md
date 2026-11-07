A web-terminal layout stored for a user comes back whole on the next login:
the tile in front is filled by its own panel, and the profile's default panel
takes the workspace only when the stored layout holds no service tile. Before,
the default claimed the slot as soon as it was healthy, adding itself to the
restored layout.
