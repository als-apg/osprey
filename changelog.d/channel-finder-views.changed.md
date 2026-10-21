**Breaking change:** `osprey build` writes each channel-finder index from the
facility description, and its hierarchical index differs from the hand-written
databases it replaces.

- The hierarchical index's levels are the facility's place level words, then
  `class`, `device` and `leaf`. Every level is present for every channel, and
  the node `-` names an absent place, class or device. The `field` and
  `subfield` levels are dropped: their descriptions are kept in the groups'
  `signals` sentences and shown on leaves and middle_layer Fields. A query or
  a saved path that walks the old levels no longer resolves.

The three `channel_finder.pipelines.<pipeline>.database.type` keys are gone:
each pipeline reads the one index the build writes, so there is no loader to
choose.
