The history page and the session recipe now read back every agent event of a
session, including several events of one kind emitted in the same millisecond.
A log row is now identified by its timestamp, its body and its attributes, so
such events are no longer collapsed into one or skipped between pages, while an
exact re-insert of the same record is still read once.
