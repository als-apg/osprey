`FacilityRecords.of(document)` gives one read model to every caller that asks
for the records of the same document, and `served_scope` scans the channels
once for each set of served models, so a build that renders one large facility
many times indexes and selects it once.
