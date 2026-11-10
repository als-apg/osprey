The frontmatter split of knowledge pages moves to `osprey.facility.frontmatter`; the facility
build reads its pages without importing Knowledge, and `OKFDocument.parse` delegates to it.
