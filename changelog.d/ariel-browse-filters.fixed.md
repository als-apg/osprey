The ARIEL `browse` tool filters by author and source system in the database
query instead of over the page it just fetched, so a filter on a logbook larger
than one page no longer silently drops matching entries or reports a total that
counts the whole table.
