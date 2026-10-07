A lifecycle step's Integration Test Results table now prints only when that
step wrote `check_results.xml`. A later step that writes no results no longer
repeats an earlier step's table, and the file stays in place for later steps.
