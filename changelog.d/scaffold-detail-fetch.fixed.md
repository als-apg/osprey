The scaffold panel now loads a file once when it opens the editor for it,
instead of twice. Taking ownership of a framework file, and creating a new
artifact, both opened the file in Preview and switched to Edit on the next
statement, so two requests for the same file raced for the same pane.
