ARIEL can expand logbook entries through the facility vocabulary before they
are embedded, so the entry side carries the same canonical phrases the query
side already adds. Set `ariel.vocabulary.expand_entries: true` to turn it on;
the default is off and embeds the same bytes as before. The qmd mirror renders
the same expanded text. `osprey ariel status` counts the stored vectors that
were built under a different switch, expansion rule or vocabulary than the
running configuration, and prints the command that rebuilds them.
