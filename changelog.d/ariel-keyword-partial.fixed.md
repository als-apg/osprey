ARIEL keyword search no longer comes back empty when no entry contains every
word of a plain-word query. It then returns the entries holding at least half
of the words, ranked by how many they hold, each naming its `matched_terms`
and `missing_terms`, with an INFO diagnostic listing the words no hit
contains. A vocabulary alternative counts as the one word it stands for, and
hybrid search decides that a picture caption matched by the same word count.
Operator queries, quoted phrases and patterns are matched as before.
