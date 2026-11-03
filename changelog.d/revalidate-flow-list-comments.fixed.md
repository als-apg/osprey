`osprey init --revalidate` no longer writes the comments that followed the
retired render catalog inside a one-line flow list before it: a last roster
entry ending in `access: [roster]` keeps it on one line, and the comments stay
above the key they introduce.
