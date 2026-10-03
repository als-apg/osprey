`osprey facility validate` holds each response export an mml import kept
(`imported/mml/<model>.response.json`) against the model the facility file
describes and prints one `response check <model>:` line per model to stderr,
naming the judged block nearest its bar. A block the export computed from a
model needs 0.99 of its entries inside the 5 % band; a measured block needs a
median size ratio in 0.8 to 1.25 and sign agreement of 0.95. A failing check
exits 1; nothing is written.
