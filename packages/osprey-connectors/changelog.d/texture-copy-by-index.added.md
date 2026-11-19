The texture model copies a setpoint into a labelled readback by index. A
labelled setpoint gives the index of its label, so a pair whose labels differ
(`Clear / Set` into `Off / On`) follows state for state; a number written to a
setpoint must be an integral index of the readback's labels. A write the
readback refuses (a fraction, an index past its labels) holds nothing on
either channel, and the readback starts at the state of its setpoint's start
value. A pair of equal labels or of one unlabelled type copies as before.
