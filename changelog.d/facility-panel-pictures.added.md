`osprey facility panels` draws a device class seen from its end on its card.
A `pictures.csv` row may now state a size (`w_mm`, `h_mm`), an angle
(`angle_deg`), the signal it shows live (`signal`) and the place it applies
under (`under`); shape kinds (`yoke`, `outline`, `pole`, `coil`,
`corrector_winding`, `bore`, `gap`) draw the body, at default proportions where
no size is stated, and other kinds are parts placed on a coil or in a column
beside the yoke. A scale caption names what the drawing is scaled from.
`check` sorts the classes by how they are drawn and notes a picture signal that
binds no channel or channels of two devices. An optional `state_labels.csv`
names the word each value of a discrete channel means, and a command's button
reads it.
