`osprey mml emit` stops the virtual-accelerator lane when a coupled setpoint's
family states no finite `Setpoint` `Range` on both edges for it, and names the
family, the device and the row. It used to write that setpoint writable with an
edge missing, which the served model then left unbanded. State a band in the
export, or latch the family in the mapping.
