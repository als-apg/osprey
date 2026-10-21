An imported MML model wires its tune readback to the engine's tunes: the
mapping's `tune` block names one address per plane, each read as a scalar, or
one address read as a waveform of every plane, typed `value_type: waveform`.
The pyat engine serves a waveform readback wired to `tune` or `chromaticity`
as the whole output, read-only; a non-float setpoint, or a non-float readback
wired to no array output, still stops the build. The seeded measurement file
reads the tune on that readback and allows the tune response.

The import wires the energy knob to the deck energy through the export's
energy table, its readback following the setpoint read-only; wires every
member's readback of a shared supply; wires a family with no stated hardware
nominal from the deck where neither a turning conversion nor a series needs
the nominal; carries a device's polarity on its slice where the deck holds the
other sign; and seeds `scenarios/readout.yaml` once from the export's monitor
readout, naming each readout it cannot carry. A monitor's offset that the
export's conversion holds is served through the wiring, so the Middle Layer's
correction, Gain x (Raw - Offset), gives back the model's position; an offset
the conversion is not seen to hold, and a gain other than 1, is not carried.
The draft mapping proposes the energy knob, a correcting dipole string and the
cavity, and a signal role for each magnet current and monitor position field;
the import writes that role on each channel and keys its group's sentence by
it. A readback several setpoints share pairs none of them.

A view stop on a record only `fixes.yaml` produced names `fixes.yaml`.
