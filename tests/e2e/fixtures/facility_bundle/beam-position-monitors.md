# Beam position monitors

Beam position monitors report the transverse centroid of the stored beam at
fixed locations around the ring. Each monitor houses four button pickups whose
induced charge is digitised and combined into a horizontal and a vertical
reading. The difference-over-sum calculation cancels the dependence on stored
current, so a reading stays meaningful as the beam decays between fills.

Readings are published on the control system once per turn and averaged into a
ten-hertz stream for the orbit feedback. A monitor whose electronics have lost
their calibration constants reports a plausible but wrong offset, which the
orbit correction then treats as real and steers the beam towards.
