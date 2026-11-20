A `pyat` readback record that states another calibration than the setpoint on
its element, attribute and index now reads that setpoint's physics value
through its own calibration, so one quantity can be served in a second unit;
before, it served the setpoint's value. Its computed `default` is the value it
starts with, as is that of a readback no setpoint names as its pair or whose
pair the simulation does not follow. A
readback of a deck property that states another calibration than its setpoint
stops the build with `engine-invalid`.
