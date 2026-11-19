A readback's seed `follows: <setpoint>` may follow a setpoint of another type
into a labelled readback, which then takes the state of the same index: from
a labelled setpoint of as many labels (`Clear / Set` into `Off / On`), or from
a setpoint that holds a number (`1` holds the second label). The build refuses
a setpoint of another label count or of neither labels nor a number, and a
number setpoint whose start value is no index of the readback's labels. The
default never copies by index. `follows.json` states each followed pair's
`copy` (`value` or `index`) and, for an index copy, the label of each index.
