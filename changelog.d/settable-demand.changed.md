A bluesky plan device with a distinct readback now reports its demand as
well as its readback: `ConnectorSettable.read()` and `describe()` carry
`<name>_setpoint` (the setpoint channel, read live through the connector)
beside `<name>` (the readback), the same convention as ophyd's positioners.
Only the readback stays hinted, so `bps.rd` and live tables keep answering
where the device is. A plan that settle-checks a slow device (an
insertion-device gap, a ramping magnet) reads both facts off the one device
it drives instead of needing a second device that aliases the setpoint
channel under another name. A device whose readback aliases its setpoint
reports one key, as before.
