A refused control-system write names every reason it was refused, the deployment's
`writes_enabled` setting first. A script launched with writes off whose target has since been
turned off from the control-target chip, or whose recorded write state cannot be read, hears both
reasons rather than only the launch.
