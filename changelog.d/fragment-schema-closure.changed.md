An application `image:` fragment may no longer carry top-level `configs` or
`secrets` (bind the file on the service instead), and a long-form volume's
`type` must be `bind`, `volume` or `tmpfs`, spelled literally.
