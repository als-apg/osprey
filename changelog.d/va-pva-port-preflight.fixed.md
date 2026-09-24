A host-port collision on the virtual accelerator's pvAccess port now names
`services.virtual_accelerator.pva_port`, the key that moves it. It used to name
the Channel Access port's key, which left the collision in place.
