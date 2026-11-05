A deployed optimizer binds the deployment's write ledger, `var/write_ledger`,
read-write as `OSPREY_WRITE_LEDGER_DIR` whenever it binds the control-context
tree, and warns at start when it cannot write there.
