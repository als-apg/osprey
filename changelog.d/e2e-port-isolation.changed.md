E2E fixtures that start real services no longer squat a live deployment's
ports: the virtual-accelerator harnesses bind ephemeral host ports instead of
5064, and deploy-shaped e2e tests pin their own thousand-port block instead of
the default 10000.
