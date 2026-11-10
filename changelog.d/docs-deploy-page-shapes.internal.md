A docs test compares the deploy and CLI pages' quoted names, shapes and messages with the code
that writes them: the qmd model file names, the proxy bridge in every shipped Dockerfile, the
docker and podman compose invocations `compose_base_cmd` builds, the `[network]` backend the
podman preflight requires, the `.env` copies and the `.env.users` migration, the managed-policy
JSON `read_managed_policy_env` reads, the derived-key refusal, the companion block `osprey chat`
prints, and the image-name order `resolve_image_defaults` composes.
