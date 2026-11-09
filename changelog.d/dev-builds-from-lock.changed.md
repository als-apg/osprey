`osprey build --dev` / `osprey up --dev` images, and the project venv built
from an editable checkout, now install every third-party package at the
version the checkout's `uv.lock` holds instead of resolving it fresh from the
index, and no longer prime a released `osprey-framework` first. `--dev` needs
`uv` and a current lock; a missing or stale `uv.lock` stops the build with a
pointer to `uv lock`. Release builds are unchanged.
