How-to pages for ARIEL, facility knowledge, deployment and health checks no
longer give wrong instructions. The ARIEL quick start now runs `osprey build`
before `osprey up`, and its command table lists every `osprey ariel` verb and
the `hybrid` search mode. The picture-search page names `osprey ariel sync
--watch`, which the `ariel-sync` service runs. Search-module and adapter keys
go under `config:` in `profile.yml`, not in `config.yml`. The facility-knowledge
page says `osprey knowledge` has three verbs and shows its real help. The
compose-template excerpt matches the shipped PostgreSQL template. The
project-image page says `osprey build` re-renders the Dockerfile every build.
New docs tests check the ARIEL command table, the knowledge help screen and
the template excerpt against the code.
