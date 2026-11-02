`osprey build` no longer removes `build/nginx/` and `build/docker-compose.web.yml`
from under a running web-terminal stack, so the landing page keeps being served
while a rebuild runs.
