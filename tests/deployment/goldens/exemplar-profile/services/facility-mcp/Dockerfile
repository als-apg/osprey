# Demo Facility's own MCP server image.
#
# A facility-owned service directory that carries a Dockerfile gets its own
# image-build job in the scaffolded pipeline. Nothing here depends on OSPREY:
# the container speaks MCP over streamable HTTP and the agent reaches it by URL,
# so the framework never needs to be installed alongside it.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./

# Unbuffered so container logs appear as the server writes them rather than
# when the process exits.
ENV PYTHONUNBUFFERED=1
EXPOSE 8200

CMD ["python", "server.py"]
