"""Entry point for ``python -m osprey.services.channel_finder.mcp.middle_layer``."""

from osprey.mcp_env import load_dotenv_from_project
from osprey.mcp_server.common import redirect_logging_to_stderr


def main() -> None:
    load_dotenv_from_project()
    redirect_logging_to_stderr()

    from osprey.services.channel_finder.mcp.middle_layer.server import create_server

    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
