"""Entry point for ``python -m osprey.mcp_server.channel_finder_in_context``."""

from osprey.mcp_env import load_dotenv_from_project
from osprey.mcp_server.startup import redirect_logging_to_stderr


def main() -> None:
    load_dotenv_from_project()
    redirect_logging_to_stderr()

    from osprey.mcp_server.channel_finder_in_context.server import create_server

    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
