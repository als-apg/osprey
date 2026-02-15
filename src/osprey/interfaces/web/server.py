"""FastAPI Web Server for Osprey Debug UI.

This module provides a FastAPI application with WebSocket support for
streaming Osprey events to a browser-based debug interface.

The server:
    - Serves a static HTML/JS debug UI
    - Accepts WebSocket connections for real-time event streaming
    - Executes agent queries and streams events to connected clients
    - Supports multiple concurrent client connections
"""

from pathlib import Path

from osprey.events import parse_event, register_fallback_handler
from osprey.utils.config import get_config_value
from osprey.utils.rich_colors import init_terminal_colors

from .event_handler import WebEventHandler

# Lazy imports for optional dependencies
FastAPI = None
WebSocket = None
WebSocketDisconnect = None
StaticFiles = None
HTMLResponse = None


def _ensure_dependencies():
    """Ensure FastAPI and related dependencies are available."""
    global FastAPI, WebSocket, WebSocketDisconnect, StaticFiles, HTMLResponse

    if FastAPI is None:
        try:
            from fastapi import FastAPI as _FastAPI
            from fastapi import WebSocket as _WebSocket
            from fastapi import WebSocketDisconnect as _WebSocketDisconnect
            from fastapi.responses import HTMLResponse as _HTMLResponse
            from fastapi.staticfiles import StaticFiles as _StaticFiles

            FastAPI = _FastAPI
            WebSocket = _WebSocket
            WebSocketDisconnect = _WebSocketDisconnect
            StaticFiles = _StaticFiles
            HTMLResponse = _HTMLResponse
        except ImportError as e:
            raise ImportError(
                "Web UI dependencies not installed. Install with: pip install osprey-framework[web]"
            ) from e


def create_app(
    config_path: str | Path | None = None,
    title: str = "Osprey Debug UI",
) -> "FastAPI":
    """Create and configure the FastAPI application.

    Args:
        config_path: Path to osprey config.yml (optional)
        title: Title for the API documentation

    Returns:
        Configured FastAPI application
    """
    _ensure_dependencies()

    # Query terminal colors at startup (if TTY available)
    init_terminal_colors()

    app = FastAPI(title=title)

    # Store config path for later use
    app.state.config_path = config_path

    # Get static files directory
    static_dir = Path(__file__).parent / "static"

    # Mount static files if directory exists
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main debug UI page."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return index_path.read_text()
        return HTMLResponse(
            content="<h1>Osprey Debug UI</h1><p>Static files not found.</p>",
            status_code=200,
        )

    @app.get("/api/config")
    async def get_config():
        """Return current configuration info."""
        try:
            return {
                "project_name": get_config_value("project.name", "unknown"),
                "agent_name": get_config_value("agent.name", "unknown"),
            }
        except Exception:
            return {"project_name": "unknown", "agent_name": "unknown"}

    @app.get("/api/colors")
    async def get_colors():
        """Return logging_colors from config with hex palette for component styling.

        Returns both the component-to-color-name mapping from config and
        the color-name-to-hex mapping from Rich's palette for accurate
        terminal color matching.
        """
        try:
            from osprey.utils.rich_colors import get_rich_color_hex

            # Get component color names from config
            component_colors = get_config_value("logging.logging_colors", {})

            # Build palette with hex values for all colors used
            palette = {}
            for color_name in set(component_colors.values()):
                if color_name:
                    hex_val = get_rich_color_hex(color_name)
                    if hex_val:
                        palette[color_name] = hex_val

            # Add status level colors for special event styling
            for status_color in ["green", "yellow", "red"]:
                if status_color not in palette:
                    hex_val = get_rich_color_hex(status_color)
                    if hex_val:
                        palette[status_color] = hex_val

            return {"colors": component_colors, "palette": palette}
        except Exception:
            return {"colors": {}, "palette": {}}

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket):
        """WebSocket endpoint for streaming events.

        Accepts WebSocket connections and streams events in real-time.
        Supports commands:
            - {"action": "execute", "query": "..."}: Execute a query
            - {"action": "ping"}: Heartbeat
        """
        await websocket.accept()
        handler = WebEventHandler(websocket)

        # Register as fallback handler for events outside graph execution
        unregister = register_fallback_handler(handler.create_fallback_handler())

        try:
            # Send initial connection confirmation
            await websocket.send_json(
                {
                    "type": "connected",
                    "message": "Connected to Osprey Debug UI",
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                }
            )

            while True:
                # Wait for commands from client
                data = await websocket.receive_json()
                action = data.get("action", "")

                if action == "ping":
                    await websocket.send_json({"type": "pong"})

                elif action == "execute":
                    query = data.get("query", "")
                    if query:
                        await _execute_query(handler, query, app.state.config_path)

                elif action == "disconnect":
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            unregister()

    return app


async def _execute_query(
    handler: WebEventHandler,
    query: str,
    config_path: str | Path | None = None,
) -> None:
    """Execute a query against the Osprey graph and stream events.

    Uses the same execution pattern as CLI for consistency:
    1. Initialize registry with config path
    2. Create graph with registry and checkpointer
    3. Use Gateway for proper state preparation
    4. Stream events using properly prepared state

    Args:
        handler: WebEventHandler to send events through
        query: User query to execute
        config_path: Optional config path for graph initialization
    """
    try:
        import uuid
        from datetime import datetime

        from langgraph.checkpoint.memory import MemorySaver

        from osprey.graph import create_graph
        from osprey.infrastructure.gateway import Gateway
        from osprey.registry import get_registry, initialize_registry
        from osprey.utils.config import get_config_value, get_full_configuration
        from osprey.utils.log_filter import quiet_logger

        # 1. Initialize registry with config path (like CLI)
        with quiet_logger(["registry", "CONFIG"]):
            initialize_registry(config_path=config_path)
            registry = get_registry()

        # 2. Create graph with registry and checkpointer (like CLI)
        checkpointer = MemorySaver()
        graph = create_graph(registry, checkpointer=checkpointer)

        # 3. Create Gateway (like CLI)
        gateway = Gateway()

        # 4. Set up config (like CLI)
        thread_id = f"web_session_{uuid.uuid4().hex[:8]}"
        configurable = get_full_configuration(config_path=config_path).copy()
        configurable.update(
            {
                "user_id": "web_user",
                "thread_id": thread_id,
                "chat_id": "web_chat",
                "session_id": thread_id,
                "interface_context": "web",
            }
        )
        recursion_limit = get_config_value("execution_control.limits.graph_recursion_limit", 100)
        base_config = {
            "configurable": configurable,
            "recursion_limit": recursion_limit,
        }

        # Send execution started event
        await handler.websocket.send_json(
            {
                "type": "execution_started",
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 5. Process message through Gateway (like CLI)
        result = await gateway.process_message(query, graph, base_config)

        if result.agent_state:
            from langchain_core.messages import AIMessageChunk

            # Track accumulated response and token count for streaming
            accumulated_response = ""
            token_count = 0
            current_generation_attempt = 1  # Track retry attempts from state updates

            # 6. Stream events using properly prepared state (like CLI)
            # Use multi-mode streaming to get both typed events and LLM tokens
            # subgraphs=True enables streaming from nested service graphs (e.g., Python executor)
            # "updates" mode enables tracking state changes like generation_attempt for retry distinction
            async for _ns, mode, chunk in graph.astream(
                result.agent_state,
                config=base_config,
                stream_mode=["custom", "messages", "updates"],
                subgraphs=True,
            ):
                if mode == "updates":
                    # Track state changes for retry attempt distinction
                    # generation_attempt is incremented by generator node on each retry
                    if isinstance(chunk, dict) and "generation_attempt" in chunk:
                        current_generation_attempt = chunk["generation_attempt"]
                    # Don't send updates to client - just track internally
                    continue

                elif mode == "custom":
                    # Handle typed events
                    event = parse_event(chunk)
                    if event:
                        await handler.handle(event)

                elif mode == "messages":
                    # Handle LLM token streaming
                    message_chunk, _metadata = chunk
                    # Only process AIMessageChunks (streaming tokens), skip full AIMessages
                    if not isinstance(message_chunk, AIMessageChunk):
                        continue
                    if hasattr(message_chunk, "content") and message_chunk.content:
                        token_count += 1
                        accumulated_response += message_chunk.content
                        # Send per-token data with metadata for grouped display
                        # Include generation_attempt for retry distinction
                        await handler.websocket.send_json(
                            {
                                "type": "streaming_token",
                                "token": message_chunk.content,
                                "accumulated_response": accumulated_response,
                                "token_index": token_count,
                                "node_name": _metadata.get("langgraph_node", "respond"),
                                "session_id": thread_id,
                                "timestamp": datetime.now().isoformat(),
                                "generation_attempt": current_generation_attempt,
                            }
                        )

        # Send execution completed event
        await handler.websocket.send_json(
            {
                "type": "execution_completed",
                "event_count": handler.event_count,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        from datetime import datetime

        await handler.websocket.send_json(
            {
                "type": "execution_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        )


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    config_path: str | Path | None = None,
    reload: bool = False,
) -> None:
    """Run the Osprey Debug UI server.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 8080)
        config_path: Path to osprey config.yml
        reload: Enable auto-reload for development
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn not installed. Install with: pip install osprey-framework[web]"
        ) from e

    app = create_app(config_path=config_path)

    print(f"Starting Osprey Debug UI at http://{host}:{port}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
