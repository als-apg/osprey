"""
Framework Graph Builder - LangGraph Native Implementation (Async-First)

Creates the main Osprey agent graph using existing components from the registry:
- Real infrastructure nodes (gateway, task_extraction, classifier, orchestrator, monitor)
- Real capabilities loaded from registry
- Router-controlled execution flow via conditional edges
- Convention-based error handling with distributed retry policies
- LangGraph-native TypedDict state management
- Automatic prompt loading infrastructure preservation
- Async PostgreSQL checkpointing enabled by default for production-ready persistence
- Modern async/await patterns for concurrent execution
"""

import os

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from osprey.infrastructure.router_node import router_conditional_edge
from osprey.registry.manager import RegistryManager
from osprey.state import AgentState
from osprey.utils.logger import get_logger

logger = get_logger(name="builder", color="white")


class GraphBuildError(Exception):
    """Raised when graph building fails due to configuration issues."""

    pass


def create_graph(
    registry: RegistryManager,
    checkpointer: BaseCheckpointSaver | None = None,
    enable_debug: bool = False,
    use_postgres: bool = False,
    recursion_limit: int | None = None,
) -> StateGraph:
    """
    Create the main Osprey agent graph using registry components.

    Simple, streamlined approach:
    1. Get all nodes from registry
    2. Add them to LangGraph workflow
    3. Router handles all routing logic
    4. Done!

    Args:
        registry: Registry manager containing nodes and capabilities
        checkpointer: Async checkpointer for state persistence and memory.
                     If None, uses memory saver by default for R&D mode.
                     Enables conversation history, human-in-the-loop,
                     fault tolerance, and debugging capabilities.
        enable_debug: Enable debug logging for graph execution
        use_postgres: If True, attempts to use PostgreSQL checkpointer instead of memory


    Returns:
        Compiled LangGraph StateGraph ready for async execution with checkpointing enabled

    Raises:
        GraphBuildError: If registry is empty or missing required nodes

    Note:
        This framework is async-first. Use:
        - await graph.ainvoke(input, config)
        - async for chunk in graph.astream(input, config, stream_mode="custom"):
    # OR for multiple modes: stream_mode=["custom", "values"]
    """

    logger.info("Creating framework graph using registry components")

    # Create default checkpointer if none provided
    if checkpointer is None:
        if use_postgres:
            try:
                # Try PostgreSQL when explicitly requested
                checkpointer = create_async_postgres_checkpointer()
                logger.info("Using async PostgreSQL checkpointer")
            except Exception as e:
                # Fall back to memory saver if PostgreSQL fails
                logger.warning(f"PostgreSQL checkpointer failed: {e}")
                logger.info(
                    "Falling back to in-memory checkpointer (install 'langgraph-checkpoint-postgres psycopg[pool]' for production)"
                )
                checkpointer = create_memory_checkpointer()
        else:
            # Default to memory saver for R&D mode
            checkpointer = create_memory_checkpointer()
            logger.info(
                "Using in-memory checkpointer for R&D mode (use use_postgres=True for production)"
            )
    else:
        # User provided checkpointer
        checkpointer_type = type(checkpointer).__name__
        logger.info(f"Using provided {checkpointer_type} checkpointer")

    # Get all nodes from registry (infrastructure + capabilities)
    all_nodes = registry.get_all_nodes().items()  # Get (name, callable) pairs

    logger.info(f"Building graph with {len(all_nodes)} nodes from registry")

    # Validate registry has required nodes
    if len(all_nodes) == 0:
        raise GraphBuildError(
            "Registry contains no nodes. Please ensure applications are properly configured and loaded."
        )

    node_names = [name for name, _ in all_nodes]

    # Check for task_extraction node (required for entry point)
    if "task_extraction" not in node_names:
        raise GraphBuildError(
            f"Registry missing required 'task_extraction' node. Available nodes: {node_names}"
        )

    # Create StateGraph with LangGraph-native TypedDict state
    workflow = StateGraph(AgentState)

    # Add all nodes to workflow - registry provides (name, callable) pairs
    for name, node_callable in all_nodes:
        # All node callables from registry are functions (from decorators)
        workflow.add_node(name, node_callable)
        logger.debug(f"Added node: {name}")

    # Set up routing - router handles everything
    _setup_router_controlled_flow(workflow, node_names)

    # Compile with LangGraph native features - async checkpointing always enabled
    compile_kwargs = {"debug": enable_debug, "checkpointer": checkpointer}

    compiled_graph = workflow.compile(**compile_kwargs)

    # Final success message with correct checkpointer type
    checkpointer_type = type(checkpointer).__name__
    mode = "production" if use_postgres or checkpointer_type == "PostgresSaver" else "R&D"
    logger.success(
        f"Successfully created async framework graph with {len(all_nodes)} nodes and {checkpointer_type} checkpointing enabled ({mode} mode)"
    )

    return compiled_graph


def _setup_router_controlled_flow(workflow: StateGraph, node_names):
    """Set up router-controlled execution flow with router as central decision authority."""

    # Entry point is router - the central decision-making authority
    workflow.set_entry_point("router")

    # Build routing map from all node names (router can route to any registered node)
    routing_map = {name: name for name in node_names}
    routing_map["END"] = END

    # Add routing logic - router is the central hub
    for name in node_names:
        if name == "router":
            # Router uses conditional edges to decide where to route next
            workflow.add_conditional_edges(name, router_conditional_edge, routing_map)
        elif name in ["respond", "clarify", "error"]:
            # Response nodes end the flow (no further routing needed)
            workflow.add_edge(name, END)
        else:
            # All business logic nodes route back to router for next decision
            workflow.add_edge(name, "router")

    logger.debug(
        "Set up router-controlled execution flow with router as central decision authority"
    )


# ==============================================================================
# Checkpointing - Modern PostgreSQL by Default
# ==============================================================================


def create_async_postgres_checkpointer(db_uri: str | None = None) -> BaseCheckpointSaver:
    """
    Create async PostgreSQL checkpointer for production use.

    This is the default production checkpointer.

    Args:
        db_uri: PostgreSQL connection URI. If None, attempts to get from environment
                or uses local development database.

    Returns:
        BaseCheckpointSaver: Configured async PostgreSQL checkpointer

    Raises:
        ImportError: If required dependencies are not installed
        Exception: If connection fails
    """

    # Get database URI from parameter, environment, or default
    if db_uri is None:
        db_uri = os.getenv("POSTGRESQL_URI")
        if db_uri is None:
            # Default to local development database
            db_uri = "postgresql://postgres:postgres@localhost:5432/osprey"
            logger.warning(f"No PostgreSQL URI provided, using default: {db_uri}")

    # Import required components
    try:
        import psycopg_pool
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
    except ImportError as e:
        raise ImportError(
            f"Required PostgreSQL dependencies not installed: {e}. "
            "Install with: pip install langgraph-checkpoint-postgres psycopg[pool]"
        )

    # Create sync connection pool for PostgresSaver
    try:
        # Use sync connection pool with proper configuration
        sync_pool = psycopg_pool.ConnectionPool(
            conninfo=db_uri,
            max_size=20,
            min_size=5,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "prepare_threshold": 0,
            },
        )

        # Create PostgresSaver with sync connection (works with async graphs)
        checkpointer = PostgresSaver(conn=sync_pool)

        # Setup tables on first use
        checkpointer.setup()

        logger.info("Created PostgreSQL checkpointer with sync connection pool")
        return checkpointer

    except Exception as e:
        logger.error(f"Failed to create PostgreSQL checkpointer: {e}")
        raise


def create_memory_checkpointer() -> BaseCheckpointSaver:
    """Create in-memory checkpointer for testing and development."""
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Created in-memory checkpointer (for testing/development)")
    return MemorySaver()


# ==============================================================================
# Setup Functions for Database Initialization
# ==============================================================================


async def setup_postgres_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """
    Set up PostgreSQL checkpointer by creating required tables.

    This should be called once during application startup.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        if isinstance(checkpointer, PostgresSaver):
            # Sync checkpointer - setup is already called in create function
            logger.info("PostgreSQL checkpoint tables already set up (sync)")
        elif isinstance(checkpointer, AsyncPostgresSaver):
            # Async checkpointer - call async setup
            await checkpointer.setup()
            logger.info("PostgreSQL checkpoint tables created/verified (async)")
        else:
            logger.warning("Checkpointer is not a PostgresSaver, skipping table creation")

    except Exception as e:
        logger.error(f"Failed to set up PostgreSQL checkpointer: {e}")
        raise
