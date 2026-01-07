"""
Real-time Analytics WebSocket Server

This module provides WebSocket-based real-time analytics streaming for the
Osprey Framework GUI analytics dashboard, replacing polling with push-based updates.

Key Features:
- WebSocket server for dashboard connections
- Real-time metrics streaming
- Event-driven metric updates
- Multiple concurrent client support
- Automatic reconnection handling
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("realtime_analytics_websocket")

# Try to import websockets, provide fallback if not available
try:
    import websockets
    from websockets.server import WebSocketServerProtocol

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any  # Type hint fallback
    logger.warning("websockets not available. Install with: pip install websockets")


@dataclass
class MetricUpdate:
    """Real-time metric update message."""

    timestamp: float
    metric_type: str  # "routing", "cache", "feedback", etc.
    data: dict[str, Any]

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {"timestamp": self.timestamp, "type": self.metric_type, "data": self.data}
        )


@dataclass
class ClientConnection:
    """WebSocket client connection info."""

    websocket: WebSocketServerProtocol
    client_id: str
    connected_at: float
    subscriptions: set[str]  # Metric types client is subscribed to

    def __hash__(self):
        return hash(self.client_id)


class MetricsEventBus:
    """
    Event bus for metrics updates.

    Allows components to publish metrics updates that are
    automatically broadcast to connected WebSocket clients.
    """

    def __init__(self):
        """Initialize event bus."""
        self.listeners: list[Callable[[MetricUpdate], None]] = []
        logger.info("Initialized MetricsEventBus")

    def subscribe(self, listener: Callable[[MetricUpdate], None]):
        """Subscribe to metric updates.

        Args:
            listener: Callback function for metric updates.
        """
        self.listeners.append(listener)
        logger.debug(f"Added listener (total: {len(self.listeners)})")

    def unsubscribe(self, listener: Callable[[MetricUpdate], None]):
        """Unsubscribe from metric updates.

        Args:
            listener: Callback function to remove.
        """
        if listener in self.listeners:
            self.listeners.remove(listener)
            logger.debug(f"Removed listener (total: {len(self.listeners)})")

    def publish(self, update: MetricUpdate):
        """Publish metric update to all listeners.

        Args:
            update: Metric update to publish.
        """
        for listener in self.listeners:
            try:
                listener(update)
            except Exception as e:
                logger.error(f"Error in listener: {e}")


class RealtimeAnalyticsWebSocket:
    """
    WebSocket server for real-time analytics streaming.

    Provides:
    - WebSocket endpoint for dashboard clients
    - Real-time metric broadcasting
    - Client subscription management
    - Automatic reconnection support
    """

    def __init__(self, host: str = "localhost", port: int = 8765, enable_compression: bool = True):
        """Initialize WebSocket server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            enable_compression: Enable WebSocket compression.
        """
        self.host = host
        self.port = port
        self.enable_compression = enable_compression

        # Client management
        self.clients: set[ClientConnection] = set()
        self.client_counter = 0

        # Event bus
        self.event_bus = MetricsEventBus()
        self.event_bus.subscribe(self._on_metric_update)

        # Server state
        self.server = None
        self.running = False

        # Statistics
        self.stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "start_time": None,
        }

        logger.info(
            f"Initialized RealtimeAnalyticsWebSocket: "
            f"{host}:{port}, compression={enable_compression}"
        )

    async def start(self):
        """Start WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start server: websockets not available")
            return

        if self.running:
            logger.warning("Server already running")
            return

        try:
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                compression="deflate" if self.enable_compression else None,
            )

            self.running = True
            self.stats["start_time"] = time.time()

            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop(self):
        """Stop WebSocket server."""
        if not self.running:
            return

        self.running = False

        # Close all client connections
        for client in list(self.clients):
            try:
                await client.websocket.close()
            except Exception as e:
                logger.error(f"Error closing client connection: {e}")

        self.clients.clear()

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info("WebSocket server stopped")

    def broadcast_metric(
        self, metric_type: str, data: dict[str, Any], target_subscriptions: set[str] | None = None
    ):
        """Broadcast metric update to subscribed clients.

        Args:
            metric_type: Type of metric.
            data: Metric data.
            target_subscriptions: Optional set of subscription types to target.
        """
        update = MetricUpdate(timestamp=time.time(), metric_type=metric_type, data=data)

        self.event_bus.publish(update)

    def get_statistics(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with server statistics.
        """
        uptime = None
        if self.stats["start_time"]:
            uptime = time.time() - self.stats["start_time"]

        return {
            "running": self.running,
            "connected_clients": len(self.clients),
            "total_connections": self.stats["total_connections"],
            "messages_sent": self.stats["total_messages_sent"],
            "messages_received": self.stats["total_messages_received"],
            "uptime_seconds": uptime,
            "host": self.host,
            "port": self.port,
        }

    # Private methods

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket client connection.

        Args:
            websocket: WebSocket connection.
            path: Connection path.
        """
        # Create client connection
        self.client_counter += 1
        client = ClientConnection(
            websocket=websocket,
            client_id=f"client_{self.client_counter}",
            connected_at=time.time(),
            subscriptions=set(),
        )

        self.clients.add(client)
        self.stats["total_connections"] += 1

        logger.info(f"Client connected: {client.client_id} (total: {len(self.clients)})")

        try:
            # Send welcome message
            await self._send_to_client(
                client,
                {"type": "welcome", "client_id": client.client_id, "server_time": time.time()},
            )

            # Handle messages
            async for message in websocket:
                await self._handle_message(client, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client.client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client.client_id}: {e}")
        finally:
            # Clean up
            self.clients.discard(client)
            logger.info(f"Client removed: {client.client_id} (remaining: {len(self.clients)})")

    async def _handle_message(self, client: ClientConnection, message: str):
        """Handle message from client.

        Args:
            client: Client connection.
            message: Message string.
        """
        self.stats["total_messages_received"] += 1

        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "subscribe":
                # Subscribe to metric types
                metric_types = data.get("metric_types", [])
                client.subscriptions.update(metric_types)

                await self._send_to_client(
                    client, {"type": "subscribed", "metric_types": list(client.subscriptions)}
                )

                logger.debug(f"Client {client.client_id} subscribed to: {metric_types}")

            elif msg_type == "unsubscribe":
                # Unsubscribe from metric types
                metric_types = data.get("metric_types", [])
                client.subscriptions.difference_update(metric_types)

                await self._send_to_client(
                    client, {"type": "unsubscribed", "metric_types": metric_types}
                )

                logger.debug(f"Client {client.client_id} unsubscribed from: {metric_types}")

            elif msg_type == "ping":
                # Respond to ping
                await self._send_to_client(client, {"type": "pong", "timestamp": time.time()})

            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {client.client_id}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _send_to_client(self, client: ClientConnection, data: dict):
        """Send data to client.

        Args:
            client: Client connection.
            data: Data to send.
        """
        try:
            message = json.dumps(data)
            await client.websocket.send(message)
            self.stats["total_messages_sent"] += 1
        except Exception as e:
            logger.error(f"Error sending to client {client.client_id}: {e}")

    def _on_metric_update(self, update: MetricUpdate):
        """Handle metric update from event bus.

        Args:
            update: Metric update.
        """
        # Broadcast to subscribed clients
        asyncio.create_task(self._broadcast_update(update))

    async def _broadcast_update(self, update: MetricUpdate):
        """Broadcast update to subscribed clients.

        Args:
            update: Metric update to broadcast.
        """
        message = update.to_json()

        # Send to all subscribed clients
        for client in list(self.clients):
            # Check if client is subscribed to this metric type
            if not client.subscriptions or update.metric_type in client.subscriptions:
                try:
                    await client.websocket.send(message)
                    self.stats["total_messages_sent"] += 1
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client.client_id}: {e}")


class AnalyticsMetricsPublisher:
    """
    Publisher for analytics metrics to WebSocket server.

    Integrates with existing RoutingAnalytics to publish
    metrics in real-time.
    """

    def __init__(self, websocket_server: RealtimeAnalyticsWebSocket):
        """Initialize publisher.

        Args:
            websocket_server: WebSocket server instance.
        """
        self.websocket_server = websocket_server
        logger.info("Initialized AnalyticsMetricsPublisher")

    def publish_routing_decision(
        self,
        query: str,
        project: str,
        confidence: float,
        routing_time_ms: float,
        cache_hit: bool,
        mode: str,
    ):
        """Publish routing decision metric.

        Args:
            query: User query.
            project: Selected project.
            confidence: Routing confidence.
            routing_time_ms: Routing time in milliseconds.
            cache_hit: Whether decision came from cache.
            mode: Routing mode (automatic/manual).
        """
        self.websocket_server.broadcast_metric(
            metric_type="routing_decision",
            data={
                "query": query[:100],  # Truncate for privacy
                "project": project,
                "confidence": confidence,
                "routing_time_ms": routing_time_ms,
                "cache_hit": cache_hit,
                "mode": mode,
                "timestamp": time.time(),
            },
        )

    def publish_cache_statistics(self, stats: dict[str, Any]):
        """Publish cache statistics.

        Args:
            stats: Cache statistics.
        """
        self.websocket_server.broadcast_metric(metric_type="cache_stats", data=stats)

    def publish_analytics_summary(self, summary: dict[str, Any]):
        """Publish analytics summary.

        Args:
            summary: Analytics summary data.
        """
        self.websocket_server.broadcast_metric(metric_type="analytics_summary", data=summary)

    def publish_feedback_event(
        self, query: str, project: str, feedback: str, correct_project: str | None = None
    ):
        """Publish user feedback event.

        Args:
            query: User query.
            project: Selected project.
            feedback: Feedback type (correct/incorrect).
            correct_project: Correct project if feedback was incorrect.
        """
        self.websocket_server.broadcast_metric(
            metric_type="feedback_event",
            data={
                "query": query[:100],
                "project": project,
                "feedback": feedback,
                "correct_project": correct_project,
                "timestamp": time.time(),
            },
        )
