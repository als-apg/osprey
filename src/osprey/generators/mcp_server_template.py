"""MCP Server Template Generator.

Generates demo MCP servers for testing Osprey MCP capabilities.
Uses FastMCP for simple, Pythonic MCP server implementation.
"""

from pathlib import Path


def generate_mcp_server(server_name: str = "demo_mcp", port: int = 3001) -> str:
    """Generate MCP server Python code.

    Args:
        server_name: Name for the server
        port: Port to run on

    Returns:
        Complete Python source code for MCP server
    """

    # Always use weather tools for demo
    tool_code = _get_weather_tools()
    tool_metadata = _get_weather_metadata()
    description = "Weather Demo MCP Server"

    server_code = f'''#!/usr/bin/env python3
"""
{description}

A minimal MCP server for testing Osprey MCP capability generation.
Built with FastMCP - a Pythonic framework for MCP server development.

INSTALLATION:
    pip install fastmcp

USAGE:
    python {server_name}_server.py

    Server will run on http://localhost:{port}
    SSE endpoint: http://localhost:{port}/sse

TEST WITH OSPREY:
    osprey generate capability --from-mcp http://localhost:{port} --name {server_name}
"""

from datetime import datetime, timedelta
from typing import Optional, Literal
from fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("{description}")


# =============================================================================
# Tool Implementations
# =============================================================================

{tool_code}


# =============================================================================
# Tool Metadata (for display purposes)
# =============================================================================

TOOL_METADATA = {tool_metadata}


# =============================================================================
# Server Startup with Osprey Styling
# =============================================================================

def print_startup_info():
    """Display server startup information using Osprey styling."""
    try:
        from osprey.cli.styles import console, Messages, Styles

        console.print()
        console.print("=" * 70)
        console.print(f"[{{Styles.HEADER}}]{description}[/{{Styles.HEADER}}]")
        console.print("=" * 70)
        console.print()

        console.print(f"  [{{Styles.LABEL}}]Server URL:[/{{Styles.LABEL}}] [{{Styles.VALUE}}]http://localhost:{port}[/{{Styles.VALUE}}]")
        console.print(f"  [{{Styles.LABEL}}]SSE Endpoint:[/{{Styles.LABEL}}] [{{Styles.VALUE}}]http://localhost:{port}/sse[/{{Styles.VALUE}}]")
        console.print()

        console.print(f"[{{Styles.HEADER}}]Available Tools:[/{{Styles.HEADER}}]")
        console.print()

        # Display tool information
        for i, tool_info in enumerate(TOOL_METADATA, 1):
            console.print(f"  [{{Styles.ACCENT}}]{{i}}. {{tool_info['name']}}[/{{Styles.ACCENT}}]")
            if tool_info.get('description'):
                console.print(f"     [{{Styles.DIM}}]{{tool_info['description']}}[/{{Styles.DIM}}]")

            # Show parameters
            if tool_info.get('parameters'):
                console.print(f"     [{{Styles.LABEL}}]Parameters:[/{{Styles.LABEL}}]")
                for param in tool_info['parameters']:
                    param_name = param['name']
                    param_type = param.get('type', 'any')
                    param_required = param.get('required', False)
                    param_desc = param.get('description', '')
                    req_marker = "required" if param_required else "optional"

                    console.print(f"       • [{{Styles.VALUE}}]{{param_name}}[/{{Styles.VALUE}}] "
                                f"([{{Styles.DIM}}]{{param_type}}, {{req_marker}}[/{{Styles.DIM}}])")
                    if param_desc:
                        console.print(f"         [{{Styles.DIM}}]{{param_desc}}[/{{Styles.DIM}}]")
            console.print()

        console.print(f"[{{Styles.INFO}}]Next Steps:[/{{Styles.INFO}}]")
        console.print(f"  1. Keep this server running")
        console.print(f"  2. In another terminal: [{{Styles.COMMAND}}]osprey generate capability --from-mcp http://localhost:{port} --name {server_name}[/{{Styles.COMMAND}}]")
        console.print()
        console.print(f"  [{{Styles.DIM}}]Press Ctrl+C to stop the server[/{{Styles.DIM}}]")
        console.print("=" * 70)
        console.print()

    except ImportError:
        # Fallback to basic printing if Osprey styles not available
        print("=" * 70)
        print(f"{description}")
        print("=" * 70)
        print(f"\\nServer URL: http://localhost:{port}")
        print(f"SSE endpoint: http://localhost:{port}/sse")
        print()
        print("Available Tools:")
        for i, tool_info in enumerate(TOOL_METADATA, 1):
            print(f"  {{i}}. {{tool_info['name']}}")
            if tool_info.get('description'):
                print(f"     {{tool_info['description']}}")
            if tool_info.get('parameters'):
                print(f"     Parameters:")
                for param in tool_info['parameters']:
                    req = "required" if param.get('required') else "optional"
                    print(f"       - {{param['name']}} ({{param.get('type', 'any')}}, {{req}})")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 70)
        print()


if __name__ == "__main__":
    import sys
    import time
    from threading import Thread

    def delayed_info():
        """Print startup info after FastMCP banner."""
        time.sleep(0.5)  # Wait for FastMCP banner to finish
        print_startup_info()
        sys.stdout.flush()

    # Start info printer in background thread
    Thread(target=delayed_info, daemon=True).start()

    # Run the server with SSE transport
    mcp.run(transport="sse", host="127.0.0.1", port={port})
'''

    return server_code


def _get_weather_tools() -> str:
    """Get Weather tool implementations."""
    return '''@mcp.tool()
def get_current_weather(
    location: str = "San Francisco",
    units: Literal["celsius", "fahrenheit"] = "celsius"
) -> dict:
    """Get current weather conditions for a location.

    If location is not provided, defaults to San Francisco."""
    # Mock weather data
    temp_c = 18
    temp_f = 64
    temp = temp_c if units == "celsius" else temp_f

    # Use current UTC timestamp
    current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "location": location,
        "coordinates": {"lat": 37.7749, "lon": -122.4194},
        "current": {
            "timestamp": current_time,
            "temperature": temp,
            "feels_like": temp - 2,
            "conditions": "Partly Cloudy",
            "description": "Scattered clouds with mild temperatures",
            "humidity": 65,
            "wind_speed": 12,
            "wind_direction": "NW",
            "pressure": 1013,
            "visibility": 10,
            "uv_index": 5
        },
        "units": units,
        "success": True
    }


@mcp.tool()
def get_forecast(
    location: str = "San Francisco",
    days: int = 5,
    units: Literal["celsius", "fahrenheit"] = "celsius"
) -> dict:
    """Get weather forecast for upcoming days.

    If location is not provided, defaults to San Francisco."""
    # Mock forecast data
    forecast_data = []
    conditions_cycle = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for i in range(min(days, 7)):
        temp_high_c = 20 + i
        temp_low_c = 12 + i
        temp_high_f = 68 + i * 2
        temp_low_f = 54 + i * 2

        # Use current date + i days for forecast
        forecast_date = datetime.utcnow() + timedelta(days=i)
        date_str = forecast_date.strftime("%Y-%m-%d")
        day_of_week = day_names[forecast_date.weekday()]

        forecast_data.append({
            "date": date_str,
            "day_of_week": day_of_week,
            "temperature_high": temp_high_c if units == "celsius" else temp_high_f,
            "temperature_low": temp_low_c if units == "celsius" else temp_low_f,
            "conditions": conditions_cycle[i % 5],
            "description": f"Expected {conditions_cycle[i % 5].lower()} conditions",
            "precipitation_chance": (i * 15) % 60,
            "humidity": 55 + (i * 5),
            "wind_speed": 8 + i,
            "uv_index": 6 - i if 6 - i > 0 else 1
        })

    return {
        "location": location,
        "coordinates": {"lat": 35.6762, "lon": 139.6503},
        "forecast_days": len(forecast_data),
        "daily": forecast_data,
        "units": units,
        "success": True
    }


@mcp.tool()
def get_weather_alerts(
    location: str = "San Francisco",
    severity: Literal["all", "severe", "moderate", "minor"] = "all"
) -> dict:
    """Get active weather alerts and warnings for a location.

    If location is not provided, defaults to San Francisco."""
    # Mock alert data - sometimes return alerts, sometimes empty
    # For demo purposes, locations with "Miami" or "Storm" return alerts
    has_alerts = "miami" in location.lower() or "storm" in location.lower()

    # Use current UTC timestamp
    current_time = datetime.utcnow()
    current_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    alerts = []
    if has_alerts:
        # Alert starts now, ends in 2 days
        start_time = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time = (current_time + timedelta(days=2, hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

        alerts = [
            {
                "id": "alert_001",
                "type": "Hurricane Warning",
                "severity": "severe",
                "headline": "Hurricane Warning in effect",
                "description": "Hurricane conditions expected within 36 hours. Prepare for severe weather.",
                "start_time": start_time,
                "end_time": end_time,
                "affected_areas": [f"{location} area"],
                "issued_by": "National Weather Service",
                "instructions": "Follow evacuation orders and prepare emergency supplies."
            }
        ]

    return {
        "location": location,
        "coordinates": {"lat": 25.7617, "lon": -80.1918},
        "alert_count": len(alerts),
        "alerts": alerts,
        "last_updated": current_time_str,
        "success": True
    }
'''


def _get_weather_metadata() -> str:
    """Get weather tool metadata."""
    return """[
    {
        "name": "get_current_weather",
        "description": "Get current weather conditions for a location",
        "parameters": [
            {"name": "location", "type": "string", "required": False, "description": "City name or coordinates (defaults to San Francisco)"},
            {"name": "units", "type": "string", "required": False, "description": "Temperature units (celsius/fahrenheit)"}
        ]
    },
    {
        "name": "get_forecast",
        "description": "Get weather forecast for upcoming days",
        "parameters": [
            {"name": "location", "type": "string", "required": False, "description": "City name or coordinates (defaults to San Francisco)"},
            {"name": "days", "type": "integer", "required": False, "description": "Number of forecast days (1-7)"},
            {"name": "units", "type": "string", "required": False, "description": "Temperature units (celsius/fahrenheit)"}
        ]
    },
    {
        "name": "get_weather_alerts",
        "description": "Get active weather alerts and warnings for a location",
        "parameters": [
            {"name": "location", "type": "string", "required": False, "description": "City name or coordinates (defaults to San Francisco)"},
            {"name": "severity", "type": "string", "required": False, "description": "Filter by alert severity (all/severe/moderate/minor)"}
        ]
    }
]"""


def write_mcp_server_file(
    output_path: Path, server_name: str = "demo_mcp", port: int = 3001
) -> Path:
    """Generate and write MCP server file.

    Args:
        output_path: Where to write the server file
        server_name: Name for the server
        port: Port to run on

    Returns:
        Path to written file
    """
    code = generate_mcp_server(server_name, port)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    output_path.chmod(0o755)  # Make executable

    return output_path
