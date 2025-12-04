"""
{{ app_display_name }} - Control System Assistant

A production-grade template demonstrating control system integration patterns.
Based on the ALS Accelerator Assistant deployment (arXiv:2509.17255).

Features:
- Natural language channel finding (in-context or hierarchical pipelines)
- Historical data analysis (mock archiver)
- Live control system reads (mock EPICS)
- Complete benchmarking system
- Optional MCP server deployment

Architecture:
- Service Layer: Business logic (services/)
- Capability Layer: Osprey integration (capabilities/)
- Optional MCP: Standalone server deployment (services/channel_finder/)
"""

__version__ = "0.1.0"

