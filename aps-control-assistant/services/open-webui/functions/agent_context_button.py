"""
title: ALS Assistant Agent Context
author: ALS Assistant Team
version: 0.1.0
required_open_webui_version: 0.5.1
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyUzYuNDggMjIgMTIgMjJTMjIgMTcuNTIgMjIgMTJTMTcuNTIgMiAxMiAyWk0xMiAyMEM3LjU5IDIwIDQgMTYuNDEgNCAxMlM3LjU5IDQgMTIgNFMyMCA3LjU5IDIwIDEyUzE2LjQxIDIwIDEyIDIwWiIgZmlsbD0iY3VycmVudENvbG9yIi8+CjxwYXRoIGQ9Ik0xMiA2VjhNMTIgMTZWMThNMTAgMTJIMTRNOCAxMkg2TTE4IDEySDE2IiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KPC9zdmc+
Description: View current ALS Assistant Agent context data and available information
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Action:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        show_detailed_values: bool = Field(
            default=True, description="Show detailed data values and sample data"
        )
        show_technical_info: bool = Field(
            default=False, description="Show technical information like data types and structure"
        )
        max_sample_items: int = Field(
            default=5, description="Maximum number of sample items to show for lists"
        )

    def __init__(self):
        self.valves = self.Valves()

    def extract_context_summary_from_messages(self, messages: list) -> dict[str, Any] | None:
        """Extract agent context summary from assistant messages."""

        try:
            logger.info(f"Extracting context from {len(messages)} messages")

            # Look through messages in reverse order (most recent first)
            for i, message in enumerate(reversed(messages)):
                try:
                    logger.debug(f"Checking message {i}: role={message.get('role')}, has_info={message.get('info') is not None}")

                    if message.get("role") == "assistant" and message.get("info"):
                        info_keys = list(message["info"].keys())
                        logger.debug(f"Message {i} info keys: {info_keys}")

                        # Check for context summary (check both old and new key names for compatibility)
                        if "als_assistant_agent_context" in message["info"]:
                            context_data = message["info"]["als_assistant_agent_context"]
                            logger.info(f"Found agent context with {context_data.get('total_context_items', 0)} items")
                            return context_data
                        elif "als_assistant_context_summary" in message["info"]:
                            context_data = message["info"]["als_assistant_context_summary"]
                            logger.info(f"Found agent context with {len(context_data.get('context_details', {}))} categories")
                            return context_data

                except Exception as e:
                    logger.error(f"Error processing message {i}: {e}")
                    continue

            logger.info("No agent context found in any message")
            return None

        except Exception as e:
            logger.error(f"Error extracting context from messages: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def format_context_summary_markdown(self, context_summary: dict[str, Any], user_valves) -> str:
        """Format the agent context summary as a well-structured markdown string."""
        # Handle both new and old data formats
        context_data = context_summary.get("context_data") or context_summary.get("context_details", {})

        if not context_summary or not context_data:
            return "# 🧠 ALS Assistant Agent Context\n\n> No context data available. The agent has not yet collected or processed any data."

        # Header with overview - handle both new and old formats
        total_categories = context_summary.get("context_types_count") or len(context_data)
        total_items = context_summary.get("total_context_items", 0)

        markdown = "# 🧠 ALS Assistant Agent Context\n\n"
        markdown += f"📊 **Available Context Categories:** {total_categories}\n"
        markdown += f"📋 **Total Context Items:** {total_items}\n\n"

        categories = list(context_data.keys())
        if categories:
            markdown += f"**Categories:** {', '.join(categories)}\n\n"

        markdown += "---\n\n"

        # Process each context category
        for context_type, contexts_dict in context_data.items():
            # Category header
            category_emoji = self._get_category_emoji(context_type)
            markdown += f"## {category_emoji} {context_type.replace('_', ' ').title()}\n\n"

            # Process each context item in this category
            for context_key, context_info in contexts_dict.items():
                context_type_name = context_info.get("type", "Unknown")
                markdown += f"### 🔹 {context_key}\n\n"
                markdown += f"**Type:** {context_type_name}\n\n"

                # Create summary table
                markdown = self._add_context_summary_table(markdown, context_info, user_valves)

                # Add detailed values if requested
                if user_valves.show_detailed_values:
                    markdown = self._add_detailed_values(markdown, context_info, user_valves)

                markdown += "\n---\n\n"

        # Footer
        markdown += "✨ *Agent context data available for use in subsequent queries*"

        return markdown

    def _get_category_emoji(self, context_type: str) -> str:
        """Get appropriate emoji for context category."""
        emoji_map = {
            "PV_ADDRESSES": "📍",
            "TIME_RANGE": "⏰",
            "PV_VALUES": "📊",
            "ARCHIVER_DATA": "📈",
            "ANALYSIS_RESULTS": "🔬",
            "VISUALIZATION_RESULTS": "📊",
            "OPERATION_RESULTS": "⚙️",
            "MEMORY_CONTEXT": "🧠",
            "CONVERSATION_RESULTS": "💬"
        }
        return emoji_map.get(context_type, "📁")

    def _add_context_summary_table(self, markdown: str, context_info: dict[str, Any], user_valves) -> str:
        """Add a summary table for the context item."""
        # Common fields
        summary_table = "| Field | Value |\n|-------|-------|\n"

        # Type-specific summary information
        context_type = context_info.get("type", "Unknown")

        if context_type == "PV Addresses":
            total_pvs = context_info.get("total_pvs", 0)
            summary_table += f"| **Total PVs** | {total_pvs} |\n"
            description = context_info.get("description", "N/A")
            summary_table += f"| **Description** | {description} |\n"

        elif context_type == "Time Range":
            start_time = context_info.get("start_time", "N/A")
            end_time = context_info.get("end_time", "N/A")
            duration = context_info.get("duration", "N/A")
            summary_table += f"| **Start Time** | {start_time} |\n"
            summary_table += f"| **End Time** | {end_time} |\n"
            summary_table += f"| **Duration** | {duration} |\n"

        elif context_type == "PV Values":
            pv_data = context_info.get("pv_data", {})
            summary_table += f"| **PV Count** | {len(pv_data)} |\n"

        elif context_type == "Archiver Data":
            total_points = context_info.get("total_points", 0)
            pv_count = context_info.get("pv_count", 0)
            time_info = context_info.get("time_info", "N/A")
            summary_table += f"| **Total Points** | {total_points:,} |\n"
            summary_table += f"| **PV Count** | {pv_count} |\n"
            summary_table += f"| **Time Info** | {time_info} |\n"

        elif context_type in ["Analysis Results", "Visualization Results", "Operation Results"]:
            field_count = context_info.get("field_count", 0)
            summary_table += f"| **Field Count** | {field_count} |\n"
            available_fields = context_info.get("available_fields", [])
            if available_fields:
                fields_str = ", ".join(available_fields[:5])
                if len(available_fields) > 5:
                    fields_str += f" (and {len(available_fields) - 5} more)"
                summary_table += f"| **Available Fields** | {fields_str} |\n"

        elif context_type == "Memory Context":
            memory_count = context_info.get("memory_count", 0)
            oldest_memory = context_info.get("oldest_memory", "N/A")
            newest_memory = context_info.get("newest_memory", "N/A")
            summary_table += f"| **Memory Count** | {memory_count} |\n"
            summary_table += f"| **Oldest Memory** | {oldest_memory} |\n"
            summary_table += f"| **Newest Memory** | {newest_memory} |\n"

        elif context_type == "Conversation Results":
            message_type = context_info.get("message_type", "N/A")
            summary_table += f"| **Message Type** | {message_type} |\n"

        return markdown + summary_table + "\n"

    def _add_detailed_values(self, markdown: str, context_info: dict[str, Any], user_valves) -> str:
        """Add detailed values section."""
        context_type = context_info.get("type", "Unknown")

        if context_type == "PV Addresses":
            pv_list = context_info.get("pv_list", [])
            if pv_list:
                markdown += "**PV Addresses:**\n"
                for i, pv in enumerate(pv_list[:user_valves.max_sample_items]):
                    markdown += f"- `{pv}`\n"
                if len(pv_list) > user_valves.max_sample_items:
                    markdown += f"- *(and {len(pv_list) - user_valves.max_sample_items} more)*\n"
                markdown += "\n"

        elif context_type == "PV Values":
            pv_data = context_info.get("pv_data", {})
            if pv_data:
                markdown += "**PV Values:**\n"
                count = 0
                for pv_name, pv_info in pv_data.items():
                    if count >= user_valves.max_sample_items:
                        break
                    value = pv_info.get("value", "N/A")
                    units = pv_info.get("units", "")
                    timestamp = pv_info.get("timestamp", "N/A")
                    markdown += f"- `{pv_name}`: {value} {units} *(@ {timestamp})*\n"
                    count += 1
                if len(pv_data) > user_valves.max_sample_items:
                    markdown += f"- *(and {len(pv_data) - user_valves.max_sample_items} more)*\n"
                markdown += "\n"

        elif context_type == "Archiver Data":
            pv_names = context_info.get("pv_names", [])
            sample_values = context_info.get("sample_values", {})

            if pv_names:
                markdown += "**Available PVs:**\n"
                for pv in pv_names[:user_valves.max_sample_items]:
                    markdown += f"- `{pv}`"
                    if pv in sample_values:
                        values = sample_values[pv][:3]  # Show first 3 sample values
                        values_str = ", ".join([f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in values])
                        markdown += f" (sample: {values_str}...)"
                    markdown += "\n"
                if len(pv_names) > user_valves.max_sample_items:
                    markdown += f"- *(and {len(pv_names) - user_valves.max_sample_items} more)*\n"
                markdown += "\n"

        elif context_type in ["Analysis Results", "Visualization Results", "Operation Results"]:
            results = context_info.get("results", {})
            if results:
                markdown += "**Results:**\n"
                count = 0
                for key, value in results.items():
                    if count >= user_valves.max_sample_items:
                        break
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        markdown += f"- **{key.replace('_', ' ').title()}**: *(large data structure)*\n"
                    else:
                        markdown += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                    count += 1
                if len(results) > user_valves.max_sample_items:
                    markdown += f"- *(and {len(results) - user_valves.max_sample_items} more)*\n"
                markdown += "\n"

        elif context_type == "Memory Context":
            memories = context_info.get("memories", [])
            if memories:
                markdown += "**Memory Entries:**\n"
                for i, memory in enumerate(memories[:user_valves.max_sample_items]):
                    content = memory.get("content", "N/A")
                    timestamp = memory.get("timestamp", "N/A")
                    markdown += f"- {content} *(@ {timestamp})*\n"
                if len(memories) > user_valves.max_sample_items:
                    markdown += f"- *(and {len(memories) - user_valves.max_sample_items} more)*\n"
                markdown += "\n"

        elif context_type == "Conversation Results":
            full_response = context_info.get("full_response", "N/A")
            if full_response and len(full_response) > 200:
                markdown += f"**Response Preview:** {full_response[:200]}...\n\n"
            elif full_response:
                markdown += f"**Full Response:** {full_response}\n\n"

        return markdown

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict | None:
        """Display formatted agent context using a popup modal."""
        logger.info(f"User - Name: {__user__['name']}, ID: {__user__['id']} - Requesting ALS Assistant agent context")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Retrieving agent context...", "done": False},
            }
        )

        try:
            # Log debug information about the request
            logger.info(f"Processing agent context request for user {__user__.get('name', 'unknown')}")
            logger.info(f"Message count: {len(body.get('messages', []))}")

            # Extract context summary from the last assistant message
            context_summary = self.extract_context_summary_from_messages(body.get("messages", []))

            if not context_summary:
                logger.info("No agent context found in messages")
                # Show no context popup
                no_context_js = """
                try {
                    // Remove any existing context popup
                    const existingPopup = document.getElementById('agent-context-popup');
                    if (existingPopup) {
                        existingPopup.remove();
                    }

                    // Create overlay
                    const overlay = document.createElement('div');
                    overlay.id = 'agent-context-popup';
                    overlay.style.cssText = `
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.8);
                        z-index: 10000;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    `;

                    // Create popup content
                    const popup = document.createElement('div');
                    popup.style.cssText = `
                        background: white;
                        border-radius: 8px;
                        padding: 24px;
                        width: 90%;
                        max-width: 600px;
                        max-height: 80vh;
                        overflow-y: auto;
                        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                        color: #333;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    `;

                    popup.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 2px solid #e5e7eb;">
                            <h2 style="margin: 0; color: #374151; font-size: 20px; font-weight: 600;">🧠 ALS Assistant Agent Context</h2>
                            <button id="context-close-btn" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px;">Close</button>
                        </div>

                        <div style="text-align: center; padding: 40px 20px; color: #6b7280; font-size: 16px; line-height: 1.6;">
                            <div style="font-size: 48px; margin-bottom: 16px;">⚠️</div>
                            <h3 style="margin: 0 0 16px 0; color: #374151; font-size: 18px; font-weight: 600;">No Agent Context Available</h3>
                            <p style="margin: 0; color: #6b7280;">No ALS Assistant agent context found in recent messages. The agent context is populated when:</p>
                            <ul style="text-align: left; margin: 20px 0; color: #6b7280; line-height: 1.8;">
                                <li><strong>PV addresses</strong> are found</li>
                                <li><strong>Time ranges</strong> are parsed</li>
                                <li><strong>Data is retrieved</strong> from archiver or EPICS</li>
                                <li><strong>Analysis</strong> is performed</li>
                                <li><strong>Operations</strong> are executed</li>
                            </ul>
                            <p style="margin: 0; color: #6b7280; font-style: italic;">Execute an ALS Assistant query that involves data collection or analysis to populate the agent context.</p>
                        </div>
                    `;

                    // Add event listeners
                    overlay.appendChild(popup);
                    document.body.appendChild(overlay);

                    // Close button
                    document.getElementById('context-close-btn').onclick = function() {
                        overlay.remove();
                    };

                    // Close on overlay click
                    overlay.onclick = function(e) {
                        if (e.target === overlay) {
                            overlay.remove();
                        }
                    };

                } catch (error) {
                    console.error('Error creating context popup:', error);
                    alert('Error displaying agent context');
                }
                """

                await __event_call__(
                    {
                        "type": "execute",
                        "data": {"code": no_context_js},
                    }
                )

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "No agent context available", "done": True},
                    }
                )
                return

            logger.info(f"Found agent context: {list(context_summary.keys())}")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Formatting agent context...", "done": False},
                }
            )

            # Format the context summary as HTML for the popup
            formatted_context = self.format_context_summary_html(context_summary, user_valves)

            # Create JavaScript to show popup with context
            context_js = f"""
            try {{
                // Remove any existing context popup
                const existingPopup = document.getElementById('agent-context-popup');
                if (existingPopup) {{
                    existingPopup.remove();
                }}

                // Create overlay
                const overlay = document.createElement('div');
                overlay.id = 'agent-context-popup';
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.8);
                    z-index: 10000;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                `;

                // Create popup content
                const popup = document.createElement('div');
                popup.style.cssText = `
                    background: white;
                    border-radius: 8px;
                    padding: 24px;
                    width: 90%;
                    max-width: 1000px;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                    color: #333;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                `;

                popup.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 2px solid #e5e7eb;">
                        <h2 style="margin: 0; color: #374151; font-size: 20px; font-weight: 600;">🧠 ALS Assistant Agent Context</h2>
                        <button id="context-close-btn" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px;">Close</button>
                    </div>

                    <div style="line-height: 1.6;">
                        {formatted_context}
                    </div>
                `;

                // Add event listeners
                overlay.appendChild(popup);
                document.body.appendChild(overlay);

                // Close button
                document.getElementById('context-close-btn').onclick = function() {{
                    overlay.remove();
                }};

                // Close on overlay click
                overlay.onclick = function(e) {{
                    if (e.target === overlay) {{
                        overlay.remove();
                    }}
                }};

            }} catch (error) {{
                console.error('Error creating context popup:', error);
                alert('Error displaying agent context: ' + error.message);
            }}
            """

            await __event_call__(
                {
                    "type": "execute",
                    "data": {"code": context_js},
                }
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Agent context displayed", "done": True},
                }
            )

            context_categories = len(context_summary.get("context_details", {}))
            logger.info(f"User - Name: {__user__['name']}, ID: {__user__['id']} - Agent context popup displayed successfully ({context_categories} categories)")

        except Exception as e:
            logger.error(f"Error processing agent context: {e}")

            error_js = f"""
            try {{
                // Remove any existing context popup
                const existingPopup = document.getElementById('agent-context-popup');
                if (existingPopup) {{
                    existingPopup.remove();
                }}

                // Create overlay
                const overlay = document.createElement('div');
                overlay.id = 'agent-context-popup';
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.8);
                    z-index: 10000;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                `;

                // Create popup content
                const popup = document.createElement('div');
                popup.style.cssText = `
                    background: white;
                    border-radius: 8px;
                    padding: 24px;
                    width: 90%;
                    max-width: 600px;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                    color: #333;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                `;

                popup.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 2px solid #e5e7eb;">
                        <h2 style="margin: 0; color: #dc2626; font-size: 20px; font-weight: 600;">❌ Error Processing Agent Context</h2>
                        <button id="context-close-btn" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px;">Close</button>
                    </div>

                    <div style="padding: 20px; background: #fef2f2; border-radius: 6px; border: 1px solid #fecaca;">
                        <p style="margin: 0 0 16px 0; color: #dc2626; font-weight: 500;">An error occurred while processing the agent context:</p>
                        <pre style="margin: 0; font-family: monospace; background: #fff; padding: 12px; border-radius: 4px; border: 1px solid #e5e7eb; color: #374151; font-size: 13px; overflow-x: auto;">{str(e)}</pre>
                        <p style="margin: 16px 0 0 0; color: #6b7280; font-size: 14px;">Please check the logs for more details.</p>
                    </div>
                `;

                // Add event listeners
                overlay.appendChild(popup);
                document.body.appendChild(overlay);

                // Close button
                document.getElementById('context-close-btn').onclick = function() {{
                    overlay.remove();
                }};

                // Close on overlay click
                overlay.onclick = function(e) {{
                    if (e.target === overlay) {{
                        overlay.remove();
                    }}
                }};

            }} catch (error) {{
                console.error('Error creating error popup:', error);
                alert('Error displaying agent context error: ' + error.message);
            }}
            """

            await __event_call__(
                {
                    "type": "execute",
                    "data": {"code": error_js},
                }
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Error processing agent context", "done": True},
                }
            )

    def format_context_summary_html(self, context_summary: dict[str, Any], user_valves) -> str:
        """Format the agent context summary as HTML for the popup display."""
        # Handle both new and old data formats
        context_data = context_summary.get("context_data") or context_summary.get("context_details", {})

        if not context_summary or not context_data:
            return '<div style="text-align: center; padding: 40px; color: #6b7280; font-style: italic;">No context data available.</div>'

        # Header with overview - handle both new and old formats
        total_categories = context_summary.get("context_types_count") or len(context_data)
        total_items = context_summary.get("total_context_items", 0)

        html = f"""
        <div style="margin-bottom: 24px; padding: 20px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 18px; font-weight: 600; color: #1f2937;">📊 Context Overview</div>
                <div style="font-size: 14px; color: #6b7280;">Available for use in subsequent queries</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
                <div style="text-align: center; padding: 12px; background: white; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 24px; font-weight: 700; color: #059669;">{total_categories}</div>
                    <div style="font-size: 13px; color: #6b7280; font-weight: 500;">Categories</div>
                </div>
                <div style="text-align: center; padding: 12px; background: white; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 24px; font-weight: 700; color: #0369a1;">{total_items}</div>
                    <div style="font-size: 13px; color: #6b7280; font-weight: 500;">Total Items</div>
                </div>
            </div>
        """

        categories = list(context_data.keys())
        if categories:
            html += f'<div style="font-size: 14px; color: #4b5563;"><strong>Categories:</strong> {", ".join(categories)}</div>'

        html += '</div>'

        # Process each context category
        for context_type, contexts_dict in context_data.items():
            # Category header
            category_emoji = self._get_category_emoji(context_type)
            html += f"""
            <div style="margin-bottom: 28px; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;">
                <div style="background: #f1f5f9; padding: 16px; border-bottom: 1px solid #e2e8f0;">
                    <h3 style="margin: 0; color: #1f2937; font-size: 16px; font-weight: 600;">
                        {category_emoji} {context_type.replace('_', ' ').title()}
                    </h3>
                </div>
                <div style="padding: 20px;">
            """

            # Process each context item in this category
            for context_key, context_info in contexts_dict.items():
                context_type_name = context_info.get("type", "Unknown")
                html += f"""
                <div style="margin-bottom: 24px; padding: 16px; background: #fafbfc; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 12px 0; color: #1f2937; font-size: 15px; font-weight: 600;">
                        🔹 {context_key}
                    </h4>
                    <div style="margin-bottom: 12px; font-size: 13px; color: #6b7280;">
                        <strong>Type:</strong> {context_type_name}
                    </div>
                """

                # Create summary table
                html += self._add_context_summary_table_html(context_info, user_valves)

                # Add detailed values if requested
                if user_valves.show_detailed_values:
                    html += self._add_detailed_values_html(context_info, user_valves)

                html += '</div>'

            html += '</div></div>'

        return html

    def _add_context_summary_table_html(self, context_info: dict[str, Any], user_valves) -> str:
        """Add a summary table for the context item in HTML format."""
        # Common fields
        summary_html = """
        <div style="overflow-x: auto; margin-bottom: 16px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="background: #e2e8f0;">
                        <th style="padding: 8px 12px; text-align: left; border: 1px solid #cbd5e1; font-weight: 600; color: #374151;">Field</th>
                        <th style="padding: 8px 12px; text-align: left; border: 1px solid #cbd5e1; font-weight: 600; color: #374151;">Value</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Type-specific summary information
        context_type = context_info.get("type", "Unknown")

        if context_type == "PV Addresses":
            total_pvs = context_info.get("total_pvs", 0)
            description = context_info.get("description", "N/A")
            summary_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Total PVs</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{total_pvs}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Description</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{description}</td>
                </tr>
            """

        elif context_type == "Time Range":
            start_time = context_info.get("start_time", "N/A")
            end_time = context_info.get("end_time", "N/A")
            duration = context_info.get("duration", "N/A")
            summary_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Start Time</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{start_time}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">End Time</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{end_time}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Duration</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{duration}</td>
                </tr>
            """

        elif context_type == "PV Values":
            pv_data = context_info.get("pv_data", {})
            summary_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">PV Count</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{len(pv_data)}</td>
                </tr>
            """

        elif context_type == "Archiver Data":
            total_points = context_info.get("total_points", 0)
            pv_count = context_info.get("pv_count", 0)
            time_info = context_info.get("time_info", "N/A")
            summary_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Total Points</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{total_points:,}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">PV Count</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{pv_count}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Time Info</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{time_info}</td>
                </tr>
            """

        elif context_type in ["Analysis Results", "Visualization Results", "Operation Results"]:
            field_count = context_info.get("field_count", 0)
            summary_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Field Count</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{field_count}</td>
                </tr>
            """
            available_fields = context_info.get("available_fields", [])
            if available_fields:
                fields_str = ", ".join(available_fields[:5])
                if len(available_fields) > 5:
                    fields_str += f" (and {len(available_fields) - 5} more)"
                summary_html += f"""
                    <tr>
                        <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Available Fields</td>
                        <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{fields_str}</td>
                    </tr>
                """

        elif context_type == "Memory Context":
            memory_count = context_info.get("memory_count", 0)
            oldest_memory = context_info.get("oldest_memory", "N/A")
            newest_memory = context_info.get("newest_memory", "N/A")
            summary_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Memory Count</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{memory_count}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Oldest Memory</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{oldest_memory}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Newest Memory</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{newest_memory}</td>
                </tr>
            """

        elif context_type == "Conversation Results":
            message_type = context_info.get("message_type", "N/A")
            summary_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Message Type</td>
                    <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{message_type}</td>
                </tr>
            """

        summary_html += '</tbody></table></div>'
        return summary_html

    def _add_detailed_values_html(self, context_info: dict[str, Any], user_valves) -> str:
        """Add detailed values section in HTML format."""
        context_type = context_info.get("type", "Unknown")
        html = ""

        if context_type == "PV Addresses":
            pv_list = context_info.get("pv_list", [])
            if pv_list:
                html += """
                <div style="margin-top: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">PV Addresses:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto;">
                """
                for i, pv in enumerate(pv_list[:user_valves.max_sample_items]):
                    html += f'<div style="margin-bottom: 4px; color: #1f2937;">• {pv}</div>'
                if len(pv_list) > user_valves.max_sample_items:
                    html += f'<div style="color: #6b7280; font-style: italic;">• (and {len(pv_list) - user_valves.max_sample_items} more)</div>'
                html += '</div></div>'

        elif context_type == "PV Values":
            pv_data = context_info.get("pv_data", {})
            if pv_data:
                html += """
                <div style="margin-top: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">PV Values:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 12px; max-height: 200px; overflow-y: auto;">
                """
                count = 0
                for pv_name, pv_info in pv_data.items():
                    if count >= user_valves.max_sample_items:
                        break
                    value = pv_info.get("value", "N/A")
                    units = pv_info.get("units", "")
                    timestamp = pv_info.get("timestamp", "N/A")
                    html += f'<div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px; border: 1px solid #e2e8f0;"><strong style="color: #1f2937; font-family: monospace;">{pv_name}:</strong> <span style="color: #059669; font-weight: 600;">{value} {units}</span> <span style="color: #6b7280; font-size: 11px;">@ {timestamp}</span></div>'
                    count += 1
                if len(pv_data) > user_valves.max_sample_items:
                    html += f'<div style="color: #6b7280; font-style: italic; text-align: center; margin-top: 8px;">• (and {len(pv_data) - user_valves.max_sample_items} more)</div>'
                html += '</div></div>'

        elif context_type == "Archiver Data":
            pv_names = context_info.get("pv_names", [])
            sample_values = context_info.get("sample_values", {})

            if pv_names:
                html += """
                <div style="margin-top: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">Available PVs:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 12px; max-height: 200px; overflow-y: auto;">
                """
                for pv in pv_names[:user_valves.max_sample_items]:
                    html += f'<div style="margin-bottom: 4px; color: #1f2937; font-family: monospace;">• {pv}'
                    if pv in sample_values:
                        values = sample_values[pv][:3]  # Show first 3 sample values
                        values_str = ", ".join([f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in values])
                        html += f' <span style="color: #6b7280; font-size: 11px;">(sample: {values_str}...)</span>'
                    html += '</div>'
                if len(pv_names) > user_valves.max_sample_items:
                    html += f'<div style="color: #6b7280; font-style: italic; text-align: center; margin-top: 8px;">• (and {len(pv_names) - user_valves.max_sample_items} more)</div>'
                html += '</div></div>'

        elif context_type in ["Analysis Results", "Visualization Results", "Operation Results"]:
            results = context_info.get("results", {})
            if results:
                html += """
                <div style="margin-top: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">Results:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 12px; max-height: 200px; overflow-y: auto;">
                """
                count = 0
                for key, value in results.items():
                    if count >= user_valves.max_sample_items:
                        break
                    display_key = key.replace('_', ' ').title()
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        html += f'<div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px; border: 1px solid #e2e8f0;"><strong style="color: #1f2937;">{display_key}:</strong> <span style="color: #6b7280; font-style: italic;">(large data structure)</span></div>'
                    else:
                        html += f'<div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px; border: 1px solid #e2e8f0;"><strong style="color: #1f2937;">{display_key}:</strong> <span style="color: #059669;">{value}</span></div>'
                    count += 1
                if len(results) > user_valves.max_sample_items:
                    html += f'<div style="color: #6b7280; font-style: italic; text-align: center; margin-top: 8px;">• (and {len(results) - user_valves.max_sample_items} more)</div>'
                html += '</div></div>'

        elif context_type == "Memory Context":
            memories = context_info.get("memories", [])
            if memories:
                html += """
                <div style="margin-top: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">Memory Entries:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 12px; max-height: 200px; overflow-y: auto;">
                """
                for i, memory in enumerate(memories[:user_valves.max_sample_items]):
                    content = memory.get("content", "N/A")
                    timestamp = memory.get("timestamp", "N/A")
                    html += f'<div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px; border: 1px solid #e2e8f0;"><div style="color: #1f2937; margin-bottom: 4px;">{content}</div><div style="color: #6b7280; font-size: 11px;">@ {timestamp}</div></div>'
                if len(memories) > user_valves.max_sample_items:
                    html += f'<div style="color: #6b7280; font-style: italic; text-align: center; margin-top: 8px;">• (and {len(memories) - user_valves.max_sample_items} more)</div>'
                html += '</div></div>'

        elif context_type == "Conversation Results":
            full_response = context_info.get("full_response", "N/A")
            if full_response:
                html += """
                <div style="margin-top: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">Response:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 12px; max-height: 200px; overflow-y: auto;">
                """
                if len(full_response) > 200:
                    html += f'<div style="color: #1f2937; line-height: 1.4;"><strong>Preview:</strong> {full_response[:200]}...</div>'
                else:
                    html += f'<div style="color: #1f2937; line-height: 1.4;"><strong>Full Response:</strong> {full_response}</div>'
                html += '</div></div>'

        return html


# Action registration - required for OpenWebUI to recognize this as an action button
actions = [
    {
        "id": "als_assistant_agent_context",
        "name": "Agent Context",
        "description": "View current ALS Assistant agent context data and available information",
        "icon_url": "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyUzYuNDggMjIgMTIgMjJTMjIgMTcuNTIgMjIgMTJTMTcuNTIgMiAxMiAyWk0xMiAyMEM3LjU5IDIwIDQgMTYuNDEgNCAxMlM3LjU5IDQgMTIgNFMyMCA3LjU5IDIwIDEyUzE2LjQxIDIwIDEyIDIwWiIgZmlsbD0iY3VycmVudENvbG9yIi8+CjxwYXRoIGQ9Ik0xMiA2VjhNMTIgMTZWMThNMTAgMTJIMTRNOCAxMkg2TTE4IDEySDE2IiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KPC9zdmc+"
    }
]
