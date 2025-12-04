"""
title: ALS Assistant Execution History
author: ALS Assistant Team
version: 0.1.0
required_open_webui_version: 0.5.1
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+
Description: View ALS Assistant Agent execution history for the last response
"""

import json
import logging
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Action:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        show_detailed_steps: bool = Field(
            default=True, description="Show detailed step information"
        )
        show_timestamps: bool = Field(
            default=True, description="Show execution timestamps"
        )
        show_step_results: bool = Field(
            default=False, description="Show detailed step results (may be verbose)"
        )

    def __init__(self):
        self.valves = self.Valves()

    def extract_execution_history_from_messages(self, messages: list):
        """Extract execution history data from assistant messages."""

        # Look through messages in reverse order (most recent first)
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("info"):
                # Check for execution history (OpenWebUI always serializes to JSON)
                if "als_assistant_execution_history_raw" in message["info"]:
                    execution_data = message["info"]["als_assistant_execution_history_raw"]
                    logger.info(f"Found execution history: {len(execution_data)} records")
                    return execution_data

        return None

    def format_execution_history_html(self, execution_history, user_valves) -> str:
        """Format the execution history as HTML for popup display."""
        if not execution_history:
            return '<div style="text-align: center; padding: 40px; color: #6b7280; font-style: italic;">No execution history available.</div>'

        # Header with summary
        html = f"""
        <div style="margin-bottom: 24px; padding: 20px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 18px; font-weight: 600; color: #1f2937;">📊 Execution Overview</div>
                <div style="font-size: 14px; color: #6b7280;">Steps executed in last response</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
                <div style="text-align: center; padding: 12px; background: white; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 24px; font-weight: 700; color: #0369a1;">{len(execution_history)}</div>
                    <div style="font-size: 13px; color: #6b7280; font-weight: 500;">Total Steps</div>
                </div>
        """

        # Calculate step execution time (sum of individual step durations)
        step_duration = 0
        for record in execution_history:
            start_time_str = record.get('start_time')
            end_time_str = record.get('end_time')
            if start_time_str and end_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                    step_duration += (end_time - start_time).total_seconds()
                except:
                    pass  # Skip if datetime parsing fails

        html += f"""
                <div style="text-align: center; padding: 12px; background: white; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 24px; font-weight: 700; color: #059669;">{step_duration:.2f}s</div>
                    <div style="font-size: 13px; color: #6b7280; font-weight: 500;">Step Time</div>
                </div>
            </div>
        </div>
        """

        # Process each step
        for i, record in enumerate(execution_history, 1):
            step = record.get('step', {})
            result = record.get('result', {})

            # Step header with status
            success = result.get('success', False)
            status_emoji = "✅" if success else "❌"
            status_color = "#059669" if success else "#dc2626"
            description = step.get('description', 'Unknown step')

            html += f"""
            <div style="margin-bottom: 24px; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;">
                <div style="background: {'#f0f9ff' if success else '#fef2f2'}; padding: 16px; border-bottom: 1px solid #e2e8f0;">
                    <h3 style="margin: 0; color: #1f2937; font-size: 16px; font-weight: 600;">
                        {status_emoji} Step {i}: {description}
                    </h3>
                    <div style="margin-top: 8px; font-size: 14px; font-weight: 500; color: {status_color};">
                        {'✓ Success' if success else '✗ Failed'}
                    </div>
                </div>
                <div style="padding: 20px;">
            """

            # Summary table
            html += """
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

            # Basic info
            html += f"""
                        <tr>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Node Type</td>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937; font-family: monospace;">{step.get('node_type', 'unknown')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Status</td>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: {status_color}; font-weight: 600;">{status_emoji} {'Success' if success else 'Failed'}</td>
                        </tr>
            """

            # Success criteria
            success_criteria = step.get('success_criteria')
            if success_criteria:
                html += f"""
                        <tr>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Success Criteria</td>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{success_criteria}</td>
                        </tr>
                """

            # Timestamps
            if user_valves.show_timestamps:
                start_time_str = record.get('start_time')
                if start_time_str:
                    try:
                        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                        html += f"""
                        <tr>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Start Time</td>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{start_time.strftime('%H:%M:%S')}</td>
                        </tr>
                        """

                        end_time_str = record.get('end_time')
                        if end_time_str:
                            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                            duration = (end_time - start_time).total_seconds()
                            html += f"""
                        <tr>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: 500; color: #374151;">Duration</td>
                            <td style="padding: 8px 12px; border: 1px solid #cbd5e1; color: #1f2937;">{duration:.2f}s</td>
                        </tr>
                            """
                    except:
                        pass  # Skip if datetime parsing fails

            html += '</tbody></table></div>'

            # Input requirements
            input_requirements = step.get('input_requirements', [])
            if input_requirements and user_valves.show_detailed_steps:
                html += f"""
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">📝 Input Requirements:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 13px; color: #1f2937;">
                        {', '.join(input_requirements)}
                    </div>
                </div>
                """

            # Parameters
            parameters = step.get('parameters', {})
            if parameters and user_valves.show_detailed_steps:
                html += f"""
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">⚙️ Parameters:</h5>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-family: monospace; font-size: 12px; color: #1f2937; max-height: 200px; overflow-y: auto;">
                        <pre style="margin: 0; white-space: pre-wrap;">{json.dumps(parameters, indent=2)}</pre>
                    </div>
                </div>
                """

            # Error details if failed
            if not success:
                error = result.get('error')
                if error:
                    html += f"""
                    <div style="margin-bottom: 16px; padding: 12px; background: #fef2f2; border-radius: 4px; border: 1px solid #fecaca;">
                        <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #dc2626;">🚨 Error Details:</h5>
                        <div style="font-size: 13px; color: #7f1d1d; line-height: 1.4;">
                            <div style="margin-bottom: 4px;"><strong>Message:</strong> {error.get('message', 'No error message')}</div>
                            <div><strong>Severity:</strong> {error.get('severity', 'unknown')}</div>
                        </div>
                    </div>
                    """

            # Result data if requested and available
            if user_valves.show_step_results:
                result_data = result.get('data')
                if result_data:
                    html += f"""
                    <div style="margin-bottom: 16px;">
                        <h5 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #374151;">📊 Result Data:</h5>
                        <div style="background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-family: monospace; font-size: 12px; color: #1f2937; max-height: 300px; overflow-y: auto;">
                            <pre style="margin: 0; white-space: pre-wrap;">{json.dumps(result_data, indent=2, default=str)}</pre>
                        </div>
                    </div>
                    """

            html += '</div></div>'

        return html

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict | None:
        """Display formatted execution history using a popup modal."""
        logger.info(f"User - Name: {__user__['name']}, ID: {__user__['id']} - Requesting ALS Assistant execution history")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Retrieving execution history...", "done": False},
            }
        )

        try:
            # Log debug information about the request
            logger.info(f"Processing execution history request for user {__user__.get('name', 'unknown')}")
            logger.info(f"Message count: {len(body.get('messages', []))}")

            # Extract execution history from the last assistant message
            execution_history = self.extract_execution_history_from_messages(body.get("messages", []))

            if not execution_history:
                logger.info("No execution history found in messages")
                # Show no history popup
                no_history_js = """
                try {
                    // Remove any existing history popup
                    const existingPopup = document.getElementById('execution-history-popup');
                    if (existingPopup) {
                        existingPopup.remove();
                    }

                    // Create overlay
                    const overlay = document.createElement('div');
                    overlay.id = 'execution-history-popup';
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
                            <h2 style="margin: 0; color: #374151; font-size: 20px; font-weight: 600;">📋 ALS Assistant Execution History</h2>
                            <button id="history-close-btn" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px;">Close</button>
                        </div>

                        <div style="text-align: center; padding: 40px 20px; color: #6b7280; font-size: 16px; line-height: 1.6;">
                            <div style="font-size: 48px; margin-bottom: 16px;">⚠️</div>
                            <h3 style="margin: 0 0 16px 0; color: #374151; font-size: 18px; font-weight: 600;">No Execution History Available</h3>
                            <p style="margin: 0; color: #6b7280;">No ALS Assistant execution history found in recent messages. The execution history tracks:</p>
                            <ul style="text-align: left; margin: 20px 0; color: #6b7280; line-height: 1.8;">
                                <li><strong>Step execution details</strong></li>
                                <li><strong>Success/failure status</strong></li>
                                <li><strong>Timing information</strong></li>
                                <li><strong>Parameters and results</strong></li>
                                <li><strong>Error messages</strong></li>
                            </ul>
                            <p style="margin: 0; color: #6b7280; font-style: italic;">Execute an ALS Assistant query to generate execution history.</p>
                        </div>
                    `;

                    // Add event listeners
                    overlay.appendChild(popup);
                    document.body.appendChild(overlay);

                    // Close button
                    document.getElementById('history-close-btn').onclick = function() {
                        overlay.remove();
                    };

                    // Close on overlay click
                    overlay.onclick = function(e) {
                        if (e.target === overlay) {
                            overlay.remove();
                        }
                    };

                } catch (error) {
                    console.error('Error creating execution history popup:', error);
                    alert('Error displaying execution history');
                }
                """

                await __event_call__(
                    {
                        "type": "execute",
                        "data": {"code": no_history_js},
                    }
                )

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "No execution history available", "done": True},
                    }
                )
                return

            logger.info(f"Found execution history: {len(execution_history)} steps")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Formatting execution history...", "done": False},
                }
            )

            # Format the execution history as HTML for the popup
            formatted_history = self.format_execution_history_html(execution_history, user_valves)

            # Create JavaScript to show popup with execution history
            history_js = f"""
            try {{
                // Remove any existing history popup
                const existingPopup = document.getElementById('execution-history-popup');
                if (existingPopup) {{
                    existingPopup.remove();
                }}

                // Create overlay
                const overlay = document.createElement('div');
                overlay.id = 'execution-history-popup';
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
                    max-width: 1200px;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                    color: #333;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                `;

                popup.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 2px solid #e5e7eb;">
                        <h2 style="margin: 0; color: #374151; font-size: 20px; font-weight: 600;">📋 ALS Assistant Execution History</h2>
                        <button id="history-close-btn" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px;">Close</button>
                    </div>

                    <div style="line-height: 1.6;">
                        {formatted_history}
                    </div>
                `;

                // Add event listeners
                overlay.appendChild(popup);
                document.body.appendChild(overlay);

                // Close button
                document.getElementById('history-close-btn').onclick = function() {{
                    overlay.remove();
                }};

                // Close on overlay click
                overlay.onclick = function(e) {{
                    if (e.target === overlay) {{
                        overlay.remove();
                    }}
                }};

            }} catch (error) {{
                console.error('Error creating execution history popup:', error);
                alert('Error displaying execution history: ' + error.message);
            }}
            """

            await __event_call__(
                {
                    "type": "execute",
                    "data": {"code": history_js},
                }
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Execution history displayed", "done": True},
                }
            )

            logger.info(f"User - Name: {__user__['name']}, ID: {__user__['id']} - Execution history popup displayed successfully ({len(execution_history)} steps)")

        except Exception as e:
            logger.error(f"Error processing execution history: {e}")

            error_js = f"""
            try {{
                // Remove any existing history popup
                const existingPopup = document.getElementById('execution-history-popup');
                if (existingPopup) {{
                    existingPopup.remove();
                }}

                // Create overlay
                const overlay = document.createElement('div');
                overlay.id = 'execution-history-popup';
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
                        <h2 style="margin: 0; color: #dc2626; font-size: 20px; font-weight: 600;">❌ Error Processing Execution History</h2>
                        <button id="history-close-btn" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px;">Close</button>
                    </div>

                    <div style="padding: 20px; background: #fef2f2; border-radius: 6px; border: 1px solid #fecaca;">
                        <p style="margin: 0 0 16px 0; color: #dc2626; font-weight: 500;">An error occurred while processing the execution history:</p>
                        <pre style="margin: 0; font-family: monospace; background: #fff; padding: 12px; border-radius: 4px; border: 1px solid #e5e7eb; color: #374151; font-size: 13px; overflow-x: auto;">{str(e)}</pre>
                        <p style="margin: 16px 0 0 0; color: #6b7280; font-size: 14px;">Please check the logs for more details.</p>
                    </div>
                `;

                // Add event listeners
                overlay.appendChild(popup);
                document.body.appendChild(overlay);

                // Close button
                document.getElementById('history-close-btn').onclick = function() {{
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
                alert('Error displaying execution history error: ' + error.message);
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
                    "data": {"description": "Error processing execution history", "done": True},
                }
            )


# Action registration - required for OpenWebUI to recognize this as an action button
actions = [
    {
        "id": "als_assistant_execution_history",
        "name": "Execution History",
        "description": "View ALS Assistant execution history for the last response",
        "icon_url": "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+"
    }
]
