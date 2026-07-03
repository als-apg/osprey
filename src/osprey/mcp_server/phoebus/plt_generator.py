"""PLT file generator for Phoebus Data Browser.

Generates Data Browser XML (``.plt``) files from ``PlotConfig`` objects.

Migrated from the retired ``phoebus_launch`` server
(``als-profiles/mcp_servers/phoebus/plt_generator.py``) as the reincarnated
capability inside ``phoebus_open_databrowser`` — see
``.claude/plans/phoebus-als-integration/ADDENDUM-databrowser-reincarnation.md``,
Task 0.5. The ``.plt`` XML serializer and archiver ``<archive>`` binding are
kept verbatim; two things were dropped on migration:

* ``create_launch_uri`` (the ``myapp://`` custom-scheme launcher) — nothing in
  the live-open flow consumes it; the bridge ``POST /open`` opens the
  generated ``.plt`` directly.
* The hardcoded ALS archiver default. ``archiver_url`` now defaults to
  ``None`` (no facility baked into framework code); the caller resolves it
  from config/env (``phoebus.archiver_url`` — each facility profile supplies
  its own archiver appliance URL). When no archiver URL is supplied, PVs are
  emitted without an ``<archive>`` binding (live-only data, no historical
  backfill) rather than silently pointing at a facility that may not exist.
"""

from __future__ import annotations

import html
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from .models import PlotConfig

logger = logging.getLogger(__name__)


def sanitize_xml_text(text: str) -> str:
    """Sanitize text for safe inclusion in XML content.

    Removes control characters, replaces problematic Unicode, and applies
    HTML escaping for XML special characters.
    """
    if not text:
        return ""

    # Replace common problematic characters
    text = text.replace("°", "deg")
    text = text.replace("μ", "u")
    text = text.replace("²", "2")
    text = text.replace("³", "3")

    # Remove control characters (ASCII 0-31 except tab, newline, carriage return)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize temperature patterns
    text = re.sub(r"\b0C\b", "degC", text)
    text = re.sub(r"\bdeg\s*C\b", "degC", text)

    # Apply HTML escaping for XML special characters
    text = html.escape(text)

    return text


def create_plt_from_config(
    plot_config: PlotConfig,
    workspace_dir: Path,
    archiver_url: str | None = None,
) -> str:
    """Generate a PLT file from a PlotConfig and return the file path.

    Args:
        plot_config: Plot configuration to render.
        workspace_dir: Directory to write the .plt file into.
        archiver_url: Archiver appliance URL for PV data retrieval. Facility-
            neutral: no default is baked in here — pass ``None`` (the
            default) to omit the ``<archive>`` binding entirely, or supply a
            facility's archiver URL (e.g. resolved from
            ``phoebus.archiver_url`` in config.yml).

    Returns:
        Absolute path to the created .plt file.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".plt",
        prefix=f"{timestamp}_phoebus_databrowser_",
        dir=workspace_dir,
    )

    # Format time range
    if plot_config.time_range:
        if isinstance(plot_config.time_range.start, datetime):
            start_str = plot_config.time_range.start.strftime("%Y-%m-%d %H:%M:%S.000")
        else:
            start_str = str(plot_config.time_range.start)

        if isinstance(plot_config.time_range.end, datetime):
            end_str = plot_config.time_range.end.strftime("%Y-%m-%d %H:%M:%S.999")
        else:
            end_str = str(plot_config.time_range.end)
    else:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S.000")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S.999")

    bg = plot_config.background_color
    fg = plot_config.foreground_color

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<databrowser>
  <title>{sanitize_xml_text(plot_config.title)}</title>
  <show_toolbar>{str(plot_config.show_toolbar).lower()}</show_toolbar>
  <show_legend>{str(plot_config.show_legend).lower()}</show_legend>
  <update_period>{plot_config.update_period}</update_period>
  <scroll_step>5</scroll_step>
  <scroll>{str(plot_config.scroll).lower()}</scroll>
  <start>{sanitize_xml_text(start_str)}</start>
  <end>{sanitize_xml_text(end_str)}</end>
  <archive_rescale>STAGGER</archive_rescale>
  <foreground>
    <red>{fg[0]}</red>
    <green>{fg[1]}</green>
    <blue>{fg[2]}</blue>
  </foreground>
  <background>
    <red>{bg[0]}</red>
    <green>{bg[1]}</green>
    <blue>{bg[2]}</blue>
  </background>
  <title_font>Liberation Sans|16|1</title_font>
  <label_font>Liberation Sans|12|0</label_font>
  <scale_font>Liberation Sans|10|0</scale_font>
  <legend_font>Liberation Sans|12|0</legend_font>
  <axes>
    <axis>
      <visible>true</visible>
      <name>{sanitize_xml_text(plot_config.axis_name)}</name>
      <use_axis_name>true</use_axis_name>
      <use_trace_names>true</use_trace_names>
      <right>false</right>
      <color>
        <red>{fg[0]}</red>
        <green>{fg[1]}</green>
        <blue>{fg[2]}</blue>
      </color>"""

    if plot_config.axis_min is not None and plot_config.axis_max is not None:
        xml_content += f"""
      <min>{plot_config.axis_min}</min>
      <max>{plot_config.axis_max}</max>"""
    else:
        xml_content += """
      <min>0.0</min>
      <max>100.0</max>"""

    xml_content += f"""
      <grid>{str(plot_config.show_grid).lower()}</grid>
      <autoscale>{str(plot_config.auto_scale).lower()}</autoscale>
      <log_scale>{str(plot_config.log_scale).lower()}</log_scale>
    </axis>
  </axes>
  <pvlist>"""

    for pv in plot_config.pvs:
        display_name = pv.display_name or pv.name
        color = pv.color

        archive_block = ""
        if archiver_url:
            archive_block = f"""
      <archive>
        <name>archappl</name>
        <url>{archiver_url}</url>
        <key>1</key>
      </archive>"""

        xml_content += f"""
    <pv>
      <display_name>{sanitize_xml_text(display_name)}</display_name>
      <visible>true</visible>
      <name>{sanitize_xml_text(pv.name)}</name>
      <axis>{pv.axis}</axis>
      <color>
        <red>{color[0]}</red>
        <green>{color[1]}</green>
        <blue>{color[2]}</blue>
      </color>
      <trace_type>{pv.trace_type.value}</trace_type>
      <linewidth>{pv.line_width}</linewidth>
      <line_style>{pv.line_style.value}</line_style>
      <point_type>{pv.point_type.value}</point_type>
      <point_size>{pv.point_size}</point_size>
      <waveform_index>0</waveform_index>
      <period>0.0</period>
      <ring_size>5000</ring_size>
      <request>OPTIMIZED</request>{archive_block}
    </pv>"""

    xml_content += """
  </pvlist>"""

    if plot_config.annotations:
        xml_content += """
  <annotations>"""

        for annotation in plot_config.annotations:
            if isinstance(annotation.time_position, datetime):
                time_str = annotation.time_position.strftime("%Y-%m-%d %H:%M:%S.000")
            else:
                time_str = str(annotation.time_position)

            xml_content += f"""
    <annotation>
      <pv>{annotation.pv_index}</pv>
      <time>{sanitize_xml_text(time_str)}</time>
      <value>{annotation.value_position}</value>
      <offset>
        <x>{annotation.offset_x}</x>
        <y>{annotation.offset_y}</y>
      </offset>
      <text>{sanitize_xml_text(annotation.text)}</text>
    </annotation>"""

        xml_content += """
  </annotations>"""

    xml_content += """
</databrowser>"""

    with os.fdopen(temp_fd, "w") as f:
        f.write(xml_content)

    logger.info("Created PLT file: %s", temp_path)
    return temp_path
