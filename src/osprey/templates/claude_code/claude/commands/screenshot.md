# Screenshot — Screen Vision Workflow

Capture and analyze what operators see on control room displays.

## Tools

| Tool | Purpose |
|------|---------|
| `screenshot_capture` | Capture full screen, a specific display, region, or window |
| `list_windows` | List visible windows with IDs, positions, and sizes |
| `manage_window` | Bring a window to front, move, or resize it |

## Typical Workflow

1. **List windows** to see what's on screen: `list_windows()`
2. **Bring target to front** if needed: `manage_window(app="Phoebus", action="bring_to_front")`
3. **Capture**: `screenshot_capture(mode="window", target="Phoebus")`
4. **View** the screenshot using the `Read` tool on the returned file path
5. **Analyze** what you see and report to the operator

## Tips

- Window IDs (WIDs) are ephemeral — always call `list_windows` first to get current IDs
- Use `bring_to_front` before capturing a window to ensure it's not obscured
- For multi-monitor setups, use `mode="display"` with `target="1"`, `"2"`, etc.
- Region captures use `target="x,y,width,height"` format
- Screenshots are saved to `osprey-workspace/screenshots/` as PNG files
- The `Read` tool can view PNG files directly (multimodal)
