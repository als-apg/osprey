A trigger's `action.surface_tools` that is not a list of tool names is refused
when the triggers file loads, as `action.surface_prompt` already is. Before, it
loaded, and every fire of that trigger was refused by the worker.
