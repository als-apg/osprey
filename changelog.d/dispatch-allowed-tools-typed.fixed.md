A trigger's `action.allowed_tools` that is not a list of tool names is refused
when the triggers file loads, naming the trigger. Write one tool as `[get_pv]`;
the bare `get_pv` form is gone. A blank `allowed_tools:` means no tools. Before,
such a trigger loaded and every fire was refused by the worker.
