"""The engines that judge one prompt for a watchdog.

An engine answers ``async judge(defn, nodes, prompt, *, deadline) -> Verdict``
and never raises on a failed judgement: a judge that could not decide answers an
``unavailable`` verdict. Each engine builds its own view of the session from the
nodes, at its own budget. Engines are imported on demand, so loading one does
not cost the other's dependencies.
"""
