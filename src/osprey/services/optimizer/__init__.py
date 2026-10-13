"""Optimizer service: Xopt-driven tuning jobs behind a FastAPI job API.

Ported from tuning_scripts at commit 1bac27a1.
The heavy search stack (xopt, torch, botorch) is the ``optimizer`` extra and is
imported only below this package, never by ``osprey`` itself.

Sub-packages:
- models: request/response models, environment protocols, enums
- api: FastAPI app, routers, job tracking and control
- optimization: phase runner, evaluator, Xopt backend, custom generators
- services: execution and orchestration inside the job worker
"""
