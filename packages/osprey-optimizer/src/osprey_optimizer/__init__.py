"""Optimizer service: Xopt-driven tuning jobs behind a FastAPI job API.

Ported from tuning_scripts at commit 1bac27a1.

The base install carries the contracts: ``models`` and ``environments`` load
with pydantic, PyYAML and osprey-connectors only, and the OSPREY framework
imports them. The search stack (xopt, torch, botorch, polars, redis, fastapi)
is the ``service`` extra, installed only in the optimizer's image.

Sub-packages:
- models: request/response models, environment protocols, enums
- environments: environment files and the connector-driven environment
- api: FastAPI app, routers, job tracking and control (needs ``service``)
- optimization: phase runner, evaluator, Xopt backend, custom generators
- services: execution and orchestration inside the job worker
"""
