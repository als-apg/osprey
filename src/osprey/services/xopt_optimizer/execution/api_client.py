"""HTTP client for the tuning_scripts optimization API.

Provides an async interface (aiohttp) for submitting Xopt YAML configurations,
polling for results, and controlling optimization jobs via the tuning_scripts
REST API.
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    import aiohttp
except ImportError as e:
    raise ImportError(
        "aiohttp is required for TuningScriptsClient. Install it with: pip install aiohttp"
    ) from e

from osprey.utils.config import get_full_configuration
from osprey.utils.logger import get_logger

logger = get_logger("xopt_optimizer")

# Terminal statuses that indicate the job is done
_TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})


class TuningScriptsAPIError(Exception):
    """Error communicating with the tuning_scripts API."""

    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None):
        self.status_code = status_code
        self.detail = detail
        full_msg = message
        if status_code:
            full_msg = f"[HTTP {status_code}] {message}"
        if detail:
            full_msg = f"{full_msg} — {detail}"
        super().__init__(full_msg)


class TuningScriptsClient:
    """Async HTTP client for the tuning_scripts optimization API.

    Configuration is read from the ``xopt_optimizer.api`` section of
    Osprey's config.yml::

        xopt_optimizer:
          api:
            base_url: "http://tuning-api:8001"
            poll_interval_seconds: 5.0
            timeout_seconds: 3600

    The client can also be instantiated directly with explicit parameters.
    """

    def __init__(
        self,
        base_url: str | None = None,
        poll_interval_seconds: float | None = None,
        timeout_seconds: float | None = None,
    ):
        api_config = self._load_api_config()

        self.base_url = (base_url or api_config.get("base_url", "http://localhost:8001")).rstrip(
            "/"
        )
        self.poll_interval = poll_interval_seconds or api_config.get("poll_interval_seconds", 5.0)
        self.timeout = timeout_seconds or api_config.get("timeout_seconds", 3600)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Check API health.

        Returns:
            Health status dict (e.g. ``{"status": "ok"}``).

        Raises:
            TuningScriptsAPIError: If the API is unreachable or unhealthy.
        """
        return await self._get("/health")

    async def list_environments(self) -> list[dict[str, Any]]:
        """List available optimization environments.

        Returns:
            List of environment dicts with ``name``, ``display_name``,
            ``description``, ``valid``, and ``source`` fields.

        Raises:
            TuningScriptsAPIError: If the API is unreachable.
        """
        return await self._get("/environments")

    async def get_environment_details(self, name: str) -> dict[str, Any]:
        """Get detailed info for a specific environment (variables, objectives).

        Returns:
            Environment details dict with ``available_objectives``,
            ``default_objective``, ``observables_metadata``, etc.

        Raises:
            TuningScriptsAPIError: If the API is unreachable or env not found.
        """
        return await self._get(f"/environments/{name}")

    async def submit_config(self, config: dict[str, Any]) -> str:
        """Submit an OptimizationConfig dict to start an optimization.

        Args:
            config: Optimization config dict (environment_name, algorithm, etc.).

        Returns:
            The ``job_id`` of the submitted job.

        Raises:
            TuningScriptsAPIError: On submission failure.
        """
        data = await self._post("/optimization/start", json=config)
        return data["job_id"]

    async def submit_yaml(self, yaml_config: str, n_iterations: int | None = None) -> str:
        """Submit an Xopt YAML configuration to start an optimization.

        Args:
            yaml_config: Raw Xopt YAML string.
            n_iterations: Optional iteration count override.

        Returns:
            The ``job_id`` of the submitted job.

        Raises:
            TuningScriptsAPIError: On submission failure.
        """
        payload: dict[str, Any] = {"yaml_config": yaml_config}
        if n_iterations is not None:
            payload["n_iterations"] = n_iterations

        data = await self._post("/optimization/start-yaml", json=payload)
        return data["job_id"]

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """Get lightweight job status.

        Returns:
            Dict with ``job_id``, ``status``, ``message``, ``results_path``.
        """
        return await self._get(f"/optimization/{job_id}", params={"detail": "summary"})

    async def get_full_state(self, job_id: str) -> dict[str, Any]:
        """Get full optimization state including data.

        Returns:
            Dict with full state: data records, logs, variable names, etc.
        """
        return await self._get(f"/optimization/{job_id}", params={"detail": "full"})

    async def poll_until_complete(self, job_id: str) -> dict[str, Any]:
        """Poll job status until a terminal state is reached, then fetch full state.

        Args:
            job_id: The job to poll.

        Returns:
            Full optimization state from ``get_full_state``.

        Raises:
            TuningScriptsAPIError: On timeout or if the job ends in error.
        """
        elapsed = 0.0
        while elapsed < self.timeout:
            status_resp = await self.get_status(job_id)
            job_status = status_resp.get("status", "unknown")

            if job_status in _TERMINAL_STATUSES:
                full_state = await self.get_full_state(job_id)
                if job_status == "error":
                    error_msg = full_state.get("message") or "Optimization failed"
                    raise TuningScriptsAPIError(
                        f"Optimization job {job_id} failed: {error_msg}",
                        detail=error_msg,
                    )
                return full_state

            logger.info(f"Job {job_id} status: {job_status} (elapsed: {elapsed:.0f}s)")
            await asyncio.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise TuningScriptsAPIError(f"Timeout waiting for job {job_id} after {self.timeout}s")

    async def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a running optimization job."""
        return await self._post(f"/optimization/{job_id}/cancel")

    async def pause(self, job_id: str) -> dict[str, Any]:
        """Pause a running optimization job."""
        return await self._post(f"/optimization/{job_id}/pause")

    async def resume(self, job_id: str) -> dict[str, Any]:
        """Resume a paused optimization job."""
        return await self._post(f"/optimization/{job_id}/resume")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    return await self._handle_response(resp)
        except aiohttp.ClientError as e:
            raise TuningScriptsAPIError(f"Connection error: {e}") from e

    async def _post(self, path: str, json: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=json) as resp:
                    return await self._handle_response(resp)
        except aiohttp.ClientError as e:
            raise TuningScriptsAPIError(f"Connection error: {e}") from e

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        if resp.status >= 400:
            try:
                body = await resp.json()
                detail = body.get("detail", str(body))
            except Exception:
                detail = await resp.text()
            raise TuningScriptsAPIError(
                f"API request failed: {resp.method} {resp.url}",
                status_code=resp.status,
                detail=detail,
            )
        return await resp.json()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @staticmethod
    def _load_api_config() -> dict[str, Any]:
        try:
            config = get_full_configuration()
            return config.get("xopt_optimizer", {}).get("api", {})
        except Exception:
            return {}
