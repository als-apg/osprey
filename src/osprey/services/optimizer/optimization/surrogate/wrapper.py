"""Unified BoTorch wrapper for sklearn models with uncertainty.

Wraps any sklearn estimator that provides variance (ensemble models via
``estimators_``, or GP via ``predict(return_std=True)``) into a BoTorch
``Model`` so it can be used with standard acquisition functions.

The posterior is made differentiable via a custom ``torch.autograd.Function``
that estimates gradients using finite differences, allowing BoTorch's
gradient-based acquisition optimizers (L-BFGS-B) to work.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from sklearn.pipeline import Pipeline
from torch import Tensor

_FD_EPS = 1e-5


def _unwrap_pipeline(estimator: Any) -> Any:
    """If *estimator* is an sklearn Pipeline, return the final step."""
    if isinstance(estimator, Pipeline):
        return estimator[-1]
    return estimator


def _predict_np(inner: Any, variance_method: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run sklearn prediction and return (mean, variance) as numpy arrays."""
    if variance_method == "ensemble":
        predictions = np.array([tree.predict(X) for tree in inner.estimators_])
        return predictions.mean(axis=0), predictions.var(axis=0)
    mean, std = inner.predict(X, return_std=True)
    return np.asarray(mean), np.asarray(std) ** 2


class _SklearnPredictFunction(torch.autograd.Function):
    """Bridge sklearn (non-differentiable) into PyTorch's autograd graph via finite-difference gradients."""

    @staticmethod
    def forward(
        ctx: Any,
        X: Tensor,
        inner: Any,
        variance_method: str,
    ) -> tuple[Tensor, Tensor]:
        ctx.inner = inner
        ctx.variance_method = variance_method
        ctx.save_for_backward(X)

        X_np = X.detach().cpu().numpy().reshape(-1, X.shape[-1])
        mean_np, var_np = _predict_np(inner, variance_method, X_np)

        mean_t = torch.as_tensor(mean_np, dtype=X.dtype, device=X.device).reshape(X.shape[:-1])
        var_t = torch.as_tensor(var_np, dtype=X.dtype, device=X.device).reshape(X.shape[:-1])
        return mean_t, var_t

    @staticmethod
    def backward(ctx: Any, grad_mean: Tensor, grad_var: Tensor) -> tuple[Tensor | None, None, None]:
        (X,) = ctx.saved_tensors
        inner = ctx.inner
        variance_method = ctx.variance_method

        X_np = X.detach().cpu().numpy().reshape(-1, X.shape[-1])
        n, d = X_np.shape
        grad_X = torch.zeros_like(X)
        flat_grad = grad_X.reshape(n, d)
        flat_grad_mean = grad_mean.reshape(n)
        flat_grad_var = grad_var.reshape(n)

        for i in range(d):
            X_plus = X_np.copy()
            X_plus[:, i] += _FD_EPS
            X_minus = X_np.copy()
            X_minus[:, i] -= _FD_EPS

            m_plus, v_plus = _predict_np(inner, variance_method, X_plus)
            m_minus, v_minus = _predict_np(inner, variance_method, X_minus)

            dm = torch.as_tensor(
                (m_plus - m_minus) / (2.0 * _FD_EPS), dtype=X.dtype, device=X.device
            )
            dv = torch.as_tensor(
                (v_plus - v_minus) / (2.0 * _FD_EPS), dtype=X.dtype, device=X.device
            )

            flat_grad[:, i] = flat_grad_mean * dm + flat_grad_var * dv

        return grad_X, None, None


class SklearnBoTorchModel(Model):
    """BoTorch Model wrapping an sklearn estimator with uncertainty.

    Auto-detects two variance methods:
    1. Ensemble models (``estimators_`` attribute) — variance from tree predictions.
    2. GP-style models (``predict(return_std=True)``) — native std.

    The posterior is differentiable via finite-difference gradients so that
    BoTorch's gradient-based acquisition optimizers work correctly.
    """

    def __init__(self, sklearn_model: Any) -> None:
        super().__init__()
        self._sklearn_model = sklearn_model
        self._inner = _unwrap_pipeline(sklearn_model)

        if hasattr(self._inner, "estimators_"):
            self._variance_method = "ensemble"
        elif self._check_return_std():
            self._variance_method = "return_std"
        else:
            raise ValueError(
                f"Model {type(self._inner).__name__} does not support uncertainty estimation. "
                "Expected either `estimators_` attribute (ensemble) or `predict(return_std=True)`."
            )

    def _check_return_std(self) -> bool:
        """Check if the model's predict accepts return_std."""
        sig = inspect.signature(self._inner.predict)
        return "return_std" in sig.parameters

    @property
    def sklearn_model(self) -> Any:
        """The wrapped sklearn estimator, as passed to the constructor."""
        return self._sklearn_model

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(  # type: ignore[override]
        self,
        X: Tensor,
        observation_noise: bool = False,
        posterior_transform: Any | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """Compute posterior mean and variance from the sklearn model.

        Uses a custom autograd function so that gradients flow back to X
        via finite differences, enabling gradient-based acquisition optimization.
        The sklearn variance is already the predictive spread, so there is no
        separate observation-noise term to add.
        """
        del observation_noise
        mean_t, var_t = _SklearnPredictFunction.apply(  # type: ignore[no-untyped-call]
            X, self._inner, self._variance_method
        )
        var_t = var_t.clamp(min=1e-12)

        covar = torch.diag_embed(var_t)
        mvn = MultivariateNormal(mean_t, covar)
        posterior = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)  # type: ignore[no-any-return]
        return posterior
