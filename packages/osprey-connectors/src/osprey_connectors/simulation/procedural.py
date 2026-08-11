"""Absolute-time procedural synthesis for channels the machine model omits.

The data-driven :class:`~osprey_connectors.simulation.engine.SimulationEngine` serves only
the channels a project's ``machine.json`` declares. Everything else in the
served namespace — 1,872 of the control-assistant preset's 2,908 channels — has
no model, and the Virtual Accelerator falls back to the shared PV taxonomy for
it: ``EngineSource`` serves ``classify_pv(address).base_value`` plus a keyed
noise draw at :meth:`~osprey_connectors.pv_taxonomy.PVKind.noise_sigma`. This
module is the archiver-side half of that same fallback — the *history* of a
channel whose live value the VA synthesizes from the taxonomy.

Three properties are contracts, not preferences:

* **Values are a pure function of (channel, absolute epoch seconds).** No
  window, sample count or stream position enters the computation, so two
  archiver queries that share a timestamp return the same value at it, and a
  document written into a store now equals what a query synthesizes for that
  timestamp later. The window-relative generator this replaces could not
  satisfy either: its ``np.linspace(0, 1, n)`` axis made every value a function
  of the query's shape, so materializing it produced history that contradicted
  the next read.

* **Baselines come from one source, shared with the VA.** A channel's baseline
  is the value the VA boots it at: the ``machine.json`` seed for the ``:SP`` and
  ``:RB`` channels that carry one (the pyat-coupled and sp-echo partitions), and
  ``classify_pv(...).base_value`` for the rest (the static-noisy partition).
  :func:`baseline_value` is that rule, in one place. This is what makes the
  splice between seeded history and recorded reality seamless: when a recorder
  starts sampling the live machine at T0, the last synthesized sample and the
  first recorded one differ by noise, not by a step an operator would rightly
  read as an event.

* **The excursion is scaled to the channel's declared noise.** Every shape term
  below is expressed in multiples of the channel's own sigma rather than in
  engineering units. A channel the VA serves as a stationary ``base ± 1%`` must
  not have a history that swings 5% and then flatlines the moment recording
  starts — that seam *is* a fabricated anomaly, and a diagnosing agent would be
  right to chase it. :func:`deviation_bound` states the resulting envelope so
  seam assertions can quote it rather than hard-code a number. A
  ``noise_level`` of ``0.0`` is an explicit request for determinism and yields
  the bare baseline, exactly as the VA serves it.

Callers must convert timestamps with
:func:`osprey_connectors.simulation.series.epoch_seconds_array`, the same helper
:meth:`SimulationEngine.synthesize_series` uses. Deriving epoch seconds some
other way (integer-nanosecond division, say) agrees only to within a float ULP,
and the texture terms are continuous functions of that input — so a second
conversion route would silently cost bit-equality between two writers of the
same store.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from osprey_connectors.pv_taxonomy import classify_pv
from osprey_connectors.simulation.series import keyed_normals, pv_key_bytes, wander

__all__ = [
    "DEFAULT_NOISE_LEVEL",
    "KIND_SHAPES",
    "KindShape",
    "baseline_value",
    "deviation_bound",
    "generate_series",
]

_MINUTE = 60.0
_HOUR = 3600.0

DEFAULT_NOISE_LEVEL = 0.01
"""Relative noise fraction assumed when a caller names none.

Matches the Virtual Accelerator's ``EngineSource.DEFAULT_NOISE_LEVEL``, which
is what the live half of a deployed world serves these channels with. The value
is duplicated rather than imported: ``osprey_connectors.simulation`` must stay loadable
without the VA service package. A writer that has the deployed noise level in
hand should pass it explicitly.
"""

# Independent draw streams for this module's three stochastic terms. Subkeyed
# off the channel key so a channel that is *also* engine-served (a project can
# move a channel into machine.json) draws different numbers here than the
# engine's own ":noise"/texture streams, instead of correlating with them.
_DRIFT_SUBKEY = b":procedural:drift"
_CYCLE_SUBKEY = b":procedural:cycle"
_NOISE_SUBKEY = b":procedural:noise"

_TWO_PI = 2.0 * np.pi

# Counters are epoch milliseconds, the convention `_apply_signal_model` pins for
# every keyed draw in the simulation package.
_MS_PER_S = 1000.0

_SEAM_NOISE_SIGMAS = 5.0
"""Noise headroom :func:`deviation_bound` adds on top of the shape envelope.

A seam comparison holds two *independent* draws — the last synthesized sample
and the first recorded one — so the bound has to cover both tails, not one.
Five sigma leaves the assertion meaningful (it is a small fraction of the
excursions the previous generator produced) while making a false failure a
once-in-millions event rather than a flake to be chased.
"""

_SINE = "sine"
_SAWTOOTH = "sawtooth"

# Address suffixes of the channels no value source drives: a setpoint moves
# only when a client writes it, and its readback only echoes that write. The
# subfield is the last colon-separated segment of a manifest address, which is
# how the manifest itself is keyed.
_UNDRIVEN_SUBFIELDS = (":SP", ":RB")

# What ``serving.pvdb`` boots an analog record at with no seed of its own.
_RECORD_TYPE_DEFAULT = 0.0


@dataclass(frozen=True)
class KindShape:
    """One PV kind's texture, in multiples of that channel's noise sigma.

    The two terms play different roles and are both needed. ``drift`` is a
    :func:`~osprey_connectors.simulation.series.wander` stack — quasi-random, band-limited,
    individualized per channel by its key — which is what stops a whole family
    of channels from moving in lockstep. ``cycle`` is a single deterministic
    period carrying the kind's *recognizable* signature: the sawtooth of beam
    current decaying between refills, the ripple of a vacuum gauge, the daily
    breathing of a temperature.

    Attributes:
        drift_sigmas: Wander envelope, in sigmas.
        drift_period_s: Slowest wander period, in seconds.
        cycle_sigmas: Periodic-term amplitude, in sigmas.
        cycle_period_s: Period of that term, in seconds.
        cycle_shape: ``"sine"``, or ``"sawtooth"`` for a sudden rise followed by
            a linear decay across the period (a refill cycle).
    """

    drift_sigmas: float
    drift_period_s: float
    cycle_sigmas: float
    cycle_period_s: float
    cycle_shape: str = _SINE

    @property
    def shape_sigmas(self) -> float:
        """Largest deviation the two shape terms can reach together, in sigmas."""
        return self.drift_sigmas + self.cycle_sigmas


# Per-kind shapes, keyed by PVKind.name. The character of each kind is the one
# the window-relative generator established -- beam current sawtooths between
# refills, pressure ripples fast, temperature breathes daily, voltage is steady
# with a slow swing -- re-expressed on an absolute time axis and rescaled into
# the channel's declared noise envelope (see the module docstring). Periods are
# deliberately not harmonics of each other, so two kinds sampled over the same
# window do not appear to move together.
KIND_SHAPES: Mapping[str, KindShape] = MappingProxyType(
    {
        "beam_current": KindShape(
            drift_sigmas=0.8,
            drift_period_s=18 * _HOUR,
            cycle_sigmas=1.5,
            cycle_period_s=4 * _HOUR,
            cycle_shape=_SAWTOOTH,
        ),
        "current": KindShape(
            drift_sigmas=1.5,
            drift_period_s=12 * _HOUR,
            cycle_sigmas=1.0,
            cycle_period_s=90 * _MINUTE,
        ),
        "voltage": KindShape(
            drift_sigmas=0.5, drift_period_s=24 * _HOUR, cycle_sigmas=1.0, cycle_period_s=3 * _HOUR
        ),
        "power": KindShape(
            drift_sigmas=1.0,
            drift_period_s=9 * _HOUR,
            cycle_sigmas=1.0,
            cycle_period_s=45 * _MINUTE,
        ),
        "pressure": KindShape(
            drift_sigmas=1.5,
            drift_period_s=6 * _HOUR,
            cycle_sigmas=1.0,
            cycle_period_s=20 * _MINUTE,
        ),
        "temperature": KindShape(
            drift_sigmas=1.5,
            drift_period_s=24 * _HOUR,
            cycle_sigmas=0.75,
            cycle_period_s=35 * _MINUTE,
        ),
        "lifetime": KindShape(
            drift_sigmas=1.5,
            drift_period_s=8 * _HOUR,
            cycle_sigmas=1.0,
            cycle_period_s=100 * _MINUTE,
        ),
        "position": KindShape(
            drift_sigmas=1.5,
            drift_period_s=5 * _HOUR,
            cycle_sigmas=0.75,
            cycle_period_s=25 * _MINUTE,
        ),
        "energy": KindShape(
            drift_sigmas=0.5, drift_period_s=24 * _HOUR, cycle_sigmas=0.5, cycle_period_s=2 * _HOUR
        ),
        "default": KindShape(
            drift_sigmas=1.0, drift_period_s=12 * _HOUR, cycle_sigmas=1.0, cycle_period_s=2 * _HOUR
        ),
    }
)


def _shape_for(kind_name: str) -> KindShape:
    """The shape table entry for a kind, falling back to the generic one.

    ``classify_pv`` is the only producer of these names and every one it can
    return has an entry, so the fallback is reached only if a kind is added
    there without one here — in which case a generic texture is a better
    outcome than a ``KeyError`` on an archiver read.
    """
    return KIND_SHAPES.get(kind_name, KIND_SHAPES["default"])


def _keyed_fraction(pv_key: bytes) -> float:
    """A stable per-channel value in ``[0, 1)``, used as a cycle phase.

    Derived from the key by the same digest-then-truncate construction the draw
    primitives use, so it is reproducible across processes (unlike anything
    built on ``hash()``) and independent of the normal draws keyed off the same
    channel.
    """
    word = int.from_bytes(hashlib.sha256(pv_key).digest()[:8], "big")
    return word / 2.0**64


def _cycle_term(pv_key: bytes, times: "np.ndarray", shape: KindShape) -> "np.ndarray":
    """The kind's periodic signature at ``times``, in sigmas (bounded by 1)."""
    phase = _keyed_fraction(pv_key + _CYCLE_SUBKEY)
    turns = times / shape.cycle_period_s + phase
    if shape.cycle_shape == _SAWTOOTH:
        # +1 at the refill, decaying linearly to -1 just before the next one.
        return 1.0 - 2.0 * np.mod(turns, 1.0)
    return np.sin(_TWO_PI * turns)


def baseline_value(pv_name: str, boot_values: Mapping[str, float] | None = None) -> float:
    """The value the Virtual Accelerator settles ``pv_name`` at.

    This is the single anchoring rule the archiver's synthetic history and the
    VA's live values share, and it mirrors the VA's own three sources:

    * a channel the machine model describes is served from the model — by the
      engine on the static-noisy partition, and as the boot seed
      ``serving.pvdb`` gives a ``:SP``/``:RB`` record on the other two;
    * a ``:SP``/``:RB`` channel the model does *not* describe boots at its
      record type's default and stays there, because nothing drives a setpoint
      but a client write. The taxonomy's guess for its name is the wrong answer:
      an ion-pump ``VOLTAGE:SP`` no scenario seeds reads 0 V on the live
      machine, and history claiming 5 kV would be pure invention;
    * everything else is static-noisy telemetry with no model entry, which
      ``EngineSource`` serves from the PV taxonomy.

    A caller holding the machine model (the base seeder does) passes it as
    ``boot_values`` and gets the anchored baseline for all three fidelity
    partitions. A caller without one — a bare mock connector on a project with
    no machine file — cannot tell the second case from the third and gets the
    taxonomy baseline throughout, which is all the information available.

    Args:
        pv_name: Channel name.
        boot_values: Optional ``{address: value}`` map, conventionally
            ``machine.json``'s stored channel values. Entries that are not real
            numbers (a string-valued channel, say) are ignored rather than
            coerced — this generator is numeric.

    Returns:
        The baseline in the channel's engineering units.
    """
    if boot_values is None:
        return classify_pv(pv_name).base_value
    seed = boot_values.get(pv_name)
    if isinstance(seed, (int, float)) and not isinstance(seed, bool):
        return float(seed)
    if pv_name.endswith(_UNDRIVEN_SUBFIELDS):
        return _RECORD_TYPE_DEFAULT
    return classify_pv(pv_name).base_value


def generate_series(
    pv_name: str,
    t_abs_s: "np.ndarray | float",
    *,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    baseline: float | None = None,
) -> "np.ndarray":
    """Synthesize ``pv_name``'s value at absolute epoch seconds ``t_abs_s``.

    Scalars and arrays follow one code path, so a live read at wall-clock *now*
    and an archiver sample at that same instant agree bit-for-bit.

    Args:
        pv_name: Channel name; keys every stochastic term and picks the kind.
        t_abs_s: Absolute epoch seconds — any shape, or a scalar. Pass values
            produced by :func:`osprey_connectors.simulation.series.epoch_seconds_array`
            (see the module docstring on why the conversion route is pinned).
        noise_level: Relative noise fraction, floored per kind by
            :meth:`~osprey_connectors.pv_taxonomy.PVKind.noise_sigma` so a
            legitimately-zero baseline is not dead flat. ``0.0`` disables every
            stochastic and shape term and returns the bare baseline.
        baseline: Baseline override, normally from :func:`baseline_value`.
            ``None`` resolves the taxonomy baseline for the channel.

    Returns:
        A float64 array of values with the shape of ``t_abs_s`` (0-d for a
        scalar input).
    """
    times = np.asarray(t_abs_s, dtype=np.float64)
    kind = classify_pv(pv_name)
    base = kind.base_value if baseline is None else float(baseline)
    sigma = kind.noise_sigma(base, noise_level)
    if sigma <= 0.0:
        # An explicit request for determinism: the VA serves the bare boot
        # value at noise_level 0, and so must its history.
        return np.full(times.shape, base, dtype=np.float64)

    shape = _shape_for(kind.name)
    pv_key = pv_key_bytes(pv_name)
    # Flatten before drawing and restore the caller's shape after. The draw
    # primitives wrap uint64 arithmetic silently on arrays but warn on numpy
    # scalars, so a 0-d input would produce the right value with a spurious
    # overflow warning attached -- and the scalar path is the live-read path.
    flat = times.reshape(-1)
    values = np.full(flat.shape, base, dtype=np.float64)
    values = values + wander(
        pv_key + _DRIFT_SUBKEY, flat, shape.drift_sigmas * sigma, shape.drift_period_s
    )
    values = values + shape.cycle_sigmas * sigma * _cycle_term(pv_key, flat, shape)
    values = values + sigma * keyed_normals(pv_key + _NOISE_SUBKEY, flat * _MS_PER_S)
    return np.asarray(values.reshape(times.shape))


def deviation_bound(
    pv_name: str,
    *,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    baseline: float | None = None,
    noise_sigmas: float = _SEAM_NOISE_SIGMAS,
) -> float:
    """How far from its baseline this generator can plausibly take a channel.

    The intended reader is a seam assertion: seeded history ends at T0 and
    recorded reality begins there, and the step between them must be
    attributable to noise rather than to an event. Quoting this function keeps
    that threshold derived from the same shape table the values come from, so
    retuning a kind cannot silently invalidate the assertion.

    The bound is structural, not statistical, for the shape terms —
    :func:`~osprey_connectors.simulation.series.wander`'s weights are positive and sum to
    one, and the cycle term is a bounded sinusoid or sawtooth — and covers
    ``noise_sigmas`` of the unbounded Gaussian term on top.

    Args:
        pv_name: Channel name.
        noise_level: The relative noise the series was (or will be) generated
            with. Must match, or the bound describes a different series.
        baseline: Baseline override, as for :func:`generate_series`.
        noise_sigmas: Gaussian headroom; the default covers the two independent
            draws either side of a seam.

    Returns:
        The bound in the channel's engineering units; ``0.0`` when
        ``noise_level`` disables synthesis entirely.
    """
    kind = classify_pv(pv_name)
    base = kind.base_value if baseline is None else float(baseline)
    sigma = kind.noise_sigma(base, noise_level)
    if sigma <= 0.0:
        return 0.0
    return (_shape_for(kind.name).shape_sigmas + noise_sigmas) * sigma
