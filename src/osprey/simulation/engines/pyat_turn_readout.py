"""One monitor's turn-by-turn arrays, read out from a tracked shot.

A shot's tracking gives, at each monitor and for each turn, the share of the
bunch still alive there and the survivors' centroid. :func:`monitor_turn_arrays`
turns those into the three arrays the monitor serves -- ``x``, ``y`` and
``sum`` -- each ``trigger_sample + turns`` samples long: the samples before the
trigger see only the stored beam, and injected turn 1 sits at index
``trigger_sample``.

The sum is the injected charge times its alive share times the sum per unit
charge, plus, while the RF is on, the stored current times the sum per unit
stored current; it never falls below the floor. Positions are the sum-weighted
mix of the stored beam, which sits at zero, and the injected bunch, plus
position noise. Where the beam sum is at or below the floor there is no beam
to read: the positions are floor noise, much larger than a beam signal. With
the pilot tone on, a spike lands on every sample whose index is a multiple of
the tone's period, in all three arrays.

Every random number is a ``keyed_normals`` draw keyed on the array's record
address, the shot's value and the sample index, never on Python's ``hash``
or the clock, so a shot reads the same in every process. Everything here
takes plain numbers and arrays; nothing reads a model or a lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "NOISE_SUBKEY",
    "TONE_SPIKE_M",
    "TONE_SUBKEY",
    "TurnReadout",
    "beam_sum",
    "monitor_turn_arrays",
    "sample_draws",
    "turn_positions",
    "turn_sum",
]

#: The subkey of an array's noise stream, between the address and the shot.
NOISE_SUBKEY = b":turn_noise:"

#: The subkey of an array's tone spikes, between the address and the shot.
TONE_SUBKEY = b":tone:"

#: The sigma of a tone's position spike per unit of tone amplitude, in metres.
TONE_SPIKE_M = 2.0e-3


@dataclass(frozen=True)
class TurnReadout:
    """The monitor calibration and electronics a readout applies.

    Attributes:
        sum_per_charge: The sum one unit of injected charge reads.
        sum_per_stored: The sum one unit of stored current reads.
        sum_floor: The sum a monitor reads with no beam; no sample is lower.
        position_noise: The sigma of a position sample where beam is seen, in metres.
        sum_noise: The sigma of a sum sample, in the sum's unit.
        floor_noise: The sigma of a position sample at the floor, in metres.
        tone_period: The samples between two of the tone's spikes.
        tone_amplitude: The tone's strength: a position spike's sigma is
            ``tone_amplitude * TONE_SPIKE_M`` metres, a sum spike's height is
            ``tone_amplitude * sum_per_charge`` times the magnitude of a draw.
    """

    sum_per_charge: float
    sum_per_stored: float
    sum_floor: float
    position_noise: float
    sum_noise: float
    floor_noise: float
    tone_period: int
    tone_amplitude: float


def sample_draws(address: str, subkey: bytes, shot: int, samples: int) -> np.ndarray:
    """The standard normals of one array's ``subkey`` stream for ``shot``, one per sample.

    Args:
        address: The array's record address.
        subkey: The stream's subkey.
        shot: The shot's value.
        samples: The length of the array.

    Returns:
        ``keyed_normals(channel_key_bytes(address) + subkey + <shot>, 0..samples-1)``.
    """
    from osprey_connectors.simulation.series import channel_key_bytes, keyed_normals

    key = channel_key_bytes(address) + subkey + str(int(shot)).encode("ascii")
    return np.asarray(keyed_normals(key, np.arange(samples)), dtype=np.float64)


def _tone_mask(samples: int, period: int) -> np.ndarray:
    """True on every sample whose index is a multiple of ``period``."""
    if period <= 0:
        raise ValueError(f"the tone period must be a positive number of samples, not {period}")
    mask: np.ndarray = np.arange(samples) % int(period) == 0
    return mask


def beam_sum(
    alive: Any,
    *,
    trigger_sample: int,
    charge: float,
    stored_current: float,
    rf_on: bool,
    readout: TurnReadout,
) -> tuple[np.ndarray, np.ndarray]:
    """The injected and stored parts of the sum, sample by sample, before floor and noise.

    Args:
        alive: The share of the bunch alive at the monitor, one per turn; NaN
            or a negative share counts as none.
        trigger_sample: The samples before injected turn 1.
        charge: The shot's charge.
        stored_current: The stored current held before the shot.
        rf_on: Whether the RF holds a stored beam.
        readout: The calibration.

    Returns:
        ``(injected, stored)``, each ``trigger_sample + turns`` long.
    """
    share = np.nan_to_num(np.asarray(alive, dtype=np.float64), nan=0.0)
    share = np.maximum(share, 0.0)
    injected = np.concatenate([np.zeros(trigger_sample), charge * share * readout.sum_per_charge])
    stored_level = stored_current * readout.sum_per_stored if rf_on else 0.0
    stored = np.full(injected.shape, max(float(stored_level), 0.0))
    return injected, stored


def turn_sum(
    physics: np.ndarray,
    *,
    address: str,
    shot: int,
    tone_on: bool,
    readout: TurnReadout,
) -> np.ndarray:
    """The sum array: the physics sum plus noise, floored, then the tone's spikes.

    Args:
        physics: The beam sum per sample, injected plus stored.
        address: The sum record's address, which keys its noise.
        shot: The shot's value.
        tone_on: Whether the pilot tone is on.
        readout: The calibration.

    Returns:
        A float64 array the length of ``physics``, no sample below the floor.
    """
    samples = physics.shape[0]
    reading = np.asarray(physics, dtype=np.float64)
    if readout.sum_noise != 0.0:
        reading = reading + readout.sum_noise * sample_draws(address, NOISE_SUBKEY, shot, samples)
    reading = np.maximum(reading, readout.sum_floor)
    if tone_on and readout.tone_amplitude != 0.0:
        height = readout.tone_amplitude * readout.sum_per_charge
        spikes = height * np.abs(sample_draws(address, TONE_SUBKEY, shot, samples))
        reading = reading + np.where(_tone_mask(samples, readout.tone_period), spikes, 0.0)
    return reading


def turn_positions(
    centroid: Any,
    injected: np.ndarray,
    stored: np.ndarray,
    *,
    trigger_sample: int,
    address: str,
    shot: int,
    tone_on: bool,
    readout: TurnReadout,
) -> np.ndarray:
    """One axis's position array.

    Where the beam sum exceeds the floor, the position is the sum-weighted
    mix of the stored beam at zero and the injected centroid, plus position
    noise; elsewhere it is floor noise. Both noises read one stream of the
    address. With the tone on, its spikes are added on top.

    Args:
        centroid: The injected survivors' centroid, one per turn; NaN where
            none is alive.
        injected: The injected part of the sum, per sample.
        stored: The stored part of the sum, per sample.
        trigger_sample: The samples before injected turn 1.
        address: The position record's address, which keys its noise.
        shot: The shot's value.
        tone_on: Whether the pilot tone is on.
        readout: The calibration.

    Returns:
        A float64 array the length of ``injected``, with no NaN.
    """
    samples = injected.shape[0]
    centre = np.concatenate([np.zeros(trigger_sample), np.asarray(centroid, dtype=np.float64)])
    seen = ~np.isnan(centre) & (injected > 0.0)
    weight = np.where(seen, injected, 0.0)
    total = weight + stored
    beam = total > readout.sum_floor
    safe_total = np.where(total > 0.0, total, 1.0)
    mixed = weight * np.where(seen, centre, 0.0) / safe_total
    position = np.where(beam, mixed, 0.0)
    if readout.position_noise != 0.0 or readout.floor_noise != 0.0:
        draws = sample_draws(address, NOISE_SUBKEY, shot, samples)
        sigma = np.where(beam, readout.position_noise, readout.floor_noise)
        position = position + sigma * draws
    if tone_on and readout.tone_amplitude != 0.0:
        spikes = (
            readout.tone_amplitude
            * TONE_SPIKE_M
            * sample_draws(address, TONE_SUBKEY, shot, samples)
        )
        position = position + np.where(_tone_mask(samples, readout.tone_period), spikes, 0.0)
    return np.asarray(position, dtype=np.float64)


def monitor_turn_arrays(
    *,
    alive: Any,
    centroid_x: Any,
    centroid_y: Any,
    trigger_sample: int,
    charge: float,
    stored_current: float,
    rf_on: bool,
    tone_on: bool,
    addresses: tuple[str, str, str],
    shot: int,
    readout: TurnReadout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One monitor's ``x``, ``y`` and ``sum`` arrays for one shot.

    Args:
        alive: The share of the bunch alive at the monitor, one per turn.
        centroid_x: The survivors' horizontal centroid, one per turn, NaN where none.
        centroid_y: The survivors' vertical centroid, turn for turn.
        trigger_sample: The samples before injected turn 1; zero or more.
        charge: The shot's charge.
        stored_current: The stored current held before the shot.
        rf_on: Whether the RF is on; off, there is no stored beam.
        tone_on: Whether the pilot tone is on.
        addresses: The ``x``, ``y`` and ``sum`` record addresses, which key
            each array's noise.
        shot: The shot's value, which keys every array's noise.
        readout: The calibration and electronics.

    Returns:
        ``(x, y, sum)``, float64 arrays of ``trigger_sample + turns`` samples, no NaN.

    Raises:
        ValueError: The per-turn inputs differ in length, or the trigger sample
            is negative.
    """
    trigger_sample = int(trigger_sample)
    if trigger_sample < 0:
        raise ValueError(f"the trigger sample must be zero or more, not {trigger_sample}")
    alive = np.asarray(alive, dtype=np.float64)
    centroid_x = np.asarray(centroid_x, dtype=np.float64)
    centroid_y = np.asarray(centroid_y, dtype=np.float64)
    if not (alive.shape == centroid_x.shape == centroid_y.shape) or alive.ndim != 1:
        raise ValueError(
            "alive share and centroids must be one sample per turn each: "
            f"got {alive.shape}, {centroid_x.shape}, {centroid_y.shape}"
        )
    address_x, address_y, address_sum = addresses
    injected, stored = beam_sum(
        alive,
        trigger_sample=trigger_sample,
        charge=charge,
        stored_current=stored_current,
        rf_on=rf_on,
        readout=readout,
    )
    x, y = (
        turn_positions(
            centroid,
            injected,
            stored,
            trigger_sample=trigger_sample,
            address=address,
            shot=shot,
            tone_on=tone_on,
            readout=readout,
        )
        for centroid, address in ((centroid_x, address_x), (centroid_y, address_y))
    )
    total = turn_sum(
        injected + stored, address=address_sum, shot=shot, tone_on=tone_on, readout=readout
    )
    return x, y, total
