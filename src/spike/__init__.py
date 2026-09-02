"""Spike detection package (Phase 3)."""

from src.spike.spike_scorer import (
    SPIKE_FEATURE_NAMES,
    SPIKE_WINDOW_1H,
    SPIKE_WINDOW_24H,
    SpikeEvent,
    SpikeScorer,
    detect_spike_events,
    select_spike_threshold_on_validation,
)

__all__ = [
    "SPIKE_FEATURE_NAMES",
    "SPIKE_WINDOW_1H",
    "SPIKE_WINDOW_24H",
    "SpikeEvent",
    "SpikeScorer",
    "detect_spike_events",
    "select_spike_threshold_on_validation",
]
