"""Stochastic Slingshot Bandwidth & Telemetry Stream Generator.

Simulates bursty packet and queue dynamics ('estilingue na banda') where
inter-arrival delays (IPD) collapse into high-density bursts, followed by
idle accumulation phases. Values fluctuate between 0 and 100.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Iterator


@dataclass
class TelemetryEvent:
    timestamp: float
    value: int
    is_burst: bool
    label: str


class SlingshotTelemetryStream:
    """Generates non-uniform, bursty telemetry events with slingshot dynamics."""

    def __init__(
        self,
        base_interval_s: float = 1.0,
        burst_probability: float = 0.25,
        anomaly_threshold: int = 80,
    ):
        self.base_interval_s = base_interval_s
        self.burst_probability = burst_probability
        self.anomaly_threshold = anomaly_threshold
        self._in_burst = False
        self._burst_remaining = 0

    def next_event(self) -> tuple[TelemetryEvent, float]:
        """Calculates next event and required sleep interval before next tick."""
        now = time.time()

        # State transition: trigger burst release ("estilingue")
        if not self._in_burst:
            if random.random() < self.burst_probability:
                self._in_burst = True
                self._burst_remaining = random.randint(3, 6)

        if self._in_burst:
            # Slingshot release: high metric spike, compressed inter-packet delay
            val = random.randint(70, 99)
            delay = random.uniform(0.1, 0.35)
            self._burst_remaining -= 1
            if self._burst_remaining <= 0:
                self._in_burst = False
            label = "BURST_SPIKE" if val >= self.anomaly_threshold else "BURST_ELEVATED"
            is_burst = True
        else:
            # Accumulation phase: nominal metrics, normal delay
            val = random.randint(10, 55)
            delay = random.uniform(0.8 * self.base_interval_s, 1.4 * self.base_interval_s)
            label = "NOMINAL"
            is_burst = False

        event = TelemetryEvent(
            timestamp=now,
            value=val,
            is_burst=is_burst,
            label=label,
        )
        return event, delay

    async def aiter_events(self, max_events: int = 10) -> AsyncGenerator[TelemetryEvent, None]:
        """Asynchronously streams telemetry events with real-time slingshot delays."""
        for _ in range(max_events):
            event, delay = self.next_event()
            yield event
            await asyncio.sleep(delay)

    def iter_events_fast(self, max_events: int = 10) -> Iterator[TelemetryEvent]:
        """Synchronously emits events for rapid batch simulation without physical delays."""
        for _ in range(max_events):
            event, _ = self.next_event()
            yield event
