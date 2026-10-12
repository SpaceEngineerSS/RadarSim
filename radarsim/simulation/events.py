"""
Simulation events.

The engine records noteworthy changes as they happen so the GUI, reports and
tests do not have to infer them from frame-to-frame differences.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EventKind(Enum):
    DEPLOYED = "deployed"  # chaff / decoy / late target enters the scene
    FIRST_DETECTION = "first_detection"
    TRACK_INITIATED = "track_initiated"
    TRACK_CONFIRMED = "track_confirmed"
    TRACK_LOST = "track_lost"
    MASKED = "masked"  # line of sight blocked by terrain
    UNMASKED = "unmasked"
    JAMMED = "jammed"  # noise jamming costs more than JAMMED_LOSS_DB
    BURN_THROUGH = "burn_through"  # detected while still being jammed


# Jamming loss above which a look counts as jammed [dB].
JAMMED_LOSS_DB = 10.0


@dataclass(frozen=True)
class SimEvent:
    time: float
    kind: EventKind
    message: str
    target_id: Optional[int] = None
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "kind": self.kind.value,
            "message": self.message,
            "target_id": self.target_id,
            "track_id": self.track_id,
        }
