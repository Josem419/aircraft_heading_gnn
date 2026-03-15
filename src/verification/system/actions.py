"""Action representations for the system."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Action:
    """Action taken by the agent.

    For the aircraft heading advisory system, this is a commanded heading.

    Attributes:
        heading_command: target heading in radians
        metadata: additional action data
    """

    heading_command: float
    metadata: Optional[Dict[str, Any]] = None
