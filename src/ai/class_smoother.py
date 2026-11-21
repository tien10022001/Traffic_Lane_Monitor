from collections import Counter, deque
from typing import Dict, Optional, Tuple
from src.core.config_loader import get_config


class ClassSmoother:
    """
    Maintains a rolling window of predicted class_ids for each track_id.
    Resolves final class via majority voting to prevent label jitter (e.g. car vs truck).
    """

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size or get_config().tracking.class_smoothing_window
        self.track_history: Dict[int, deque] = {}

    def update(self, track_id: int, class_id: int) -> int:
        """Add new detection class_id and return smoothed majority class_id."""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.window_size)

        history = self.track_history[track_id]
        history.append(class_id)

        # Majority voting
        counter = Counter(history)
        majority_class, _ = counter.most_common(1)[0]
        return majority_class

    def remove(self, track_id: int) -> None:
        """Remove history when a track is deleted."""
        self.track_history.pop(track_id, None)

    def prune(self, active_track_ids: set) -> None:
        """Prune stale tracks no longer active."""
        stale_ids = [tid for tid in self.track_history if tid not in active_track_ids]
        for tid in stale_ids:
            del self.track_history[tid]
