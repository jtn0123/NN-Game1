"""Independent human buttons without expanding the checkpoint action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast


@dataclass(frozen=True)
class HumanControls:
    move: int
    jump: bool
    shoot: bool
    interact: bool

    @classmethod
    def from_request(cls, value: Any) -> HumanControls:
        fields = {"move", "jump", "shoot", "interact"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("each control must contain move, jump, shoot and interact")
        move = value["move"]
        if isinstance(move, bool) or not isinstance(move, int) or move not in (-1, 0, 1):
            raise ValueError("move must be an integer from -1 to 1")
        if any(not isinstance(value[name], bool) for name in ("jump", "shoot", "interact")):
            raise ValueError("jump, shoot and interact must be booleans")
        return cls(**value)

    def legacy_action(self) -> Optional[int]:
        """Return an exact existing action, or None for a compound input."""
        if self.interact:
            return 9 if not (self.move or self.jump or self.shoot) else None
        if self.jump and self.shoot:
            return None
        if self.jump:
            return 4 if self.move < 0 else 5 if self.move > 0 else 3
        if self.shoot:
            return 7 if self.move < 0 else 8 if self.move > 0 else 6
        return 1 if self.move < 0 else 2 if self.move > 0 else 0


class HumanControlsMixin:
    """Inject human buttons only for one authoritative step; legacy steps stay exact."""

    # MainMine's existing step already holds climbables on neutral input.
    LEGACY_CLIMB_HOLD = False

    def step_human(self: Any, controls: HumanControls) -> Any:
        action = controls.legacy_action()
        self._active_human_controls = controls
        self._human_demo_action = action
        try:
            state, reward, done, info = self.step(action if action is not None else self.IDLE)
            return state, reward, done, {**info, "human_demo_action": self._human_demo_action}
        finally:
            self._active_human_controls = None

    def _decode_action(self: Any, action: int) -> tuple[int, bool, bool, bool]:
        controls = getattr(self, "_active_human_controls", None)
        if controls is None:
            return cast(Any, super())._decode_action(action)
        return controls.move, controls.jump, controls.shoot, controls.interact

    def _apply_player_input(self: Any, move_dir: int, wants_jump: bool) -> None:
        cast(Any, super())._apply_player_input(move_dir, wants_jump)
        controls = getattr(self, "_active_human_controls", None)
        if (
            controls is not None
            and self._is_on_ladder()
            and not (controls.jump or controls.interact)
        ):
            self.vy = 0.0
            if not self.LEGACY_CLIMB_HOLD:
                # Legacy cave IDLE/SHOOT descends. Such a hold cannot be stored as
                # that action in a training demo, even without compound buttons.
                self._human_demo_action = None

    def _record_history_step(
        self: Any, action: int, previous_target: Any, previous_distance: float
    ) -> None:
        controls = getattr(self, "_active_human_controls", None)
        if controls is not None and controls.legacy_action() is None:
            # Private history tokens retain multiple active buttons in the same
            # seven feature channels. They are never protocol or demo actions.
            mask = (controls.move + 1) * 8
            mask += int(controls.jump) + int(controls.shoot) * 2 + int(controls.interact) * 4
            action = -2 - mask
        cast(Any, super())._record_history_step(action, previous_target, previous_distance)

    def _history_action_features(self: Any, action: int, progress_delta: float) -> list[float]:
        features = cast(Any, super())._history_action_features(action, progress_delta)
        if action <= -2:
            mask = -2 - action
            move = mask // 8 - 1
            features[1:6] = [
                float(move < 0),
                float(move > 0),
                float(bool(mask & 1)),
                float(bool(mask & 2)),
                float(bool(mask & 4)),
            ]
        return features
