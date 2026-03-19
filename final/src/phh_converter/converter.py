"""Convert IRCHand to PokerKit HandHistory (state-driven only)."""

from .state_converter import (
    convert_to_hand_history_state_driven,
    DEFAULT_SB,
    DEFAULT_BB,
)


def convert_to_hand_history(
    hand,
    sb: int = DEFAULT_SB,
    bb: int = DEFAULT_BB,
    *,
    raise_on_error: bool = False,
):
    """
    Convert IRCHand to PokerKit HandHistory.

    Uses state-driven conversion: drives PokerKit's state machine directly.
    Failures surface which action broke (via ConversionError when raise_on_error=True).
    """
    return convert_to_hand_history_state_driven(
        hand, sb=sb, bb=bb, raise_on_error=raise_on_error
    )
