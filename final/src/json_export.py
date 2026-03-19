"""Export IRCHand objects to JSON for downstream ML processing."""

import json
from pathlib import Path

from .irc_parser.merger import IRCHand


def hand_to_dict(hand: IRCHand) -> dict:
    """Convert IRCHand to a JSON-serializable dict."""
    return {
        "hand_id": hand.hand_id,
        "n_players": hand.n_players,
        "players": hand.players,
        "board_cards": hand.board_cards.split() if hand.board_cards else [],
        "pot_preflop": hand.pot_preflop,
        "pot_flop": hand.pot_flop,
        "pot_turn": hand.pot_turn,
        "pot_river": hand.pot_river,
        "pdb_rows": [
            {
                "player": r.player,
                "position": r.position,
                "preflop_actions": r.preflop_actions,
                "flop_actions": r.flop_actions,
                "turn_actions": r.turn_actions,
                "river_actions": r.river_actions,
                "stack": r.stack,
                "bet1": r.bet1,
                "bet2": r.bet2,
                "hole_cards": r.hole_cards.split() if r.hole_cards else None,
            }
            for r in hand.pdb_rows
        ],
    }


def export_hands_json(hands: dict[str, IRCHand], path: Path) -> None:
    """Export hands to a JSON file."""
    data = {hid: hand_to_dict(h) for hid, h in hands.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_hands_json_batched(
    hands: dict[str, IRCHand],
    output_dir: Path,
    *,
    batch_size: int = 100,
) -> list[Path]:
    """
    Export hands to batched JSON files (hands_000.json, hands_001.json, ...).

    Each file contains at most batch_size hands. Returns list of written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(hands.items())
    written: list[Path] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batch = {hid: hand_to_dict(h) for hid, h in chunk}
        batch_num = i // batch_size
        out_path = output_dir / f"hands_{batch_num:04d}.json"
        with open(out_path, "w") as f:
            json.dump(batch, f, indent=2)
        written.append(out_path)
    return written


def export_hands_jsonl(hands: dict[str, IRCHand], path: Path) -> None:
    """Export hands to JSONL (one JSON object per line)."""
    with open(path, "w") as f:
        for hid, hand in hands.items():
            rec = hand_to_dict(hand)
            rec["hand_id"] = hid
            f.write(json.dumps(rec) + "\n")
