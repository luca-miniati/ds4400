"""Parser for IRC hroster (hand roster) files."""

from pathlib import Path


def parse_hroster(path: Path) -> dict[str, tuple[int, list[str]]]:
    """
    Parse an hroster file.

    Returns a dict mapping hand_id to (n_players, [player_names]).
    Player order: first = SB position, last = button (for NLH).
    """
    rosters = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            hand_id = parts[0]
            n_players = int(parts[1])
            players = parts[2:]
            if len(players) != n_players:
                # Sometimes roster may have extra/missing; use what we have
                players = players[:n_players] if len(players) >= n_players else players
            rosters[hand_id] = (n_players, players)
    return rosters
