#!/usr/bin/env python3
"""Show sample hands from a PHH file in readable format."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    from pokerkit import HandHistory

    phh_dir = REPO / "data" / "phh" / "phh"
    base_dir = REPO / "data" / "phh"
    if phh_dir.exists():
        phh_files = sorted(phh_dir.glob("hands_*.phhs"))
    elif (base_dir / "hands.phhs").exists():
        phh_files = [base_dir / "hands.phhs"]
    else:
        print(f"No PHH files found in {base_dir}")
        return 1

    if not phh_files:
        print(f"No PHH files found. Run the pipeline first.")
        return 1

    hands: list = []
    for p in phh_files:
        with open(p, "rb") as f:
            hands.extend(HandHistory.load_all(f))

    print(f"Total hands: {len(hands)}\n")
    print("=" * 70)
    print("Sample hands (first 3)")
    print("=" * 70)

    for i, hh in enumerate(hands[:3]):
        hand_id = getattr(hh, "hand", "?")
        players = getattr(hh, "players", [])
        print(f"\n--- Hand {i + 1}: id={hand_id}, players={players} ---\n")
        for line in hh:
            print(f"  {line}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
