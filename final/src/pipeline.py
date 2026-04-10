"""Orchestration for IRC to PHH conversion pipeline."""

from pathlib import Path

from .irc_parser import merge_hands
from .json_export import export_hands_json_batched
from .phh_converter import convert_to_hand_history


def discover_irc_dirs(root: Path, nlh_only: bool = True) -> list[tuple[Path, Path, Path]]:
    """
    Discover (hdb, hroster, pdb_dir) tuples under root.
    Looks for holdem*/YYYYMM/ structure.
    If nlh_only=True, only includes holdem1, holdem2, holdem3 (NLH) paths.
    """
    found = []
    for hdb_path in root.rglob("hdb"):
        if not hdb_path.is_file():
            continue
        if nlh_only:
            parts = hdb_path.parts
            if "holdem1" not in parts and "holdem2" not in parts and "holdem3" not in parts:
                continue
        parent = hdb_path.parent
        hroster = parent / "hroster"
        pdb_dir = parent / "pdb"
        if hroster.exists() and pdb_dir.exists() and pdb_dir.is_dir():
            found.append((hdb_path, hroster, pdb_dir))
    return found


def run_pipeline(
    irc_root: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    verbose: bool = True,
) -> tuple[int, int, int]:
    """
    Run the full pipeline: parse IRC, convert to PHH, write output.

    Returns (parsed_count, converted_count, failed_count).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    phh_dir = output_dir / "phh"
    json_dir.mkdir(exist_ok=True)
    phh_dir.mkdir(exist_ok=True)

    locations = discover_irc_dirs(Path(irc_root))
    if not locations:
        if verbose:
            print(f"No IRC data found under {irc_root}")
        return 0, 0, 0

    all_hands: list[tuple[str, object]] = []  # (hand_id, IRCHand)
    hands_dict: dict = {}

    for hdb_path, hroster_path, pdb_dir in locations:
        hands = merge_hands(hdb_path, hroster_path, pdb_dir)
        for hid, h in hands.items():
            all_hands.append((hid, h))
            hands_dict[hid] = h

    parsed_count = len(all_hands)

    # Export parsed hands to JSON in batches of 100 (avoids huge single file)
    batched = export_hands_json_batched(hands_dict, json_dir, batch_size=100)
    if verbose:
        print(f"Exported {parsed_count} parsed hands to {len(batched)} batched JSON files in {json_dir.name}/")

    if limit:
        all_hands = all_hands[:limit]

    converted = []
    failed = 0

    for hand_id, hand in all_hands:
        hh = convert_to_hand_history(hand)
        if hh is not None:
            converted.append((hand_id, hh))
        else:
            failed += 1
            if verbose and failed <= 5:
                print(f"  Failed to convert hand {hand_id}")

    # Write PHH output in batches of 1000
    if converted:
        from pokerkit import HandHistory

        batch_size = 1000
        written: list[Path] = []
        for i in range(0, len(converted), batch_size):
            chunk = converted[i : i + batch_size]
            batch_num = i // batch_size
            out_path = phh_dir / f"hands_{batch_num:04d}.phhs"
            with open(out_path, "wb") as f:
                HandHistory.dump_all([hh for _, hh in chunk], f)
            written.append(out_path)
        if verbose:
            print(f"Wrote {len(converted)} hands to {len(written)} batched PHH files in {phh_dir.name}/")

    return parsed_count, len(converted), failed
