import csv
import re
import subprocess
import sys
import os
from pathlib import Path

ARENA = Path(__file__).with_name("arena.py")
OUTPUT = Path(__file__).with_name("results.csv")
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions" / "reference" / "LAB2"

HIDER = "reference/LAB2/4"

BASE_ARGS = [
    "--pacman-obs-radius", "5",
    "--ghost-obs-radius", "5",
    "--start-mode", "stochastic",
    "--no-viz",
]


def discover_seekers():
    """
    Discover all reference LAB2 submissions.
    Excludes the hider (4) to avoid self-play.
    
    Returns:
        Sorted list of seeker paths (reference/LAB2/N)
    """
    if not SUBMISSIONS_DIR.exists():
        print(f"ERROR: Submissions directory not found at {SUBMISSIONS_DIR}")
        sys.exit(1)
    
    seekers = []
    for item in SUBMISSIONS_DIR.iterdir():
        if item.is_dir() and item.name != "4":
            # Check if it contains a valid agent.py
            if (item / "agent.py").exists():
                seekers.append(f"reference/LAB2/{item.name}")
    
    return sorted(seekers, key=lambda x: int(x.split('/')[-1]))


SEEKERS = discover_seekers()


def run_game(seeker, hider):
    cmd = [
        sys.executable,
        str(ARENA),
        "--seek", seeker,
        "--hide", hider,
        *BASE_ARGS,
    ]

    # Set UTF-8 encoding for subprocess to handle Unicode characters
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    completed = subprocess.run(
        cmd,
        cwd=ARENA.parent,
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=env,
    )

    output = completed.stdout + "\n" + completed.stderr

    if completed.returncode != 0:
        return "error", "", output

    if re.search(r"WINNER: .* \(Pacman\)", output):
        result = "seeker_wins"
    elif re.search(r"WINNER: .* \(Ghost\)", output):
        result = "hider_wins"
    elif "DRAW" in output:
        result = "draw"
    else:
        result = "unknown"

    match = re.search(r"Total Steps:\s*(\d+)", output)
    steps = match.group(1) if match else ""

    return result, steps, output


def main():
    rows = []

    total = len(SEEKERS)
    game_no = 0

    print(f"\n{'='*70}")
    print(f"{'TOURNAMENT: All LAB2 Seekers vs LAB2/4 Hider':^70}")
    print(f"{'='*70}")
    print(f"Hider: {HIDER}")
    print(f"Seekers: {len(SEEKERS)} submissions found")
    print(f"Total matches: {total}")
    print(f"{'='*70}\n")

    error_log = OUTPUT.with_name("error_log.txt")
    error_log.write_text("")  # Clear error log

    for seeker in SEEKERS:
        game_no += 1
        print(f"[{game_no}/{total}] {seeker} vs {HIDER}")

        result, steps, output = run_game(seeker, HIDER)
        
        if result == "error":
            # Log full error
            with error_log.open("a", encoding="utf-8") as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"ERROR: {seeker} vs {HIDER}\n")
                f.write(f"{'='*70}\n")
                f.write(output)
                f.write(f"\n")
            print(f"  ERROR - Full details saved to error_log.txt")

        rows.append({
                "seeker": seeker,
                "hider": HIDER,
                "result": result,
                "steps": steps,
            })

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seeker", "hider", "result", "steps"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: results.csv")


if __name__ == "__main__":
    main()
