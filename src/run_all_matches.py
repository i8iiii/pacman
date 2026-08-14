import csv
import re
import subprocess
import sys
import os
from pathlib import Path

ARENA = Path(__file__).with_name("arena.py")
OUTPUT = Path(__file__).with_name("results.csv")

# ============================================================
# CONFIGURATION
# Specify exactly which submissions participate
# Example:
# SEEKERS = [1, 2]
# HIDERS = [1, 4, 7, 16]
# ============================================================

SEEKERS = [0, 2]
HIDERS = [0, 4, 7, 16]

BASE_ARGS = [
    "--pacman-obs-radius", "5",
    "--ghost-obs-radius", "5",
    "--start-mode", "stochastic",
    "--no-viz",
]


def make_path(number):
    return f"reference/LAB2/{number}"


def run_game(seeker, hider):
    cmd = [
        sys.executable,
        str(ARENA),
        "--seek", seeker,
        "--hide", hider,
        *BASE_ARGS,
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        cmd,
        cwd=ARENA.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
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

    seekers = [make_path(n) for n in SEEKERS]
    hiders = [make_path(n) for n in HIDERS]

    total = len(seekers) * len(hiders)
    game_no = 0

    print(f"\n{'=' * 70}")
    print(f"{'TOURNAMENT':^70}")
    print(f"{'=' * 70}")
    print(f"Seekers: {SEEKERS}")
    print(f"Hiders:  {HIDERS}")
    print(f"Total matches: {total}")
    print(f"{'=' * 70}\n")

    error_log = OUTPUT.with_name("error_log.txt")
    error_log.write_text("", encoding="utf-8")

    for seeker in seekers:
        for hider in hiders:
            game_no += 1

            print(f"[{game_no}/{total}] {seeker} vs {hider}")

            result, steps, output = run_game(seeker, hider)

            if result == "error":
                with error_log.open("a", encoding="utf-8") as f:
                    f.write(f"\n{'=' * 70}\n")
                    f.write(f"ERROR: {seeker} vs {hider}\n")
                    f.write(f"{'=' * 70}\n")
                    f.write(output)
                    f.write("\n")

                print("  ERROR - Logged to error_log.txt")

            rows.append({
                "seeker": seeker,
                "hider": hider,
                "result": result,
                "steps": steps,
            })

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seeker", "hider", "result", "steps"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {OUTPUT}")
    print(f"Error log: {error_log}")


if __name__ == "__main__":
    main()