import csv
import re
import subprocess
import sys
import os
from pathlib import Path

ARENA = Path(__file__).with_name("arena.py")
OUTPUT = Path(__file__).with_name("results.csv")

SEEKERS = [4]
HIDERS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

BASE_ARGS = [
    "--pacman-obs-radius", "5",
    "--ghost-obs-radius", "5",
    "--start-mode", "deterministic",
    "--no-viz",
]


def make_path(number):
    return f"run_benchmark/{number}"


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

    for seeker, seeker_number in zip(seekers, SEEKERS):
        for hider, hider_number in zip(hiders, HIDERS):
            game_no += 1

            print(f"[{game_no}/{total}] {seeker_number} vs {hider_number}")

            result, steps, output = run_game(seeker, hider)

            if result == "error":
                with error_log.open("a", encoding="utf-8") as f:
                    f.write(f"\n{'=' * 70}\n")
                    f.write(f"ERROR: {seeker_number} vs {hider_number}\n")
                    f.write(f"{'=' * 70}\n")
                    f.write(output)
                    f.write("\n")

                print("  ERROR - Logged to error_log.txt")

            rows.append({
                "seeker": seeker_number,
                "hider": hider_number,
                "result": result,
                "steps": steps,
            })

    valid_rows = [
        row for row in rows
        if row["steps"].isdigit()
    ]

    if valid_rows:
        avg_steps = sum(
            int(row["steps"])
            for row in valid_rows
        ) / len(valid_rows)

        best = min(
            valid_rows,
            key=lambda row: int(row["steps"])
        )

        worst = max(
            valid_rows,
            key=lambda row: int(row["steps"])
        )

        best_matchup = f"{best['seeker']} vs {best['hider']}"
        worst_matchup = f"{worst['seeker']} vs {worst['hider']}"

        best_steps = best["steps"]
        worst_steps = worst["steps"]

    else:
        avg_steps = ""
        best_matchup = ""
        worst_matchup = ""
        best_steps = ""
        worst_steps = ""

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seeker",
            "hider",
            "result",
            "steps",
        ])

        for row in rows:
            writer.writerow([
                row["seeker"],
                row["hider"],
                row["result"],
                row["steps"],
            ])

        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow([
            "avg_steps",
            f"{avg_steps:.2f}" if avg_steps != "" else ""
        ])
        writer.writerow([
            "best_matchup",
            best_matchup
        ])
        writer.writerow([
            "best_steps",
            best_steps
        ])
        writer.writerow([
            "worst_matchup",
            worst_matchup
        ])
        writer.writerow([
            "worst_steps",
            worst_steps
        ])

    print(f"\nResults written to: {OUTPUT}")
    print(f"Error log: {error_log}")

    if valid_rows:
        print(f"\nAverage steps: {avg_steps:.2f}")
        print(f"Best matchup:  {best_matchup} ({best_steps} steps)")
        print(f"Worst matchup: {worst_matchup} ({worst_steps} steps)")
    else:
        print("\nNo valid matches with step counts.")


if __name__ == "__main__":
    main()