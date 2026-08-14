
import csv
import re
import subprocess
import sys
import os
from pathlib import Path

ARENA = Path(__file__).with_name("arena.py")
OUTPUT = Path(__file__).with_name("results.csv")

SEEKERS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
HIDERS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Number of matches for each seeker vs hider matchup
MATCHES_PER_PAIR = 3

BASE_ARGS = [
    "--pacman-obs-radius", "5",
    "--ghost-obs-radius", "5",
    "--start-mode", "stochastic",
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

    total_matchups = len(seekers) * len(hiders)
    total_games = total_matchups * MATCHES_PER_PAIR

    matchup_no = 0
    game_no = 0

    print(f"\n{'=' * 70}")
    print(f"{'TOURNAMENT':^70}")
    print(f"{'=' * 70}")
    print(f"Seekers: {SEEKERS}")
    print(f"Hiders:  {HIDERS}")
    print(f"Matches per matchup: {MATCHES_PER_PAIR}")
    print(f"Total matchups: {total_matchups}")
    print(f"Total games: {total_games}")
    print(f"{'=' * 70}\n")

    error_log = OUTPUT.with_name("error_log.txt")
    error_log.write_text("", encoding="utf-8")

    for seeker, seeker_number in zip(seekers, SEEKERS):
        for hider, hider_number in zip(hiders, HIDERS):

            if seeker_number == hider_number:
                print(
                    f"\n[{matchup_no + 1}/{total_matchups}] "
                    f"Seeker {seeker_number} vs Hider {hider_number} "
                    f"- SKIPPED (same number)"
                )
                continue

            matchup_no += 1

            print(
                f"\n[{matchup_no}/{total_matchups}] "
                f"Seeker {seeker_number} vs Hider {hider_number}"
            )

            matchup_results = []
            matchup_steps = []

            for match_index in range(1, MATCHES_PER_PAIR + 1):
                game_no += 1

                print(
                    f"  Match {match_index}/{MATCHES_PER_PAIR} "
                    f"[Game {game_no}/{total_games}]"
                )

                result, steps, output = run_game(seeker, hider)

                matchup_results.append(result)

                if steps.isdigit():
                    matchup_steps.append(int(steps))

                if result == "error":
                    with error_log.open("a", encoding="utf-8") as f:
                        f.write(f"\n{'=' * 70}\n")
                        f.write(
                            f"ERROR: {seeker_number} vs {hider_number} "
                            f"(match {match_index})\n"
                        )
                        f.write(f"{'=' * 70}\n")
                        f.write(output)
                        f.write("\n")

                    print("    ERROR - Logged to error_log.txt")

                elif steps.isdigit():
                    print(f"    {result} - {steps} steps")

                else:
                    print(f"    {result} - no step count")

            if matchup_steps:
                avg_steps = sum(matchup_steps) / len(matchup_steps)
            else:
                avg_steps = None

            seeker_wins = matchup_results.count("seeker_wins")
            hider_wins = matchup_results.count("hider_wins")
            draws = matchup_results.count("draw")
            errors = matchup_results.count("error")
            unknown = matchup_results.count("unknown")

            result_parts = []

            if seeker_wins:
                result_parts.append(
                    f"seeker_wins {seeker_wins}/{MATCHES_PER_PAIR}"
                )

            if hider_wins:
                result_parts.append(
                    f"hider_wins {hider_wins}/{MATCHES_PER_PAIR}"
                )

            if draws:
                result_parts.append(
                    f"draw {draws}/{MATCHES_PER_PAIR}"
                )

            if errors:
                result_parts.append(
                    f"error {errors}/{MATCHES_PER_PAIR}"
                )

            if unknown:
                result_parts.append(
                    f"unknown {unknown}/{MATCHES_PER_PAIR}"
                )

            result_summary = ", ".join(result_parts)

            rows.append({
                "seeker": seeker_number,
                "hider": hider_number,
                "result": result_summary,
                "avg_steps": avg_steps,
            })

            if avg_steps is not None:
                print(f"  Average: {avg_steps:.2f} steps")
            else:
                print("  Average: no valid step counts")

    # =========================
    # Seeker summaries
    #
    # Seeker objective:
    # LOWER steps = BETTER
    # =========================

    seeker_summaries = []

    for seeker_number in SEEKERS:
        seeker_rows = [
            row
            for row in rows
            if row["seeker"] == seeker_number
            and row["avg_steps"] is not None
        ]

        total_wins = sum(
            row["result"].count("seeker_wins")
            for row in seeker_rows
        )

        if not seeker_rows:
            seeker_summaries.append({
                "seeker": seeker_number,
                "wins": total_wins,
                "avg_steps": None,
                "worst_matchup": "",
                "worst_avg_steps": None,
            })
            continue

        seeker_avg = sum(
            row["avg_steps"]
            for row in seeker_rows
        ) / len(seeker_rows)

        # Highest steps = worst for seeker
        worst = max(
            seeker_rows,
            key=lambda row: row["avg_steps"]
        )

        seeker_summaries.append({
            "seeker": seeker_number,
            "wins": total_wins,
            "avg_steps": seeker_avg,
            "worst_matchup": f"{worst['seeker']} vs {worst['hider']}",
            "worst_avg_steps": worst["avg_steps"],
        })

    # =========================
    # Hider summaries
    #
    # Hider objective:
    # HIGHER steps = BETTER
    # =========================

    hider_summaries = []

    for hider_number in HIDERS:
        hider_rows = [
            row
            for row in rows
            if row["hider"] == hider_number
            and row["avg_steps"] is not None
        ]

        total_wins = sum(
            row["result"].count("hider_wins")
            for row in hider_rows
        )

        if not hider_rows:
            hider_summaries.append({
                "hider": hider_number,
                "wins": total_wins,
                "avg_steps": None,
                "worst_matchup": "",
                "worst_avg_steps": None,
            })
            continue

        hider_avg = sum(
            row["avg_steps"]
            for row in hider_rows
        ) / len(hider_rows)

        # Lowest steps = worst for hider
        worst = min(
            hider_rows,
            key=lambda row: row["avg_steps"]
        )

        hider_summaries.append({
            "hider": hider_number,
            "wins": total_wins,
            "avg_steps": hider_avg,
            "worst_matchup": f"{worst['seeker']} vs {worst['hider']}",
            "worst_avg_steps": worst["avg_steps"],
        })

    # =========================
    # Dynamic column names
    # =========================

    avg_column = f"avg_steps_{MATCHES_PER_PAIR}_matches"
    worst_avg_column = f"worst_avg_steps_{MATCHES_PER_PAIR}_matches"

    # =========================
    # Write output
    # =========================

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seeker",
            "hider",
            "result",
            avg_column,
        ])

        for row in rows:
            writer.writerow([
                row["seeker"],
                row["hider"],
                row["result"],
                f"{row['avg_steps']:.2f}"
                if row["avg_steps"] is not None
                else "",
            ])

        # Seeker summary
        writer.writerow([])
        writer.writerow([
            "SEEKER",
            "wins",
            "avg_steps",
            "worst_matchup",
            worst_avg_column,
        ])

        for summary in seeker_summaries:
            writer.writerow([
                summary["seeker"],
                summary["wins"],
                f"{summary['avg_steps']:.2f}"
                if summary["avg_steps"] is not None
                else "",
                summary["worst_matchup"],
                f"{summary['worst_avg_steps']:.2f}"
                if summary["worst_avg_steps"] is not None
                else "",
            ])

        # Hider summary
        writer.writerow([])
        writer.writerow([
            "HIDER",
            "wins",
            "avg_steps",
            "worst_matchup",
            worst_avg_column,
        ])

        for summary in hider_summaries:
            writer.writerow([
                summary["hider"],
                summary["wins"],
                f"{summary['avg_steps']:.2f}"
                if summary["avg_steps"] is not None
                else "",
                summary["worst_matchup"],
                f"{summary['worst_avg_steps']:.2f}"
                if summary["worst_avg_steps"] is not None
                else "",
            ])

    print(f"\nResults written to: {OUTPUT}")
    print(f"Error log: {error_log}")

    print("\nSeeker summaries:")

    for summary in seeker_summaries:
        if summary["avg_steps"] is None:
            print(
                f"Seeker {summary['seeker']}: "
                f"{summary['wins']} wins, "
                "no valid matches"
            )
        else:
            print(
                f"Seeker {summary['seeker']}: "
                f"{summary['wins']} wins, "
                f"avg {summary['avg_steps']:.2f}, "
                f"worst matchup {summary['worst_matchup']} "
                f"({summary['worst_avg_steps']:.2f} avg steps)"
            )

    print("\nHider summaries:")

    for summary in hider_summaries:
        if summary["avg_steps"] is None:
            print(
                f"Hider {summary['hider']}: "
                f"{summary['wins']} wins, "
                "no valid matches"
            )
        else:
            print(
                f"Hider {summary['hider']}: "
                f"{summary['wins']} wins, "
                f"avg {summary['avg_steps']:.2f}, "
                f"worst matchup {summary['worst_matchup']} "
                f"({summary['worst_avg_steps']:.2f} avg steps)"
            )


if __name__ == "__main__":
    main()

