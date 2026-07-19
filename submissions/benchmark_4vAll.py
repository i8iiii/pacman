#!/usr/bin/env python3
"""Benchmark agent 4 (seeker) versus all hide agents in the submissions folder."""

import subprocess
import sys
from pathlib import Path

SUBMISSIONS_DIR = Path(__file__).parent
SRC_DIR = SUBMISSIONS_DIR.parent / "src"
ARENA = SRC_DIR / "arena.py"
VENV_PYTHON = str(SUBMISSIONS_DIR.parent / ".venv" / "bin" / "python")
ROUNDS = 5
MAX_STEPS = 200

HIDE_AGENTS = [f"reference/{i}" for i in range(17)] + ["LAB2"]


def run_game(seeker: str, hider: str, max_steps: int = MAX_STEPS) -> dict:
    cmd = [
        VENV_PYTHON, str(ARENA),
        "--seek", seeker,
        "--hide", hider,
        "--no-viz",
        "--max-steps", str(max_steps),
        "--step-timeout", "3",
        "--start-mode", "deterministic",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    stdout = proc.stdout
    result = {"winner": None, "steps": 0, "status": "ok"}

    if "WINNER: 4 (Pacman)" in stdout:
        result["winner"] = "4"
    elif "WINNER:" in stdout:
        for line in stdout.splitlines():
            if "WINNER:" in line:
                result["winner"] = line.split("WINNER:")[1].strip().split(" ")[0]
                break
    elif "DRAW" in stdout:
        result["winner"] = "draw"

    for line in stdout.splitlines():
        if "Total Steps:" in line:
            try:
                result["steps"] = int(line.split(":")[1].strip())
            except ValueError:
                pass

    if proc.returncode != 0:
        result["status"] = "error"

    return result


def main():
    print("=" * 70)
    print("BENCHMARK: Agent 4 (Seeker) vs All Hide Agents")
    print(f"Rounds per matchup: {ROUNDS}")
    print("=" * 70)

    hide_agents_exist = [h for h in HIDE_AGENTS if (SUBMISSIONS_DIR / h / "agent.py").exists()]
    if not hide_agents_exist:
        print("ERROR: No hide agents found!")
        sys.exit(1)

    print(f"\nFound {len(hide_agents_exist)} hide agents: {', '.join(hide_agents_exist)}")
    print()

    results = {}

    for hider in hide_agents_exist:
        print(f"\n--- {hider} ---")
        wins_4 = 0
        wins_hider = 0
        draws = 0
        errors = 0
        total_steps = 0

        for r in range(1, ROUNDS + 1):
            print(f"  Round {r}/{ROUNDS}...", end=" ", flush=True)
            result = run_game("4", hider)

            if result["status"] == "error":
                print("ERROR")
                errors += 1
            elif result["winner"] == "4":
                print("4 wins")
                wins_4 += 1
            elif result["winner"] == "draw":
                print("draw")
                draws += 1
            else:
                print(f"{result['winner']} wins")
                wins_hider += 1
            total_steps += result["steps"]

        valid_games = wins_4 + wins_hider + draws
        win_rate = (wins_4 / valid_games * 100) if valid_games > 0 else 0
        avg_steps = total_steps / max(1, ROUNDS - errors)

        results[hider] = {
            "wins_4": wins_4, "wins_hider": wins_hider,
            "draws": draws, "errors": errors,
            "win_rate": win_rate, "avg_steps": avg_steps,
        }

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Opponent':20s} | 4 Wins | Hider Wins | Draws | WinRate | AvgSteps")
    print("-" * 70)
    for hider, r in results.items():
        print(f"  {hider:20s} | {r['wins_4']:>7} | {r['wins_hider']:>10} | {r['draws']:>5} | {r['win_rate']:>5.0f}%  | {r['avg_steps']:>6.0f}")

    total_wins = sum(r["wins_4"] for r in results.values())
    total_games = sum(r["wins_4"] + r["wins_hider"] + r["draws"] for r in results.values())
    total_opponents = len(results)
    print("-" * 70)
    print(f"Overall: {total_wins}/{total_games} wins ({total_wins/total_games*100:.0f}%) across {total_opponents} opponents")
    print("=" * 70)


if __name__ == "__main__":
    main()
