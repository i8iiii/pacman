#!/usr/bin/env python3
import subprocess
import re
import sys

def run_single_game(seek_id, hide_id, submissions_dir="/home/ntdat/Documents/pacman/submissions"):
    """Run a single game and return the number of steps if pacman wins, else None"""
    cmd = [
        "python", "src/arena.py",
        "--seek", seek_id,
        "--hide", hide_id,
        "--submissions-dir", submissions_dir,
        "--no-viz",
        "--max-steps", "200",
        "--start-mode", "deterministic"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/ntdat/Documents/pacman")
    output = result.stdout + result.stderr
    print(output)  # debug

    # Look for total steps
    steps_match = re.search(r"Total Steps: (\d+)", output)
    winner_match = re.search(r"WINNER:.*\(Pacman\)", output)

    if steps_match and winner_match:
        return int(steps_match.group(1))
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark_deterministic.py <seek_id> <hide_id> <num_games=5>")
        sys.exit(1)

    seek_id = sys.argv[1]
    hide_id = sys.argv[2]
    num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"Running {num_games} games: {seek_id} (seeker) vs {hide_id} (hider)")
    print("-" * 60)

    total_steps = 0
    wins = 0
    all_steps = []

    for i in range(num_games):
        print(f"\n--- Game {i+1} ---")
        steps = run_single_game(seek_id, hide_id)
        if steps is not None:
            wins += 1
            total_steps += steps
            all_steps.append(steps)
            print(f"Game {i+1}: WIN in {steps} steps")
        else:
            print(f"Game {i+1}: LOSS or timeout")

    print("-" * 60)
    if wins > 0:
        avg = total_steps / wins
        print(f"Results: {wins}/{num_games} wins")
        print(f"Average steps to capture: {avg:.2f}")
        print(f"Min steps: {min(all_steps)}, Max steps: {max(all_steps)}")
    else:
        print("No wins to calculate average.")

if __name__ == "__main__":
    main()
