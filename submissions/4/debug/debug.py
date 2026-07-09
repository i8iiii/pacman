from pathlib import Path
from datetime import datetime


DEBUG_ENABLED = True


def log(message: str) -> None:
   if not DEBUG_ENABLED:
      return

   try:
      log_path = Path(__file__).resolve().parent / "log.txt"
      log_path.parent.mkdir(parents=True, exist_ok=True)

      with open(log_path, "a", encoding="utf-8") as file:
         file.write(str(message) + "\n")

   except Exception:
      pass


def clear_log() -> None:
   if not DEBUG_ENABLED:
      return

   try:
      log_path = Path(__file__).resolve().parent / "log.txt"
      log_path.parent.mkdir(parents=True, exist_ok=True)

      with open(log_path, "w", encoding="utf-8") as file:
         file.write("=== DEBUG LOG ===\n")
         file.write(f"Started: {datetime.now()}\n\n")

   except Exception:
      pass

