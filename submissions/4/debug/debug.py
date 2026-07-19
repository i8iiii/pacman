from pathlib import Path
from datetime import datetime
import traceback


DEBUG_ENABLED = False
LOG_PATH = Path(__file__).resolve().parent / "log.txt"


def _format(value) -> str:
   if hasattr(value, "name"):
      return value.name
   if isinstance(value, tuple):
      return "(" + ",".join(str(item) for item in value) + ")"
   if isinstance(value, float):
      return f"{value:.2f}".rstrip("0").rstrip(".")
   return str(value)


def _format_fields(fields: dict) -> str:
   parts = []
   for name, value in fields.items():
      if value is None or value is False or value == []:
         continue
      if name == "flags":
         value = ",".join(value)
      elif name == "pacman_best" and isinstance(value, tuple):
         value = f"{_format(value[0])}->{_format(value[1])}"
      else:
         value = _format(value)
      parts.append(f"{name}={value}")
   return " ".join(parts)


def log(message: str) -> None:
   if not DEBUG_ENABLED:
      return

   try:
      LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

      with open(LOG_PATH, "a", encoding="utf-8") as file:
         file.write(str(message) + "\n")

   except Exception:
      pass


def clear_log() -> None:
   if not DEBUG_ENABLED:
      return

   try:
      LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

      with open(LOG_PATH, "w", encoding="utf-8") as file:
         file.write("=== HIDE AGENT DIAGNOSTIC LOG ===\n")
         file.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")

   except Exception:
      pass


def start_turn(step, mode, ghost, pacman, previous=None) -> None:
   log(f"\n[TURN {step}] mode={mode}")
   log(f"  state: {_format_fields({'ghost': ghost, 'pacman': pacman, 'previous': previous})}")


def context(**fields) -> None:
   formatted = _format_fields(fields)
   if formatted:
      log(f"  context: {formatted}")


def event(name: str, **fields) -> None:
   formatted = _format_fields(fields)
   log(f"  {name}:{' ' + formatted if formatted else ''}")


def candidate(move, position, **fields) -> None:
   formatted = _format_fields(fields)
   log(f"    {_format(move)} -> {_format(position)}{' ' + formatted if formatted else ''}")


def decision(move, position, reason: str) -> None:
   log(f"  decision: {_format(move)} -> {_format(position)}")
   log(f"  reason: {reason}")


def finish_turn(runtime_ms: float) -> None:
   log(f"  runtime: {round(runtime_ms)} ms")


def log_exception(error: Exception) -> None:
   log(f"  [ERROR] {type(error).__name__}: {error}")
   for line in traceback.format_exc().rstrip().splitlines():
      log(f"    {line}")

