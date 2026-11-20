"""Check the weather data and the calibrated generator.

    uv run python scripts/check_weather.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.weather import report  # noqa: E402

if __name__ == "__main__":
    report()
