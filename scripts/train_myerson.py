"""Train the single-item Myerson reserve-price learning sanity check."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comb_auction.mech_myerson_rl import main


if __name__ == "__main__":
    main()
