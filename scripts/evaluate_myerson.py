"""Evaluate theoretical Myerson and the learned reserve-price policy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comb_auction.myerson_check_single_item import main


if __name__ == "__main__":
    main()
