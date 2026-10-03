"""`python -m rtb.generate --config configs/models/<model>.yaml`."""

import sys

from rtb.generate.runner import main

if __name__ == "__main__":
    sys.exit(main())
