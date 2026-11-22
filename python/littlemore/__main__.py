"""`python -m littlemore ...` is the same as the `littlemore` command."""
import sys

from .cli import main

sys.exit(main())
