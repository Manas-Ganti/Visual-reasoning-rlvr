"""Ensure the repo root is importable so ``import env`` / ``import data`` work
when pytest is invoked from anywhere in the tree."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
