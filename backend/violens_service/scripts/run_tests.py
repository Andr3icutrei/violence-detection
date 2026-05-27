import sys

import pytest


if __name__ == "__main__":
    args = ["-q", "--cov", "--cov-branch", "--cov-report=term-missing"]
    raise SystemExit(pytest.main(args))

