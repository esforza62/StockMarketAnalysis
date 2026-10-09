"""The in/out-of-sample boundary rule.

A trade that straddles the split is the one place a holdout can leak. Put
it in-sample and post-split prices decide an in-sample result -- the exact
lookahead the split exists to prevent. Put it out-of-sample and the later
window is credited with an entry made on information it never saw. Neither
is acceptable, so it is dropped from both, and this file pins that.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from smc_regime.portfolio_cli import OOS_SPLIT, split_trades


class T:
    """Minimal stand-in: split_trades only ever reads the two dates."""

    def __init__(self, entry, exit_):
        self.entry_date = pd.Timestamp(entry)
        self.exit_date = pd.Timestamp(exit_)

    def __repr__(self):
        return f"T({self.entry_date.date()}->{self.exit_date.date()})"


SPLIT = pd.Timestamp("2024-01-01")

# 1. wholly before -> in sample
wholly_before = T("2023-06-01", "2023-06-10")
# 2. wholly after -> out of sample
wholly_after = T("2024-06-01", "2024-06-10")
# 3. straddles the split -> dropped from BOTH
straddles = T("2023-12-28", "2024-01-04")
# 4. exits exactly ON the split -- still a post-split price, so not in sample
exits_on_split = T("2023-12-28", "2024-01-01")
# 5. enters exactly ON the split -> out of sample, the window is inclusive
enters_on_split = T("2024-01-01", "2024-01-05")

is_, oos, dropped = split_trades(
    [wholly_before, wholly_after, straddles, exits_on_split, enters_on_split], SPLIT)

assert is_ == [wholly_before], f"in-sample should be only the pre-split trade, got {is_}"
assert wholly_after in oos, "a trade entirely after the split belongs out of sample"
assert enters_on_split in oos, "entry ON the split is out of sample (>= is inclusive)"
assert len(oos) == 2, f"out-of-sample should hold exactly 2 trades, got {oos}"

# The straddler and the one exiting on the split are both dropped, and
# crucially appear in NEITHER window -- not merely absent from one.
assert dropped == 2, f"expected 2 dropped boundary trades, got {dropped}"
for t in (straddles, exits_on_split):
    assert t not in is_, f"{t} straddles the split and must not be in sample"
    assert t not in oos, f"{t} entered pre-split and must not be out of sample"

# 6. Nothing is silently invented or lost: every input lands in exactly one
# of {in, out, dropped}.
assert len(is_) + len(oos) + dropped == 5, "trades went missing or were duplicated"

# 7. A trade exiting one day BEFORE the split is in sample -- the boundary is
# not off by one in the safe direction either, which would quietly shrink the
# training window.
day_before = T("2023-12-20", "2023-12-29")
is2, oos2, d2 = split_trades([day_before], SPLIT)
assert is2 == [day_before] and not oos2 and d2 == 0, "pre-split exit must stay in sample"

# 8. Empty input is not an error.
assert split_trades([], SPLIT) == ([], [], 0), "empty input should return empty windows"

# 9. The split constant is the one the recorded findings were produced with.
# Changing it invalidates the result table in portfolio_cli's docstring, so
# this fails loudly rather than letting the prose silently describe a
# different experiment.
assert OOS_SPLIT == "2024-01-01", (
    f"OOS_SPLIT is now {OOS_SPLIT!r}; the docstring's OOS table was produced "
    "at 2024-01-01 and must be re-run and rewritten if the split moves")

print("ok - 9 checks: boundary trades dropped from both windows, split pinned")
