Plain assert scripts, no pytest dependency -- `python tests/test_grade_backfill.py`
runs one, and pytest collects them too if you happen to have it installed.

They cover the grade-tracking chain (history store, backfill, report) because
that is where a silent wrong answer is most costly: a backfilled grade that
peeked at future bars would produce a flattering performance report and no
error. `test_grade_backfill.py` checks exactly that by grading a date twice,
once with later bars present and once with them removed, and requiring the
two results to be identical.

`test_setup_context.py` guards the other silent failure in the same chain:
news sentiment and the earnings date are shown on the dashboard but must
never move a grade, and nothing about a row would look wrong if they
started to. It scores one setup twice, with and without that context, and
requires every component and the letter to match.

`test_earnings_history.py` covers the one property the earnings record
exists to guarantee: that a report date was known BEFORE the event. It checks
the transition logic that turns "next report" into "reported on" -- including
that a date moving EARLIER is a correction rather than a report, which would
otherwise invent report dates that never happened.

`test_oos_split.py` guards the boundary rule in `portfolio_cli`. A trade
straddling the in/out-of-sample split is the one place a holdout can leak:
kept in sample it lets post-split prices decide an in-sample result, and
moved out of sample it credits the later window with an entry it never made.
It must be dropped from BOTH, and the test pins that -- plus the split date
itself, because the recorded result table was produced at 2024-01-01 and the
prose would otherwise quietly describe a different experiment.

Bars are synthetic throughout -- these run without a Tiingo key.
