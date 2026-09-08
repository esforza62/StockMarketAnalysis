Plain assert scripts, no pytest dependency -- `python tests/test_grade_backfill.py`
runs one, and pytest collects them too if you happen to have it installed.

They cover the grade-tracking chain (history store, backfill, report) because
that is where a silent wrong answer is most costly: a backfilled grade that
peeked at future bars would produce a flattering performance report and no
error. `test_grade_backfill.py` checks exactly that by grading a date twice,
once with later bars present and once with them removed, and requiring the
two results to be identical.

Bars are synthetic throughout -- these run without a Tiingo key.
