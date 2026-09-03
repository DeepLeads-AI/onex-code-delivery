"""The layer-profile metric: a per-request signature over all decoder layers.

Deliberately empty of side effects. Importing this package must not open a
database connection or load a model — those happen inside the functions that
need them, so ``--help``, the unit tests and the pure-numpy metric all stay free
of heavy dependencies.
"""
