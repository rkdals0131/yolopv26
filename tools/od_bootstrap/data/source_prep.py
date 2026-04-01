from __future__ import annotations

from . import _source_prep_impl as _impl
from ._source_prep_impl import *  # noqa: F401,F403

run_bdd_standardization = _impl.run_bdd_standardization
run_aihub_standardization = _impl.run_aihub_standardization


def prepare_od_bootstrap_sources(*args, **kwargs):
    _impl.run_bdd_standardization = run_bdd_standardization
    _impl.run_aihub_standardization = run_aihub_standardization
    return _impl.prepare_od_bootstrap_sources(*args, **kwargs)
