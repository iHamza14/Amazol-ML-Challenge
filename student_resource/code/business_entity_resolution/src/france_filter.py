"""
France-specific post-processing (unseen country safety net).

Rule (adapted from the approach shared by a top-15 team, corrected for house-number
truncation noise measured on training data):

  For every France S1 entity with >= 1 predicted match, a match is "supported" when
      street token-set similarity >= FRANCE_STREET_SIM_THRESHOLD   AND
      the first-number relation is NOT a substitution / disjoint number
      (equal, truncation prefix/suffix, appears elsewhere, or one side missing are all OK).
  If at least one predicted match is supported, keep ONLY the supported ones.
  If none is supported, keep the model's predictions (the model knows best).

The filter is evaluated on US/India validation in train.py to verify it never hurts; it
operates on per-pair columns that the prediction pass already computed (st_tset,
num_first_rel), so it costs nothing at inference.
"""
import logging
import numpy as np

import config as cfg

log = logging.getLogger(__name__)

# num_first_rel codes (see features._first_num_relation)
_UNSUPPORTED_REL = {7, 8, 9}


def apply_france_filter(mask, s1_code, country, st_sim, num_rel, target_country='france',
                        street_sim_threshold=None):
    """
    mask     : boolean selection over pairs (current predictions)
    s1_code  : int S1 codes per pair
    country  : str array per pair (S1 country)
    st_sim   : street similarity per pair (0-100)
    num_rel  : first-number relation code per pair
    Returns new mask.
    """
    if not cfg.FRANCE_ENABLED:
        return mask
    th = street_sim_threshold if street_sim_threshold is not None else cfg.FRANCE_STREET_SIM_THRESHOLD
    sel = mask & (country == target_country)
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return mask
    supported = (st_sim[idx] >= th) & ~np.isin(num_rel[idx], list(_UNSUPPORTED_REL))
    # per S1: does any predicted match have support?
    codes = s1_code[idx]
    order = np.argsort(codes, kind='stable')
    codes_o = codes[order]
    sup_o = supported[order]
    starts = np.r_[0, np.flatnonzero(np.diff(codes_o)) + 1]
    any_sup = np.logical_or.reduceat(sup_o, starts)
    counts = np.diff(np.r_[starts, len(codes_o)])
    any_sup_pair = np.repeat(any_sup, counts)
    drop_o = any_sup_pair & ~sup_o
    new_mask = mask.copy()
    drop_idx = idx[order[drop_o]]
    new_mask[drop_idx] = False
    log.info(f"  {target_country} filter: {len(idx):,} predicted pairs, removed {len(drop_idx):,} unsupported")
    return new_mask
