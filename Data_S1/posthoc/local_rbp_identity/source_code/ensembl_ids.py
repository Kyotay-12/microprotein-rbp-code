"""Named, explicit Ensembl ID normalization policies.

The default removes a numeric version while preserving a pseudoautosomal
copy suffix. Legacy behavior is deliberately opt-in for historical replay.
Aggregation (e.g. adding X/Y abundances) is a separate scientific policy and
must never be hidden inside identifier normalization.
"""
import re

_IDENTIFIER = re.compile(r"^(ENS(?:G|T|P)\d+)(?:\.(\d+))?(_PAR_Y)?$")


def stable_ensembl_id(value, *, policy="preserve_PAR"):
    if not isinstance(value, str):
        raise TypeError("Ensembl identifier must be a string; missing IDs are not allowed")
    if policy == "legacy_first_dot":
        # Exact historical behavior, including its PAR_Y collision.
        return value.split(".", 1)[0]
    if policy != "preserve_PAR":
        raise ValueError(f"Unknown Ensembl normalization policy: {policy}")
    match = _IDENTIFIER.fullmatch(value)
    if not match:
        raise ValueError(f"Unrecognized Ensembl identifier: {value!r}")
    return match.group(1) + (match.group(3) or "")


def pseudoautosomal_base(value):
    """Explicit grouping helper for a documented PAR exclusion/summary only."""
    return stable_ensembl_id(value).removesuffix("_PAR_Y")


def is_par_y(value):
    return stable_ensembl_id(value).endswith("_PAR_Y")
