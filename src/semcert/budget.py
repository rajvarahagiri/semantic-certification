from __future__ import annotations

import math


def zero_failure_audits_for_upper_bound(epsilon: float, confidence: float = 0.95) -> int:
    """Exact n such that P(0 failures | p=epsilon) <= 1-confidence.

    Solves (1-epsilon)^n <= 1-confidence.
    """
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    return math.ceil(math.log(1 - confidence) / math.log(1 - epsilon))


def optimistic_budget_check(
    lead_changes_per_day: float,
    audit_budget_fraction: float,
    epsilon: float,
    demanded_populations: int,
    overlap_factor: float = 1.0,
) -> dict:
    available = lead_changes_per_day * audit_budget_fraction
    per_population = zero_failure_audits_for_upper_bound(epsilon)
    required = per_population * demanded_populations / max(overlap_factor, 1e-9)
    return {
        "lead_changes_per_day": lead_changes_per_day,
        "audit_budget_fraction": audit_budget_fraction,
        "available_full_model_equivalent_audits_per_day": available,
        "zero_failure_audits_per_population": per_population,
        "demanded_populations": demanded_populations,
        "assumed_overlap_factor": overlap_factor,
        "optimistic_audits_required": required,
        "days_to_accumulate_required_audits": required / max(available, 1e-9),
    }
