from semcert.budget import zero_failure_audits_for_upper_bound, optimistic_budget_check


def test_zero_failure_count_for_half_percent():
    n = zero_failure_audits_for_upper_bound(0.005, 0.95)
    assert 590 <= n <= 605


def test_budget_check():
    x = optimistic_budget_check(1000, 0.05, 0.005, 4, overlap_factor=2)
    assert x["available_full_model_equivalent_audits_per_day"] == 50
    assert x["optimistic_audits_required"] > 0
