import numpy as np
from semcert.scoring import cosine_distance


def test_cosine_distance_identity():
    x = np.array([1.0, 2.0])
    assert abs(cosine_distance(x, x)) < 1e-9


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_distance(a, b) - 1.0) < 1e-9
