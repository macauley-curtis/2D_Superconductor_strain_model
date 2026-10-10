from src.dirac_delta import delta


def test_delta():
    # Test case 1: de = 1, x = 0
    de = 1
    x = 0
    assert delta(de, x) == 1

    # Test case 2: de = 2, x = 1
    de = 0
    x = 1

    assert delta(de, x) == 0

    # Test case 3: de = 0.5, x = -0.5
    de = 1
    x = 1

    assert delta(de, x) == 0.5