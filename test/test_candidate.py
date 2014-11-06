import candidate


def test_get_empirical_probabilities():
    assert len(candidate.TableauSet('test/bata.txt').empirical_distribution) == 6
