from jevgate.detect import normalize_label

P2C = {"fish": "fish", "small fish": "small_fish", "crab": "crab", "shrimp": "shrimp",
       "jellyfish": "jellyfish", "starfish": "starfish"}


def test_normalize_label():
    assert normalize_label("##fish", P2C) == "fish"
    assert normalize_label("small fish", P2C) == "small_fish"
    assert normalize_label("jellyfish", P2C) == "jellyfish"   # longer phrase wins over 'fish'
    assert normalize_label("star", P2C) == "starfish"
    assert normalize_label("crab shrimp", P2C) == "shrimp"
    assert normalize_label("", P2C) is None
