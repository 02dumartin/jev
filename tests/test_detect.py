from jevgate.detect import normalize_label

CLASSES = ["fish", "crab", "shrimp", "jellyfish", "starfish"]


def test_normalize_label():
    assert normalize_label("##fish", CLASSES) == "fish"
    assert normalize_label("jellyfish", CLASSES) == "jellyfish"
    assert normalize_label("star", CLASSES) == "starfish"
    assert normalize_label("crab shrimp", CLASSES) == "shrimp"
    assert normalize_label("", CLASSES) == "object"
