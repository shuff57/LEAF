import pytest

from app.labels import build_class_names, extract_state_dict, read_label_lines


def test_sorted_string_order_not_numeric():
    # ids sort as strings: "100" < "20" < "3"
    species = {"3": "c", "20": "b", "100": "a"}
    assert build_class_names(species, None, 3) == ["a", "b", "c"]


def test_explicit_mapping_wins():
    species = {"3": "c", "20": "b", "100": "a"}
    mapping = {"0": "3", "1": 20, "2": "100"}  # values may be ints or strings
    assert build_class_names(species, mapping, 3) == ["c", "b", "a"]


def test_count_mismatch_is_an_error():
    with pytest.raises(ValueError, match="class_idx_to_species_id"):
        build_class_names({"1": "a", "2": "b"}, None, 3)


def test_incomplete_mapping_is_an_error():
    with pytest.raises(ValueError, match="missing key"):
        build_class_names({"1": "a"}, {"0": "1"}, 2)


def test_extract_state_dict_variants():
    plain = {"fc.weight": 1, "conv1.weight": 2}
    assert extract_state_dict(plain) == plain
    assert extract_state_dict({"epoch": 3, "model": plain}) == plain
    wrapped = {"module.fc.weight": 1, "module.conv1.weight": 2}
    assert extract_state_dict({"state_dict": wrapped}) == plain


def test_extract_state_dict_rejects_non_resnet():
    with pytest.raises(ValueError, match="fc.weight"):
        extract_state_dict({"head.weight": 1})
    with pytest.raises(ValueError):
        extract_state_dict([1, 2, 3])


def test_read_label_lines(tmp_path):
    f = tmp_path / "labels.txt"
    f.write_text("# comment\n\nQuercus robur\n  Rosa canina  \n")
    assert read_label_lines(f) == ["Quercus robur", "Rosa canina"]
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(ValueError):
        read_label_lines(empty)
