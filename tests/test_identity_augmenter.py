"""Tests for augmentation/identity.py (STAGE2_CONTRACT IdentityAugmenter section)."""

from augmentation.identity import IdentityAugmenter
from data.base import DatasetRow


def test_identity_copies_x_verbatim_and_preserves_id_y_meta() -> None:
    row = DatasetRow(id="42", x="hello world", y=1, meta={"k": "v"})
    augmenter = IdentityAugmenter(variant="paraphrase", prompt_hash="abc123")
    result = augmenter.augment(row)

    assert result.id == "42"
    assert result.x == "hello world"
    assert result.y == 1
    assert result.meta == {"k": "v"}


def test_identity_sets_variant_and_augmenter_model() -> None:
    row = DatasetRow(id="1", x="text", y=0, meta={})
    augmenter = IdentityAugmenter(variant="idiomatic", prompt_hash="deadbeef")
    result = augmenter.augment(row)

    assert result.variant == "idiomatic"
    assert result.augmenter_model == "identity"


def test_identity_sets_nonempty_prompt_hash() -> None:
    row = DatasetRow(id="1", x="text", y=0, meta={})
    augmenter = IdentityAugmenter(variant="paraphrase", prompt_hash="deadbeefcafebabe")
    result = augmenter.augment(row)

    assert result.prompt_hash == "deadbeefcafebabe"
    assert result.prompt_hash != ""


def test_identity_exposes_variant_instance_attribute() -> None:
    augmenter = IdentityAugmenter(variant="idiomatic", prompt_hash="xyz")
    assert augmenter.variant == "idiomatic"
    assert augmenter.augmenter_model == "identity"


def test_identity_meta_copy_is_independent_of_source() -> None:
    """Mutating the result's meta must not mutate the source row's meta."""
    row = DatasetRow(id="1", x="text", y=0, meta={"k": "v"})
    augmenter = IdentityAugmenter(variant="paraphrase", prompt_hash="ph")
    result = augmenter.augment(row)
    result.meta["k"] = "mutated"
    assert row.meta == {"k": "v"}
