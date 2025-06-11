from splifft.models import ModelMetadata


def test_model_metadata_dynamically_loaded() -> None:
    model_metadata = ModelMetadata.from_module(
        "splifft.models.bs_roformer",
        "BSRoformer",
        model_type="bs_roformer",
    )
    from splifft.models.bs_roformer import BSRoformer, BSRoformerConfig

    model_metadata_expected = ModelMetadata(
        model_type="bs_roformer",
        config=BSRoformerConfig,
        model=BSRoformer,
    )
    assert model_metadata == model_metadata_expected
