# mypy: disable-error-code=no-untyped-def
from splifft.models import ModelMetadata


def my_model_metadata():
    from .my_model import MyModel, MyModelConfig

    return ModelMetadata(model_type="my_model", config=MyModelConfig, model=MyModel)
