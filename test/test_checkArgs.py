from RAiDER.checkArgs import (
    modelName2Module
)

def test_model2module():
    model_module_name, model_obj = modelName2Module('ERA5')
    assert model_obj().Model() == 'ERA-5'
