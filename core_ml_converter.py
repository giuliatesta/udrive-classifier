import coremltools as ct
from keras.src.saving.saving_api import load_model

from main import MODEL_PATH

# Convert the model using the Unified Conversion API to an ML Program
model = ct.convert(
    load_model(f'{MODEL_PATH}/cnn.keras'),
    classifier_config=ct.ClassifierConfig([0, 1, 2, 3]),
    convert_to="neuralnetwork",
)

model.author = 'Giulia Testa'
model.version = "1.0.0"

model.save(f'{MODEL_PATH}/udrive_classifer.mlpackage')
