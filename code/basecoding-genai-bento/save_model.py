import bentoml
from basecoding_model import predict

# `save_model` saves a given python object or function
saved_model = bentoml.picklable_model.save_model(
    "my_python_model", predict, signatures={"__call__": {"batchable": True}}
)
print(f"Model saved: {saved_model}")