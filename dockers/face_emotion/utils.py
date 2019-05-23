from io import BytesIO
from PIL import Image

import numpy as np
import base64
import torch
import json
import six
import os


class FaceEmotion(object):
    """
    Initializes and handles de face emotion model in PyTorch
    """
    def __init__(self, model_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self.init_model()

    def decode_img(self, encoded_data):
        """Decodes the encoded data comming from a request.

        Args:
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            array: Data decoded into a usable format.

        """
        return Image.open(BytesIO(base64.b64decode(encoded_data)))

    def init_model(self):
        """Initializes the machine learning model.

        Returns:
            model (object): Loaded pre-trained model used
                to make predictions.

        """
        model_name = "senet50_ferplus_dag"
        model_def_path = os.path.join(self.model_path, model_name + '.py')
        weights_path = os.path.join(self.model_path, model_name + '.pth')
        mod = self.load_module_2or3(model_name, model_def_path)
        func = getattr(mod, model_name)
        net = func(weights_path=weights_path)
        net.eval()
        net.to(self.device)
        return net

    def load_module_2or3(self, model_name, model_def_path):
        """Load model definition module in a manner that is compatible with
        both Python2 and Python3
        Args:
            model_name: The name of the model to be loaded
            model_def_path: The filepath of the module
                            containing the definition
        Return:
            The loaded python module."""
        if six.PY3:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                model_name, model_def_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            import importlib
            dirname = os.path.dirname(model_def_path)
            sys.path.insert(0, dirname)
            module_name = os.path.splitext(os.path.basename(model_def_path))[0]
            mod = importlib.import_module(module_name)
        return mod

    def preprocess(self, image_array):
        """Prerocess the data into the right format
        to be feed in to the given model.

        Args:
            raw_data (array): Raw decoded data to be processed.

        Returns:
            array: The data ready to use in the given model.

        """
        image = image_array.resize((224, 224))
        image = np.asarray(image) - np.array([131.0912, 103.8827, 91.4953])
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        return image

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def model_predict(self, encoded_data):
        """Decodes and preprocess the data, uses the
        pretrained model to make predictions and
        returns a well formatted json output.

        Args
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            json: A response that contains the output from
                the pre-trained model.
        """
        image = self.decode_img(encoded_data)

        image = self.preprocess(image)
        predictions = self.model(image)
        if predictions.is_cuda:
            predictions = predictions.cpu()

        predictions = predictions.detach().numpy()
        out = {"emotions": self.softmax(predictions).reshape(-1,).tolist()}

        return json.dumps(out)
