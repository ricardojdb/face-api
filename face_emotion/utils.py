from io import BytesIO
from PIL import Image

import numpy as np
import base64
import torch
import json
import six
import os 

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_img(data):
    return Image.open(BytesIO(base64.b64decode(data)))

def init_model(base_path):
    model_name = "senet50_ferplus_dag"
    model_def_path = os.path.join(base_path, model_name + '.py')
    weights_path = os.path.join(base_path, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    net.eval()
    net.to(device)
    return net

def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3
    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition
    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod

def compose_transforms(meta, resize=256, center_crop=True):
    """Compose preprocessing transforms for model
    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.
    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if center_crop:
        transform_list = [transforms.Resize(resize),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)

def preprocess(image_array):
    image = image_array.resize((224,224)) 
    image = np.asarray(image) - np.array([131.0912, 103.8827, 91.4953])
    image = np.transpose(image, (2,0,1))
    image = np.expand_dims(image, 0)
    image = torch.tensor(image, dtype=torch.float32).to(device)
    return image

def model_predict(data, model):
    image = decode_img(data)

    image = preprocess(image)

    predictions = model(image).cpu().detach().numpy()
    
    out = {"emotions":softmax(predictions).reshape(-1,).tolist()}

    return json.dumps(out)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
