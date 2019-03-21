from models.resnet50_ft_dag import resnet50_ft_dag
from torchvision import transforms
from io import BytesIO
from PIL import Image

import numpy as np
import torchvision
import torch
import base64
import json
import os 

global device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    
    model = resnet50_ft_dag(path)
    model.to(device)
    model.eval()

    return model

def decode_img(data):
    return Image.open(BytesIO(base64.b64decode(data)))

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

def normalize_image(img, mean, std):
    x = np.array(img)

    for i in range(3):
        x[:,:,i] = (x[:,:,i] - mean[i]) / std[i]

    return Image.fromarray(x).convert("RGB")


def preprocess(x, meta):
    x = compose_transforms(meta)(x)
    x = x.unsqueeze(0).to(device)
    return x

def predict_embed(x, model):
    embed, _  = model(x)
    
    if embed.is_cuda:
        embed = embed.cpu()
    embed = np.reshape(embed.detach().numpy(), (-1, 1))
    
    return embed

def who_is_it(preds, database):
    min_dist = 1e4
    name = ''
    for k, emb in database.items():
        dist = np.linalg.norm(preds - emb)
        if dist < min_dist:
            name = k
            min_dist = dist

    if min_dist > 210:
        name = 'ID-' + str(len(database.keys())+1)
        database[name] = preds

    return name, min_dist

def model_predict(data, model, database):
    img = decode_img(data)

    x = preprocess(img, model.meta)
    
    preds = predict_embed(x, model)
    
    label, dist = who_is_it(preds, database)

    out = {'label':label, 'dist':'{:.3f}'.format(dist)}
    
    return json.dumps(out)


def init_dataset(dataset_path, model):
    database = {}

    for path in os.listdir(dataset_path):
        x = Image.open(dataset_path+path).convert("RGB")        
        x = preprocess(x, model.meta)
        
        embed = predict_embed(x, model)
        database[path[:-4]] = embed

    return database
