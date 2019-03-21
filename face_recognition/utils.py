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

def preprocess(x, meta):
    img = x.resize((224,224))
    img = np.array(img, dtype=np.float32)
    img = img[:,:,::-1]
    img -= meta["mean"]
    img = img.transpose(2, 0, 1)  # C x H x W
    print(img.shape)
    img = torch.from_numpy(img.copy())

    return img.unsqueeze(0).to(device)

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def predict_embed(x, model):
    embed, _  = model(x)
    
    if embed.is_cuda:
        embed = embed.cpu()
    embed = np.reshape(embed.detach().numpy(), (-1,))
    
    return l2_normalize(embed)

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
    img = preprocess(img, model.meta)

    preds = predict_embed(img, model)   
    label, dist = who_is_it(preds, database)

    out = {'label':label, 'dist':'{:.3f}'.format(dist)}
    
    return json.dumps(out)


def init_dataset(dataset_path, model):
    database = {}

    for path in os.listdir(dataset_path):
        try:
            x = Image.open(dataset_path+path).convert("RGB")
        except:
            continue
        x = preprocess(x, model.meta)
        
        embed = predict_embed(x, model)
        database[path[:-4]] = embed

    return database
