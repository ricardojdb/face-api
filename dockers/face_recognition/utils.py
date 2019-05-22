from models.resnet50_ft_dag import resnet50_ft_dag
from io import BytesIO
from PIL import Image

import numpy as np
import torch
import base64
import json
import os


class FaceRecognition(object):
    def __init__(self, model_path, dataset_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = self.init_model()
        self.database = {}
        self.init_database()

    def init_model(self):
        model_name = "resnet50_ft_dag.pth"
        model = resnet50_ft_dag(os.path.join(self.model_path, model_name))
        model.to(self.device)
        model.eval()
        return model

    def init_database(self):
        for path in os.listdir(self.dataset_path):
            try:
                x = Image.open(
                    os.path.join(self.dataset_path, path)).convert("RGB")
            except:
                continue
            x = self.preprocess(x, self.model.meta)
            embed = self.predict_embed(x)

            self.database[path[:-4]] = embed

    def decode_img(self, data):
        return Image.open(BytesIO(base64.b64decode(data)))

    def preprocess(self, x, meta):
        img = x.resize((224, 224))
        img = np.array(img, dtype=np.float32)
        img = img[:, :, ::-1]
        img -= meta["mean"]
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img.copy())
        return img.unsqueeze(0).to(self.device)

    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(
            np.maximum(np.sum(np.square(x),
                       axis=axis, keepdims=True), epsilon))
        return output

    def predict_embed(self, x):
        embed, _ = self.model(x)

        if embed.is_cuda:
            embed = embed.cpu()

        embed = np.reshape(embed.detach().numpy(), (-1,))

        return self.l2_normalize(embed)

    def who_is_it(self, preds):
        min_dist = 1e4
        name = ''
        for k, emb in self.database.items():
            dist = np.linalg.norm(preds - emb)
            if dist < min_dist:
                name = k
                min_dist = dist

        if min_dist > 0.9:
            name = 'ID-' + str(len(self.database.keys())+1)
            self.database[name] = preds

        return name, min_dist

    def model_predict(self, data):
        img = self.decode_img(data)
        img = self.preprocess(img, self.model.meta)

        preds = self.predict_embed(img)
        label, dist = self.who_is_it(preds)

        out = {'label': label, 'dist': '{:.3f}'.format(dist)}

        return json.dumps(out)
