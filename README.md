# Face API
Face API with Docker, Tensorflow and PyTorch.

### Models:
* Face Detection - TensorFlow ([SSD-Mobilenet](https://github.com/yeephycho/tensorflow-face-detection))
* Facial Emotion Recognition - PyTorch ([Resnet50](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html))
* Age and Gender Estimation - Keras ([WideResnet](https://github.com/Tony607/Keras_age_gender))
* Facial Recognition - PyTorch ([VGGFace2](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html))

### Requirements:
* Docker with Nvidia Support [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker)  
* CUDA 9 [Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)


### How to use:
In order to start the API the user must run the docker containers using the `docker-compose` command. 

To build the docker images use:
```bash
docker-compose -f dockers/docker-compose.yml build
```

To run the docker containers use:
```bash
docker-compose -f dockers/docker-compose.yml up
```

The user can access the API using the following URL:
```
http://localhost:7000/predict/
```

Each model is hosted in a different port e.g. `7001`, `7002`, `7003`.

### Test:
Change the `host` variable inside the `test_api.py` script with the IP in which the dockers are running and then run the following:
```bash
python test_api.py
```

### Run the demo:
This demo calls the API to make predictions using the webcam. 

Make sure the docker containers are running and change the `host` variable with the desired IP. 

Run the demo using:
```bash
python demo.py
```

