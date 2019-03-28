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

### Test:
Change the `host` variable inside the `test_api.py` script with the IP in which the dockers are running and then run the following:
```
python test_api.py
```

### Run the demo:
This demo calls the API to make predictions using the webcam. 

Make sure the docker containers are running and change the `host` variable with the desired IP. 

Run the demo using:
```
python demo.py
```

