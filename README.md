# Face API
Face API with Docker, Tensorflow and PyTorch.

### Models:
* Face Detection - TensorFlow ([SSD-Mobilenet](https://github.com/yeephycho/tensorflow-face-detection))
* Facial Emotion Recognition - PyTorch ([Resnet50](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#cross-modal-emotion))
* Age and Gender Estimation - Keras ([WideResnet](https://github.com/Tony607/Keras_age_gender))
* Facial Recognition - PyTorch ([VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/))

### Requirements:
* Docker with Nvidia Support [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker)  
* CUDA 9 [Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

### How to use:
Make sure the models and images are inside their respective folders.

Build and run the Dockers in parallel:
```
bash runmodels.sh
```

Stop Docker containers:
```
bash stopmodels.sh
```

### Test:

Test the APIs by running:
```
python test_api.py
```

### Demo:
This demo calls the API to make predictions using the webcam. 

To run the demo use:
```
python demo.py
```