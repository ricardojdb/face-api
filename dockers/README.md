## Docker API

Here are the docker folders used to create the images containing the flask APIs.

### How to use:

* Download the models and move them inside the `/models` folders inside each directory.
* Add the images of the faces you'd like to recognize into the `/face_recognition/images` folder.

Build and run the docker images using:
```
bash runmodels.sh
```

To stop the docker containers use:

```
bash stopmodels.sh
```