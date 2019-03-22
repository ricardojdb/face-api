#!/bin/bash

if [[ "$(sudo docker images -q facedetect:latest 2> /dev/null)" != "" ]]
then
echo Image facedetect already built
else
sudo docker build -t facedetect face_detect/
fi

if [[ "$(sudo docker images -q facemotion:latest 2> /dev/null)" != "" ]]
then
echo Image facemotion already built
else
sudo docker build -t facemotion face_emotion/ 
fi

if [[ "$(sudo docker images -q facefeatures:latest 2> /dev/null)" != "" ]]
then
echo Image facefeatures already built
else
sudo docker build -t facefeatures face_features/
fi

if [[ "$(sudo docker images -q vggface:latest 2> /dev/null)" != "" ]]
then
echo Image vggface already built
else
sudo docker build -t vggface face_recognition/
fi

sudo docker run --runtime=nvidia --name=facedetect-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_detect:/app -p 7000:7000 facedetect

sudo docker run --runtime=nvidia --name=facemotion-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_emotion:/app -p 7001:7000 facemotion

sudo docker run --runtime=nvidia --name=facefeatures-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_features:/app -p 7002:7000 facefeatures

sudo docker run --runtime=nvidia --name=vggface-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_recognition:/app -p 7003:7000 vggface
