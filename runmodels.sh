sudo docker build -t facedetect face_detect/ || echo
sudo docker run --runtime=nvidia --name=facedetect-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_detect:/app -p 7000:7000 facedetect

sudo docker build -t facemotion face_emotion/ ||  echo
sudo docker run --runtime=nvidia --name=facemotion-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_emotion:/app -p 7001:7000 facemotion

sudo docker build -t facefeatures face_features/ || echo
sudo docker run --runtime=nvidia --name=facefeatures-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_features:/app -p 7002:7000 facefeatures

sudo docker build -t vggface face_recognition/ || echo
sudo docker run --runtime=nvidia --name=vggface-service --rm \
-dit -v /home/deepai/Documents/show-room/docker_models/face_recognition:/app -p 7003:7000 vggface
