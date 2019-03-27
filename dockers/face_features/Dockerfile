# Use opencv image to have all the dependencies ready
# docker build -t facefeatures . 
FROM tensorflow/tensorflow:1.11.0-gpu-py3

# Run script
# LABEL "docker run --runtime=nvidia --name=facefeatures-service --rm -v /home/deepai/Documents/show-room/docker_models/face_features:/app -p 7000:7000 facefeatures"

# Create app/ directory
RUN mkdir /app

# Setting working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app

# Local directory as a volume
VOLUME /home/deepai/Documents/show-room/docker_models/face_features:/app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]