# Use pytorch with cuda to get GPU support
# docker build -t vggface .
FROM pytorch/pytorch:0.4_cuda9_cudnn7

# LABEL 'docker run --runtime=nvidia --name=vggface-service --rm -v /home/deepai/Documents/show-room/docker_models/face_recognition:/app -p 7000:7000 vggface'

# Create app folder
RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app

# Local directory as a volume
VOLUME /home/deepai/Documents/show-room/docker_models/face_recognition:/app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]