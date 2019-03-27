# Use pytorch image with Cuda for GPU support
# docker build -t facemotion . 
FROM pytorch/pytorch:0.4_cuda9_cudnn7

# LABEL "docker run --runtime=nvidia --name=facemotion-service --rm -v /home/deepai/Documents/show-room/docker_models/face_emotion:/app -p 7000:7000 facemotion"

# Create app folder
RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Copy the requiremets.txt file into the container at /app
COPY requirements.txt /app

# Local directory as a volume
VOLUME /home/deepai/Documents/show-room/docker_models/face_emotion:/app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]