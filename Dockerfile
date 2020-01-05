FROM tensorflow/tensorflow:latest-gpu-py3

COPY . /api

# install python
RUN apt-get update -y\
 && apt-get install -y python3 python3-pip

# install ML deps
RUN pip3 install keras==2.2.4 numpy==1.16.1 matplotlib Pillow torchvision

# install web deps
RUN pip3 install flask flask_cors

EXPOSE 8888
WORKDIR /api
CMD python3 app.py

