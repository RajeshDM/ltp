FROM pytorch/pytorch

# Copy source files
COPY . /container

# Setup extra dependencies
RUN pip3 install pytorch-lightning
RUN pip3 install tarski
RUN apt-get update
RUN apt-get -y install gringo

ENTRYPOINT ["python3", "/container/train.py"]
