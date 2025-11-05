# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.11

FROM python:${PYTHON_VERSION}-slim

LABEL fly_launch_runtime="flask"

WORKDIR /code

# Install PyTorch CPU-only first
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


COPY . .

EXPOSE 8080

CMD [ "python3", "app.py"]
