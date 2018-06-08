FROM tensorflow/tensorflow:latest-py3
WORKDIR /app
ADD . /app
RUN apt update \
    && apt install -y build-essential libsm6 libxext6 libxrender1
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install -e .
CMD ["python", "demo.py"]
