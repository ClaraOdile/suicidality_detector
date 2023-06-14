FROM python:3.10.6-buster
# FROM tensorflow/tensorflow:2.10.0
RUN mkdir suicidality_detector
COPY . /suicidality_detector
WORKDIR /suicidality_detector
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN pip install --upgrade pip
# RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
RUN pip install . 
CMD uvicorn suicidality_detector.api.fast:app --host 0.0.0.0 --port $PORT