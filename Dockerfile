FROM python:3.10.6-buster
# FROM tensorflow/tensorflow:2.10.0
RUN mkdir suicidality_detector
COPY suicidality_detector /suicidality_detector
COPY requirements_prods.txt /requirements.txt
COPY setup.py /setup.py
RUN pip install . 
# RUN pip install -r requirements.txt
CMD uvicorn suicidality_detector.api.fast:app --host 0.0.0.0 --port $PORT