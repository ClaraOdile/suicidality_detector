FROM python:3.10.6-buster
RUN mkdir suicidality_detector
COPY suicidality_detector /suicidality_detector
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn suicidality_detector.api.fast:app --host 0.0.0.0 --port $PORT
