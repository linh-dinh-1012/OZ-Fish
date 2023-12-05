FROM python:3.8-buster

COPY requirements.txt /requirements.txt
COPY best.pt /best.pt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY fast.py /fast.py

CMD uvicorn fast:app --host 0.0.0.0 --port 8080
