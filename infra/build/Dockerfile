FROM python:3.13-slim-trixie

RUN apt-get update
RUN apt-get install openjdk-21-jdk --yes

WORKDIR /nova

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/nova/

COPY src/ src/
COPY configs/ configs/
COPY data/trx/ data/trx/
COPY data/telemetry/ data/telemetry/
COPY models/.placeholder models/.placeholder
