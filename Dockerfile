FROM python:3.6-slim
LABEL maintainer="mr.ilyakos@gmail.com"
LABEL project="cryogel"
  
#RUN apt-get update && \
#	apt-get -y install \
#	gcc make apt-transport-https ca-certificates build-essential

RUN python3 --version
RUN pip3 --version

WORKDIR  /usr/src/cryo-client

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /src/
RUN ls -la /src/*
RUN mkdir -p /root/shared/results

CMD ["python3", "/src/main.py"]

