FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && apt-get install -y curl git vim \
    && pip install schedule \
    && pip install pandas \
    && pip install numpy \
    && pip install requests \
    && pip install pytz

COPY . /home
WORKDIR /home

RUN chmod +x /home/start.sh
CMD ["./start.sh"]
