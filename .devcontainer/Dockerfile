FROM python:3.9-bullseye

ENV SHELL=/bin/bash

ARG USERNAME=my_app
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update \
    && apt-get install -y sudo ca-certificates build-essential ffmpeg libsm6 libxext6 software-properties-common curl gnupg \
    && apt-get install -y git vim curl gettext tmux \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean 

WORKDIR /my_app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/primestyle-co/dotfiles /my_app/.dotfiles \
    && python3 /my_app/.dotfiles/install.py

RUN git clone https://github.com/THU-MIG/yolov10.git /my_app/app/yolov10 \
    && pip install -r /my_app/app/yolov10/requirements.txt \
    && pip install -e /my_app/app/yolov10

COPY . .

RUN chown -R $USERNAME:$USERNAME /my_app

USER $USERNAME
SHELL ["/bin/bash", "-ec"]

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]