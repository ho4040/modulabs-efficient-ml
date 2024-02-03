FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

COPY ./app /app
WORKDIR /app

# Install wget before using it to download the package
RUN apt-get update && \
    apt-get install -y wget curl

RUN wget https://developer.download.nvidia.com/compute/cusparselt/0.5.2/local_installers/cusparselt-local-repo-ubuntu2004-0.5.2_1.0-1_amd64.deb && \
    dpkg -i cusparselt-local-repo-ubuntu2004-0.5.2_1.0-1_amd64.deb && \
    cp /var/cusparselt-local-repo-ubuntu2004-0.5.2/cusparselt-local-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install libcusparselt0 libcusparselt-dev

# # Set DEBIAN_FRONTEND to noninteractive to avoid interactive prompts
# # Optionally set the TZ environment variable to your timezone
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Etc/UTC

# # Python 3.10 설치를 위한 명령어 추가
# RUN apt-get install -y software-properties-common && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.10 python3.10-venv python3.10-dev

# # Python 3.10을 기본 python 명령어로 설정 (선택사항)
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
#     update-alternatives --set python /usr/bin/python3.10

# # Python 3.10의 pip 설치
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python3.10 get-pip.py && \
#     rm get-pip.py

# # 필요한 Python 패키지 설치 (예시)
# RUN python3.10 -m pip install --upgrade pip && \
#     python3.10 -m pip install -r requirements.txt


# # Reset DEBIAN_FRONTEND
# ENV DEBIAN_FRONTEND=dialog

RUN nvcc spmm.cu -I/usr/include -L/usr/lib/x86_64-linux-gnu -lcusparse -lcusparseLt -ldl -o spmm

CMD ["./spmm"]
