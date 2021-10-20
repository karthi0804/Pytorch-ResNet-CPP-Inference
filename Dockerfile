ARG baseimage=ubuntu:18.04
FROM $baseimage

# installing basic softwares with root privilege
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	git \
	g++-8 \
	htop \
	iputils-ping \
	libglib2.0-0 \
	libsm6 \	
	libxext6 \
	libxrender-dev\
	net-tools \
	ninja-build \
	openssh-client \
	software-properties-common \
	sudo \
	unzip \
	vim \
	wget
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7 && \ 
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
RUN unzip libtorch-shared-with-deps-latest.zip 
RUN git clone https://github.com/opencv/opencv.git
WORKDIR opencv
RUN git checkout 3.3.1
RUN mkdir build
WORKDIR build
RUN cmake -GNinja .. && \
	ninja && \
	ninja install
WORKDIR /
CMD /bin/bash
