FROM ubuntu:20.04

# parts of the Dockerfile were adapted from unikrn/python-opencv

RUN echo 'deb http://security.ubuntu.com/ubuntu focal-security main' >> /etc/apt/sources.list
RUN echo 'deb http://security.ubuntu.com/ubuntu xenial-security main' >> /etc/apt/sources.list
RUN apt update --fix-missing && apt-get install tzdata -qy &&\
	apt install -qy \
	cmake \
	x11-apps vainfo git\
	python3-numpy python3-scipy python3-pip python3-setuptools \
	wget \
	xauth \
	libjpeg-dev libtiff5-dev libjasper1 libjasper-dev libpng-dev libavcodec-dev libavformat-dev \
	libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libatlas-base-dev \
	libv4l-0 libavcodec-dev libavformat-dev libavutil-dev ffmpeg \
	libswscale-dev libavresample-dev \
    libgstreamer1.0-dev \
    libvdpau-va-gl1 vdpauinfo \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad\
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-libav gstreamer1.0-vaapi gstreamer1.0-tools libavcodec-dev \
	gfortran python3-dev build-essential pkg-config &&\
	apt upgrade -qy &&\
	apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* &&\
# Build OpenCV 3.4.12
	cd /root && \
	wget -q https://github.com/opencv/opencv/archive/3.4.12.tar.gz -O opencv.tar.gz && \
	tar zxf opencv.tar.gz && rm -f opencv.tar.gz && \
	wget -q https://github.com/opencv/opencv_contrib/archive/3.4.12.tar.gz -O contrib.tar.gz && \
	tar zxf contrib.tar.gz && rm -f contrib.tar.gz && \
	cd opencv-3.4.12 && mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib-3.4.12/modules \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D BUILD_DOCS=OFF \
	-D BUILD_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_opencv_python3=ON \
	-D WITH_1394=OFF \
	-D WITH_MATLAB=OFF \
	-D WITH_OPENCL=OFF \
	-D WITH_OPENCLAMDBLAS=OFF \
	-D WITH_OPENCLAMDFFT=OFF \
	-D WITH_GSTREAMER=ON \
	-D WITH_FFMPEG=ON \
	-D CMAKE_CXX_FLAGS="-O3 -funsafe-math-optimizations" \
	-D CMAKE_C_FLAGS="-O3 -funsafe-math-optimizations" \
	.. && make -j $(nproc) && make install && \
	cd /root && rm -rf opencv-3.4.12 opencv_contrib-3.4.12 &&\
# Remove temporary packages except the ones needed by opencv
	apt purge -qy \
	build-essential \
	libpng12-dev \
	libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libatlas-base-dev \
	libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev \
	libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libatlas-base-dev \
	libavcodec-dev libavformat-dev libavutil-dev \
	libswscale-dev libavresample-dev \
    libgstreamer1.0-dev \
    python3-dev build-essential pkg-config \
	gfortran pkg-config cmake && \
	apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install scikit-learn pandas

WORKDIR densetrack
COPY . .
RUN pip3 install --upgrade numpy
RUN python3 setup.py install
ENTRYPOINT python3 -u run_idt.py
