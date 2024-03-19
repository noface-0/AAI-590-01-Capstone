apt-get update -y && apt-get install -y \
    cmake \
    libgeos-dev \
    wget \
    build-essential \
    git \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    swig \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /ta-lib-0.4.0-src.tar.gz /ta-lib

pip3 install --user -r requirements.txt;