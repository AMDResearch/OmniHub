FROM rocm/pytorch:latest

ADD Docker/requirements.txt /home


WORKDIR /home
EXPOSE 8888

ARG UID
ARG GID
ARG DEBIAN_FRONTEND=noninteractive

# Set the locale

RUN apt-get update && apt-get install -y locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

RUN chmod 777 /home && \
    apt update && \
    apt install -y sudo cmake && \
    apt install glibc-source &&\
    addgroup --gid $GID nonroot && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" nonroot && \
    echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && \
    groupmod -g 15002 render && \     
    groupmod -g 15003 video  
USER nonroot

RUN git clone https://github.com/ROCm/bitsandbytes.git &&\
    cd bitsandbytes &&\
    git checkout rocm_enabled &&\
    pip install -U pip &&\
    pip install -r requirements-dev.txt &&\
    cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx908;gfx90a;gfx942" -S . &&\
    make &&\
    pip install .


# RUN wget https://github.com/ROCm/omniperf/releases/download/v2.0.0-RC1/omniperf-2.0.0-RC1.tar.gz &&\
#     tar xfz omniperf-2.0.0-RC1.tar.gz &&\
#     mkdir omniperf_install &&\
#     cd omniperf-2.0.0-RC1 &&\
#     export INSTALL_DIR=/home/omniperf_install &&\
#     python3 -m pip install -t ${INSTALL_DIR}/python-libs -r requirements.txt &&\
#     mkdir build &&\
#     cd build &&\
#     cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/2.0.0 -DPYTHON_DEPS=${INSTALL_DIR}/python-libs -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles ..  &&\
#     make install


# RUN wget https://github.com/ROCm/omnitrace/releases/download/v1.11.2/omnitrace-1.11.2-ubuntu-20.04-PAPI-OMPT-Python3.sh &&\
#     mkdir /opt/omnitrace &&\
#     chmod +x omnitrace-1.11.2-ubuntu-20.04-PAPI-OMPT-Python3.sh &&\
#     ./omnitrace-1.11.2-ubuntu-20.04-PAPI-OMPT-Python3.sh --prefix=/opt/omnitrace --exclude-subdir

# install llama-recipes

RUN git clone https://github.com/meta-llama/llama-recipes.git &&\
    cd llama-recipes &&\
    pip install -U pip setuptools &&\
    pip install -e .

# # mkdir /opt/omnitrace
# # CMD ["jupyter", "notebook", " — ip='0.0.0.0'", " — port=8888", " — no-browser", " — allow-root"]

# -v /share2/jsantos:/share/jsantos -v /share2/aaji:/share/aaji:ro -v /home/jsantos:/home/jsantos

docker run -p 8888:8888 -dit --name jsantos_pytorch \
 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
 -v /home/josantos:/home/jsantos \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --ipc=host --shm-size 8G jsantos_pytorch

# # docker exec -it jsantos_pytorch /bin/bash


# # docker build -f Docker/Dockerfile -t jsantos_pytorch --build-arg UID=$(id -u) --build-arg GID=$(id -g) .


RUN pip install -r requirements.txt



# SHELL ["/bin/bash", "-c"]

# CMD ["/bin/bash", "-l","-c", "echo 'source /opt/omnitrace/share/omnitrace/setup-env.sh' >> ~/.bashrc"]
# SHELL ["/bin/bash", "-c"]
# RUN . /opt/omnitrace/share/omnitrace/setup-env.sh


# ENV PATH="$PATH:/home/omniperf_install/2.0.0/bin"
# ENV PYTHONPATH="/home/omniperf_install/python-libs"


# COPY Docker/docker-entrypoint.sh /home/docker-entrypoint.sh

# ENTRYPOINT [ "/home/docker-entrypoint.sh" ]