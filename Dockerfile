FROM julia:1.9.3

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y wget

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# build miniconda for python venv
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash && \
    source ~/.bashrc && \
    conda install python=3.11 pip

# copy reqs to image and build
COPY ./requirements.txt /
RUN python -m pip install -r /requirements.txt


WORKDIR /code
COPY . /code
# initial run to precompile julia deps
RUN python run_error_experiment.py --help

# RUN echo 'cd /code' >> ~/.bashrc


# overwrite julia container's "CMD ['julia']"
CMD ["/bin/bash"]
