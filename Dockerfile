FROM pytorch/pytorch

COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc ffmpeg libsm6 libxext6
RUN pip install Cython
RUN pip install -r requirements.txt

#ENTRYPOINT jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root