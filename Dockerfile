FROM pytorch/pytorch

COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc ffmpeg libsm6 libxext6
RUN pip install Cython
RUN pip install -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY . .

CMD ["python", "app.py"]