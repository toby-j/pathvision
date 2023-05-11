FROM python:3.10.10

WORKDIR /pathvision

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install --upgrade pip

# Install Python dependencies
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

RUN pip install .

WORKDIR /pathvision/pathvision/models/detectron2

RUN pip install .

WORKDIR /pathvision

CMD ["python", "main.py"]