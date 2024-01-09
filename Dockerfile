FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

COPY ./app/ .
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]