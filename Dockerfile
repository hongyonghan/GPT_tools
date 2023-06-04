FROM python:3.10-slim
LABEL maintainer="hanhongyong"
RUN apt-get update \
    && apt-get install -y libpq-dev build-essential \
    && apt-get clean
RUN mkdir /app
RUN mkdir -p /app/db && mkdir -p /app/index
COPY requirements.txt /app/
COPY db/* /app/db
COPY index/* /app/index
COPY main.py /app/
COPY run.sh /app/
COPY theme.pptx /app/
COPY chroma-collections.parquet /app/
COPY chroma-embeddings.parquet /app/



RUN pip install --user --no-cache-dir -r /app/requirements.txt
WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["python","-u","main.py",">1.log"]