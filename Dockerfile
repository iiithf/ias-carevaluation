FROM python:3.7-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

ENV QUERY "127.0.0.1:9000"
CMD ["python", "main.py"]
