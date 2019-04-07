FROM python:3.7-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

EXPOSE 8000
ENV QUERY "127.0.0.1:9000"
CMD ["python", "main.py"]
