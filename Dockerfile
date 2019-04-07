FROM tensorflow/serving
WORKDIR /models/model
COPY . /models/model
EXPOSE 8500
EXPOSE 8501
ENV MODEL_NAME "model"
