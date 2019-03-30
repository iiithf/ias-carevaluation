Generate and use (REST, gRPC) Tensorflow serving model for Car Evaluation dataset.


## setup

```bash
# build model
python3 dev.py

# deploy with tensorflow serving
docker pull tensorflow/serving
docker run -p 8501:8501 \
  --mount type=bind,source=$PWD/build,target=/models/model \
  -e MODEL_NAME=model -t tensorflow/serving

# test with client
python3 client.py
python3 client_grpc.py
```


Time to make models!
