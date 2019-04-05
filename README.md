Generate and use (REST, gRPC) Tensorflow serving model for Car Evaluation dataset.

```bash
# build model
python3 dev.py

# deploy with tensorflow serving
docker pull tensorflow/serving
docker run -p 8501:8501 \
  --mount type=bind,source=$PWD,target=/models/model \
  -e MODEL_NAME=model -t tensorflow/serving

# test with client
python3 client.py
python3 client_grpc.py

# host dataset as input service
python3 input.py

# use input service and model for inference
# <input address> <model address>
./inference.sh 127.0.0.1:1993 127.0.0.1:8501
```

Time to make models!
