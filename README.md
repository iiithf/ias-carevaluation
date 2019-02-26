# ias-carevaluation

Car Evaluation client and server for Hackathon.


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
```


time to sleep!
