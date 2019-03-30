Tensorflow serving model for Car Evaluation dataset.

```bash
docker pull tensorflow/serving
docker run -p 8501:8501 \
  --mount type=bind,source=$PWD,target=/models/model \
  -e MODEL_NAME=model -t tensorflow/serving
```
