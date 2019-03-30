input_addr=$1
model_addr=$2

echo "accessing input stream at ${input_addr} ..."
examples=$(curl ${input_addr})

echo "sending input to model at ${model_addr} ..."
classes=$(curl --request POST ${model_addr}/v1/models/model:classify --data "${examples}")

echo "recieved output from model:"
echo ${classes}
