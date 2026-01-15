docker run --rm -it \
  --name lcm-sr-ui \
  --network host \
  --privileged \
  -e PORT=4200 \
  -e NUM_WORKERS=1 \
  -e QUEUE_MAX=8 \
  -e MODEL_ROOT=/models \
  -v "$PWD/model:/models:ro,Z" \
  lcm-sr-ui:latest
