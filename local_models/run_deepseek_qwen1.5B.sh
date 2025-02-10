docker run -it --rm --gpus all \
  -p 9000:9000 \
  --pull always \
  -e DOCKER_PULL=always \
  dustynv/mlc:r36.4.0 \
    sudonim serve \
      --model dusty-nv/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_ft-MLC \
      --quantization q4f16_ft \
      --max-batch-size 1 \
      --chat-template deepseek_r1_qwen \
      --host 0.0.0.0 \
      --port 9000