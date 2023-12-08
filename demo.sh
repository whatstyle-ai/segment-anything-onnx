#!/bin/bash

cd "$(dirname "$0")"

python -m segment_anything_onnx.inference \
    --encoder_model ./models/sam_vit_l_0b3195.encoder.onnx \
    --decoder_model ./models/sam_vit_l_0b3195.decoder.onnx \
    --image ./examples/laura.jpg \
    --prompt ./examples/laura_prompt.json \
    --output ./output/laura-L.png \
    --show