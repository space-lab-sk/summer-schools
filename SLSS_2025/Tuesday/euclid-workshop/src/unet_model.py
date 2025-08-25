"""
Tiny U-Net in Keras.

Goal
- Build a small U-Net with skip connections for 256x256x1 inputs.

Model sketch
- Encoder: Conv2D + BatchNorm + ReLU blocks with MaxPooling downsamples.
- Decoder: UpSampling (or Conv2DTranspose) with skip connections from encoder.
- Output: 1-channel logits (no sigmoid here; use `from_logits=True` in loss).
"""

def build_unet(input_shape=(256,256,1), base_filters=16):
    raise NotImplementedError
