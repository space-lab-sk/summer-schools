"""
tf.data input pipeline helpers.

Goal
- Load grayscale images and masks, resize to 256x256, batch them.

Pipeline steps
- Pair images with corresponding *_mask.png files.
- Load as float32 in [0,1].
- Resize to 256x256.
- Return (image, mask) batches ready for Keras training.
"""

def make_dataset(images_dir, masks_dir, img_size=(256,256), batch_size=4, shuffle=True):
    raise NotImplementedError
