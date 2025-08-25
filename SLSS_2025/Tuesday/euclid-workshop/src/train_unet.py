"""
Keras training loop for U-Net.

Goal
- Train the tiny U-Net using images and masks from `utils.dataset.make_dataset`.

Training specifics
- Loss: BinaryCrossentropy with `from_logits=True`.
- Metric: Dice (and optionally IoU).
- Save best model checkpoint to `runs/best.keras`.
"""

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
