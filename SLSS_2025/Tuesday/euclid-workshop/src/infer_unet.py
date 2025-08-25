"""
Batch inference for U-Net predictions.

Goal
- Load a trained model and run predictions on a directory of images.

Inference steps
- Load model from a checkpoint (e.g., `runs/best.keras`).
- Predict masks, apply sigmoid and threshold at 0.5.
- Save outputs as `<name>_pred.png` in the output directory.
"""

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
