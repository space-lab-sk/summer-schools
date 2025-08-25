"""
Traditional (classical) image segmentation using OpenCV.

Goal
- Create masks with OpenCV from galaxy cutouts.

Steps
- Convert to grayscale → threshold (Otsu/adaptive/fixed) → morphology cleanup.
- Optional blur and small-component removal.
- Save masks as <name>_mask.png in the output directory.
"""

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
