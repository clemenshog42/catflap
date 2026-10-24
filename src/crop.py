"""Geometry of the segmentation crop (shared by the pipeline, the evaluation and the overlay videos).

The face box of the detector is extended relative to its size: to the sides, upwards (negative = into the box)
and downwards, where the prey is carried. The values match the box shape of the greyscale face detector
cat_face.pt, so that the crops look like the crops the segmentation model was trained on.
"""

CROP_PAD = (0.56, -0.47, 1.06)  # (sides, top, bottom), relative to the width / height of the face box


def crop_region(box, img_w, img_h, pad=CROP_PAD):
    """Returns the crop rectangle (x1, y1, x2, y2), clipped to the image, and whether the extended box
    lies completely inside the image. If it does not, part of the area where prey would be is missing."""
    x1, y1, x2, y2 = map(float, box)
    fw, fh = x2 - x1, y2 - y1
    pw, pt, pb = pad
    ex1, ey1 = x1 - fw * pw, y1 - fh * pt
    ex2, ey2 = x2 + fw * pw, y2 + fh * pb
    inside = ex1 >= 0 and ey1 >= 0 and ex2 <= img_w and ey2 <= img_h
    cx1, cy1 = max(0, int(ex1)), max(0, int(ey1))
    cx2, cy2 = min(int(img_w), int(ex2)), min(int(img_h), int(ey2))
    cy1 = min(cy2 - 1, cy1)
    return (cx1, cy1, cx2, cy2), inside
