import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def crop_limb(img, limb_mask, bg_color=None):
    """
    Crops image background according to provided mask.
    :param PIL.Image img: image to crop
    :param PIL.Image limb_mask: binary mask
    :param list bg_color: background color of cropped image
    :return PIL.Image: cropped image
    """
    if bg_color is None:
        bg_color = [255, 255, 255]
    mask = np.array(limb_mask)
    img = np.array(img)
    img[(mask[:, :, 0:3] != [255, 255, 255]).any(2)] = bg_color
    cropped = Image.fromarray(img)
    return cropped


def rotate_imgs(imgs_list, extension="png", path=None):
    """
    Rotates imgs in 90, 180 and 270 degrees and saves them to file.
    :param list imgs_list: list of paths to img provided by by glob module
    :param str extension: extension of images
    :param str path: path where to save images, if None img is save to same path as original image
    :return:
    """
    angles = [90, 180, 270]
    for img in tqdm(imgs_list):
        fname = img.replace(f".{extension}", "").split("\\")[-1]
        if path is None:
            path = img.replace(f".{extension}", "").split("\\")[0]
        img = Image.open(img)
        for angle in angles:
            img = img.rotate(angle)
            img.save(f"{path}/{fname}_r{angle}.{extension}")


def create_limb_mask(img_size=(4096, 4096), radius=1645, name="limb_mask.png"):
    """
    Create limb mask according to provided radius of sun.
    :param tuple img_size: size of sun image
    :param int radius: solar radius
    :param str name: name of limb mask file
    :return:
    """
    limb_mask = Image.new("RGB", img_size, (0, 0, 0))
    draw = ImageDraw.Draw(limb_mask)
    draw.arc(
        [
            img_size[0] // 2 - radius,
            img_size[1] // 2 - radius,
            img_size[0] // 2 + radius,
            img_size[1] // 2 + radius,
        ],
        start=0,
        end=360,
        fill=(255, 255, 255),
        width=img_size[0] // 2,
    )
    limb_mask.save(name)


def create_labels_img(coord_list, name, flip=True):
    """
    Creates black image with white event areas according to provided coordinates.
    :param list coord_list: list containing list of tuples with x and y coordinates [[(x,y), (x,y)]]
    :param name: name of limb mask file
    :param bool flip: whether to flip image
    :return:
    """
    blank_img = Image.new("RGB", (4096, 4096), (0, 0, 0))
    img = ImageDraw.Draw(blank_img)
    for annotation in coord_list:
        try:
            img.polygon(annotation, fill=(255, 255, 255))
        except TypeError:
            pass
    if flip:
        blank_img = blank_img.transpose(method=Image.FLIP_TOP_BOTTOM)
    blank_img.save(name)
