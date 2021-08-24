import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from metrics import dice_np, iou_np


def plot_imgs(imgs, masks, predictions=None, n_imgs=10):
    """
    Plots images, masks, prediction masks and overlays side by side. If predictions aren't provided, masks are used as
    overlays.
    :param numpy.array imgs: array of images
    :param numpy.array masks: array of masks
    :param numpy.array predictions: array of predictions
    :param int n_imgs: number of images to plot
    :return matplotlib.pyplot: pyplot of images side by side
    """
    if predictions is None:
        fig, axes = plt.subplots(n_imgs, 3, figsize=(12, n_imgs * 4), squeeze=False)
        overlap = masks
        ii = 2
    else:
        fig, axes = plt.subplots(n_imgs, 4, figsize=(16, n_imgs * 4), squeeze=False)
        axes[0, 2].set_title("Prediction", fontsize=22)
        overlap = predictions
        ii = 3

    # Set titles
    axes[0, 0].set_title("Image", fontsize=22)
    axes[0, 1].set_title("Mask", fontsize=22)
    axes[0, ii].set_title("Overlay", fontsize=22)

    # Plot imgs
    for i in range(n_imgs):
        # masked = np.ma.masked_where(overlap[i] == 0, overlap[i])
        zero = np.zeros((overlap[i].shape[0], overlap[i].shape[1]))
        one = overlap[i].reshape((overlap[i].shape[0], overlap[i].shape[1]))
        masked = np.stack((one, zero, zero, one), axis=-1)
        # Show imgs
        axes[i, 0].imshow(imgs[i], cmap="gray", interpolation=None)
        axes[i, 1].imshow(masks[i], cmap="gray", interpolation=None)
        axes[i, ii].imshow(imgs[i], cmap="gray", interpolation=None)
        axes[i, ii].imshow(masked, cmap="jet", alpha=0.5)

        if predictions is not None:
            axes[i, 2].imshow(predictions[i], cmap="gray", interpolation=None)
            # Show metrics - dice, iou
            dice = np.round(dice_np(y_true=masks[i], y_pred=predictions[i]), 4)
            iou = np.round(iou_np(y_true=masks[i], y_pred=predictions[i]), 4)
            axes[i, 3].text(
                0.1,
                0.9,
                f"Dice: {dice}\nIoU: {iou}",
                fontsize=15,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )
            axes[i, 3].set_axis_off()
        # Hide axis
        axes[i, 0].set_axis_off()
        axes[i, 1].set_axis_off()
        axes[i, 2].set_axis_off()
    return plt


def plot_metrics(model):
    """
    Plots training history of Keras model.
    :param model: trained model
    :return matplotlib.pyplot: pyplot of training history
    """
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    # Plot metrics
    axs[0].plot(model.history["iou"])
    axs[0].plot(model.history["dice"])
    axs[0].plot(model.history["val_iou"])
    axs[0].plot(model.history["val_dice"])
    axs[1].plot(model.history["loss"])
    axs[1].plot(model.history["val_loss"])
    # Set titles
    axs[1].set_title("Loss over epochs", fontsize=20)
    axs[1].set_ylabel("loss", fontsize=20)
    axs[1].set_xlabel("epochs", fontsize=20)
    axs[0].set_title("Metrics over epochs", fontsize=20)
    axs[0].set_ylabel("metrics", fontsize=20)
    axs[0].set_xlabel("epochs", fontsize=20)
    # Set legend
    axs[1].legend(["loss", "val_loss"], loc="center right", fontsize=15)
    axs[0].legend(
        ["iou", "dice", "val_iou", "val_dice"], loc="center right", fontsize=15
    )
    return plt


def plot_top(imgs, y_true, y_pred, best=True, n_imgs=10):
    """
    Plots best|worst images, masks, prediction masks according to dice coefficient.
    :param numpy.array imgs: array of images
    :param numpy.array y_true: array of masks
    :param numpy.array y_pred: array of predictions
    :param bool best: whether to plot best or worst results
    :param int n_imgs: number of images to plot
    :return matplotlib.pyplot: pyplot of images side by side
    """
    dice_list = []
    for y_t, y_p in zip(y_true, y_pred):
        dice_coef = round(dice_np(y_t, y_p), 4)
        dice_list.append(dice_coef)
    dice_list = np.array(dice_list)
    # Sort list by dice_coef
    idx = dice_list.argsort()
    imgs = imgs[idx]
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    if best:
        return plot_imgs(imgs[-n_imgs:], y_true[-n_imgs:], y_pred[-n_imgs:])
    else:
        return plot_imgs(imgs[:n_imgs], y_true[:n_imgs], y_pred[:n_imgs])


def create_contours(y_pred, target_size=(4096, 4096)):
    """
    Create contours coordinates from binary mask.
    :param numpy.array y_pred: array of binary mask
    :param target_size: size of image we will draw these coordinates
    :return list: list containing list of tuples with x and y coordinates [[(x,y), (x,y)]]
    """
    # (w, h)
    if isinstance(y_pred, Image.Image):
        mask = np.array(y_pred)
    elif isinstance(y_pred, np.ndarray):
        one = y_pred.reshape((y_pred.shape[0], y_pred.shape[1])) * 255
        one = np.array(one, dtype=np.uint8)
        mask = np.stack((one, one, one), axis=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        raise TypeError(
            f"Expected class: {np.ndarray} or {Image.Image} but got {type(y_pred)}"
        )

    width = mask.shape[1]
    height = mask.shape[0]

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []
        for i, point in enumerate(object):
            new_x = (int(point[0][0]) - 0) / (width - 0) * (target_size[0] - 0) - 0
            new_y = (int(point[0][1]) - 0) / (height - 0) * (target_size[1] - 0) - 0
            coords.append((new_x, new_y))
            # To make sure that polygon is fully connected
            if i == len(object) - 1:
                coords.append(coords[0])
        polygons.append(coords)
    return polygons
