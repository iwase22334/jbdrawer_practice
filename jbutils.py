from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch


def combine_images(images):
    height = max(img.shape[1] for img in images)
    width = sum(img.shape[2] for img in images)

    combined_img = torch.zeros(1, height, width)

    x_offset = 0
    for img in images:
        combined_img[:, :img.shape[1], x_offset:x_offset + img.shape[2]] = img
        x_offset += img.shape[2]

    return combined_img


def fig_new():
    return plt.figure()


def display_images_async(imgs, pause=0.0001):
    combined_img = combine_images(imgs)
    pic = transforms.ToPILImage(mode=None)(combined_img)
    plt.imshow(pic)

    plt.draw()
    plt.pause(pause)


def display_image_async(img):
    pic = transforms.ToPILImage(mode=None)(img)
    plt.imshow(pic)

    plt.draw()
    plt.pause(0.0001)


def display_image(img):
    pic = transforms.ToPILImage(mode=None)(img)
    plt.imshow(pic)

    plt.show()


def save_image(img, name):
    pic = transforms.ToPILImage(mode=None)(img)
    plt.imshow(pic)
    plt.savefig(name)


def plot_history(history, name):
    fig, ax = plt.subplots()

    # 損失の推移を描画する。
    ax.set_title("Loss")
    ax.plot(history["epoch"], history["D_loss"], label="Discriminator")
    ax.plot(history["epoch"], history["G_loss"], label="Generator")
    ax.set_xlabel("Epoch")
    ax.legend()

    fig.savefig(name)

