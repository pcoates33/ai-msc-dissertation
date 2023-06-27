# Attempt to generate a fairly basic energy based model.

from shapes import generate_shape
import torch


# Define the conv


if __name__ == "__main__":
    # main function

    # create a shape
    shape_img, shape_label = generate_shape()

    # look at some details of what is returned
    print(shape_label)
    print(shape_img.shape)
    print(shape_img[0][0])
    # media.show_image(shape_img)