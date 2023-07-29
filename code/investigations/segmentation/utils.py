import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image


def split_greyscale(greyscale, target_ranges=None):
    # Convert the grey scale 1 channel mask to multiple channels each of which is a binary mask for a
    # range of pixel values based on the number of target ranges supplied (one for each).
    # Any points in the target range are set to 1.0, the rest set to 0.0.
    # If target_ranges=[(25,75), (75,125), (125,175), (175,225)], then we end up with 4 channels, the
    # first has 1.0 set for any points that had a value >= 25 and < 75. The second uses 75 to 125 and so on.
    #
    channels = len(target_ranges)
    if len(greyscale.shape) == 3:
        greyscale = greyscale.squeeze(0)

    multi_channel = torch.zeros((channels, greyscale.shape[0], greyscale.shape[1]))
    for i in range(channels):
        (target_from, target_to) = target_ranges[i]
        channel = multi_channel[i]
        # Set values between from and to to 1.0
        channel.masked_fill_(greyscale.ge(target_from) * greyscale.lt(target_to), 1.0)

    return multi_channel


def to_greyscale(multi_channel, grey_values=None):
    # Convert the multi channel tensor to single channel grey scale.
    # Points that are 1 in one of the channels get set to the grey value from the supplied list.
    # If two channels have 1 for the same point, then it gets set to 255.
    # multi_channel should be a tensor with 3 dimensions. Channels x Height x Width
    # grey_values should be between 0 and 255, and there should be 1 for each channel
    #
    channels = multi_channel.shape[0]
    greyscale = torch.zeros((multi_channel.shape[1], multi_channel.shape[2]))

    for i in range(channels):
        grey_value = grey_values[i]
        channel = multi_channel[i]
        # Set points with value of 1.0 that are not already set to the grey_value
        greyscale.masked_fill_(channel.eq(1.0) * greyscale.eq(0), grey_value)
        # If the point on the greyscale has already been set by another channel, then reset it to 255
        greyscale.masked_fill_(channel.eq(1.0) * greyscale.ne(grey_value), 255.0)
        
    return greyscale


def test_split_greyscale():
    mask = torch.randint(low=0, high=266, size=(1, 10, 20), dtype=torch.float32)
    actual = split_greyscale(mask, [(50, 100), (150, 250)])
    assert list(actual.shape) == [2, 10, 20]
    # Check the values are as expected
    for x in range(10):
        for y in range(20):
            val = mask[0,x,y]
            if 50.0 <= val < 100.0:
                assert actual[0, x, y] == 1.0
                assert actual[1, x, y] == 0.0
            elif 150.0 <= val < 250.0:
                assert actual[0, x, y] == 0.0
                assert actual[1, x, y] == 1.0
            else:
                assert actual[0, x, y] == 0.0
                assert actual[1, x, y] == 0.0

def test_to_greyscale():
    mask = torch.Tensor([
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ],[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0]
        ],[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ],[
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
    ])
    greyscale = to_greyscale(mask, [50, 100, 150, 200])
    print(greyscale)
    assert torch.all(greyscale.eq(torch.Tensor([
            [200.0, 50.0, 255.0, 0.0],
            [50.0, 150.0, 255.0, 50.0],
            [0.0, 100.0, 255.0, 100.0]
        ])))
    
def test_show_greyscale_image():
    original = torch.zeros([4, 64, 64])
    # top left
    for x in range(4, 31):
        for y in range(4, 31):
            original[0, y, x] = 1.0
    # top right
    for x in range(38, 57):
        for y in range(8, 27):
            original[1, y, x] = 1.0
    # bottom left
    for x in range(12, 23):
        for y in range(42, 53):
            original[2, y, x] = 1.0
    # bottom left
    for x in range(44, 49):
        for y in range(44, 49):
            original[3, y, x] = 1.0
    greyscale = to_greyscale(original, [200, 150, 100, 50])
    pixel_counts = {}
    for x in range(greyscale.shape[0]):
        for y in range(greyscale.shape[1]):
            pixel_value = greyscale[x, y].item()
            if pixel_value not in pixel_counts:
                pixel_counts[pixel_value] = 0
            pixel_counts[pixel_value] += 1

    print(f"Pixel counts greyscale tensor = {pixel_counts}")

    img = torchvision.transforms.functional.to_pil_image(greyscale, mode='F')
    img = img.convert('L')
    pixel_counts = {}
    for pixel_value in list(img.getdata()):
        if pixel_value not in pixel_counts:
            pixel_counts[pixel_value] = 0
        pixel_counts[pixel_value] += 1
    
    print(f"Pixel counts in image = {pixel_counts}")
    img.show()

if __name__ == "__main__":
    # main function
    test_split_greyscale()
    test_to_greyscale()
    test_show_greyscale_image()