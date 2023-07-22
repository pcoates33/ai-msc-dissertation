# Define a UNET model to use for image segmentation.

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

class DoubleConvolution(nn.Module):
    """ Combine the two convolutions used at each level of the UNET model into a single module
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define what happens in the convolutions for each later in the UNET.
        #TODO : check why we don't need a bias, I think it's because we use BatchNorm2d, but not too sure what the batch normalisation actually does.
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetLayerConv(nn.Module):
    """ Use this to do the double convolutions in each level of the UNET model
    """
    def __init__(self, in_channels, out_channels, direction='d'):
        super().__init__()
        # Define what happens in the convolutions for each later in the UNET.
        self.conv = DoubleConvolution(in_channels, out_channels)
        self.down = False
        self.up = False
        # UNET concatenates the results of the convolution from each layer on the way down with the
        # result from the transpose convolution when going back up through the layers. The following
        # variable holds onto the result of the convolution on the way down..
        self.convoluted_before_pooling = None

        if direction == 'd':
            self.down = True
            # When going down a layer, we halve the size (width and height) by using max pool.
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        elif direction == 'u':
            self.up = True
            # When going up a layer, we reduce the number of channels back to the input and doublt the
            # size (width and height), mirroring the pooling on the way down.
            self.up_conv = nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        if self.down:
            self.convoluted_before_pooling = x
            return self.pool(x)
        
        if self.up:
            return self.up_conv(x)
        
        return x
    

class UNET(nn.Module):
    """ This is a UNET implementaiton to perform image segmentation. 
    """

    def __init__(self, in_channels=3, out_channels=1, features=64, levels=4):
        # Defaults to 3 in_channels as the images to be segmented are in RGB
        # And 1 out_channels for a greyscale segmentation mask.
        # The features are the number of channels created by convolutions at level 1
        # These are then doubled for each level we go down, and halved on the way back up.
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Use separate UnetLayerConv module at each layer on the way down and back up again
        self.going_down = nn.ModuleList()
        self.going_up = nn.ModuleList()

        features_in = in_channels
        features_out = features
        for level in range(levels):
            self.going_down.append(UNetLayerConv(features_in, features_out, direction='d'))
            features_in = features_out
            features_out = features_out * 2     # Double features each time we go down a level
            # Add the going up ones in reverse order so that they match the going downs.
            if len(self.going_up) == levels - 1:
                # bottom level doesn't include concatenation, so convolution is slightly different
                self.going_up.insert(0, UNetLayerConv(features_in, features_out, direction='u'))
            else:
                self.going_up.insert(0, UNetLayerConv(features_out*2, features_in*2, direction='u'))

        # Last things to do are a pair of convolutions to reduce the features down,
        # and a final convolution to get the grey scale output mask

        self.final_conv = nn.Sequential(
                DoubleConvolution(features*2, features),
                nn.Conv2d(features, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        down_level_outputs = []

        for down_level in self.going_down:
            x = down_level(x)
            # Save the result of the convolution from each level on the way down.
            down_level_outputs.append(down_level.convoluted_before_pooling)

        for up_level, down_level_output in zip(self.going_up, reversed(down_level_outputs)):
            x = up_level(x)
            # Need to concatenate x with the output from the matching down level. Basically add more channels
            # Just make sure they have the same size (last 2 components of shape) before doing it. Might be
            # Different if original image size isn't divisible by 2 at every level.
            if x.shape[-2:] != down_level_output.shape[-2:]:
                x = tvf.resize(x, down_level_output.shape[-2:])
            x = torch.cat((down_level_output, x), dim=1)

        # Then do the final convolutions to get to the greyscale output mask
        return self.final_conv(x)


def test():
    # Very basic test to make sure all the convolutions are the correct shape.
    in_channels = 3
    out_channels = 1
    img_width = 640
    img_height = 480
    input = torch.randn((1, in_channels, img_height, img_width))
    model = UNET(in_channels=in_channels, out_channels=out_channels, features=64, levels=5)
    output = model(input)
    print(f'input.shape = {input.shape}, output.shape = {output.shape}')
    assert list(input.shape) == [1, in_channels, img_height, img_width]
    assert list(output.shape) == [1, out_channels, img_height, img_width]


if __name__ == '__main__':
    test()