# Attempt to generate a fairly basic image segmentation classifier using a UNET.
# Use the images generated by shapes_with_templates.ShapeBuilder
# These are 480 x 640

from shapes_with_templates import ShapeBuilder
from unet_model import UNET

import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision
import time
from PIL import Image
from utils import split_greyscale, to_greyscale


if __name__ == "__main__":
    # main function

    # create a convnet
    net = UNET(out_channels=4)  # Allow segmentation of 4 different objects
    # net.load_state_dict(torch.load('./image_segmentation_net.pth'))

    # using Cross Entropy 
    critereon = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=0.0001)

    # Need to transform image to tensor - this changes pixel values from [0,255] to [0,1]
    # The normalize step changes the pixel values from [0,1] to [-1, 1]
    transformer_rgb = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transformer_grey = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5), (0.5))
    # ])

    # Use separate process to generate the shapes.
    batch_size =    10
    mini_batch_size = 5
    q_train = mp.Queue()
    q_test = mp.Queue()
    shape_builder_train = ShapeBuilder(q_train, batch_size)
    shape_builder_train.daemon = True
    shape_builder_train.start()
    shape_builder_test = ShapeBuilder(q_test, batch_size)
    shape_builder_test.daemon = True
    shape_builder_test.start()
    
    img_count = 0
    # train the network
    for epoch in range(500):

        running_loss = 0.0

        # wait for the shape_builder to finish, then get the shapes from the queue.
        batch = q_train.get(timeout=20)
        shape_builder_train.join(timeout=10)

        # kick off another thread to build more shapes        
        shape_builder_train = ShapeBuilder(q_train, batch_size)
        shape_builder_train.daemon = True
        shape_builder_train.start()
   
        img_end = 0
        # batch = create_batch(batch_size)
        while img_end < batch_size:
            img_start = img_end
            img_end = min(img_start + mini_batch_size, batch_size)
            shape_images = []
            shape_masks = []
            for img, mask, label in batch[img_start:img_end]:
                shape_images.append(transformer_rgb(img))
                shape_masks.append(split_greyscale(torch.Tensor(mask), [(25, 75), (75, 125), (125, 175), (175, 225)]))

            # Swap from lists to Tensors
            shape_images = torch.stack(shape_images)
            shape_masks = torch.stack(shape_masks)

            img_count += img_end - img_start
            
            # zero parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimise
            predictions = net.forward(shape_images)
            loss = critereon(predictions, shape_masks)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
        
        # See how many we get right
        match = 0
        # inputs = create_batch(batch_size)
        
        inputs = q_test.get(timeout=10)
        shape_builder_test.join(timeout=10)

        # kick off another 
        # TODO : add a check for last time around the loop - we don't need to build another set then.
        shape_builder_test = ShapeBuilder(q_test, batch_size)
        shape_builder_test.daemon = True
        shape_builder_test.start()

        with torch.no_grad():
            test_imgs = []
            test_masks = []
            for img, mask, label in inputs:
                test_imgs.append(transformer_rgb(img))
                test_masks.append(split_greyscale(torch.Tensor(mask), [(25, 75), (75, 125), (125, 175), (175, 225)]))
            test_imgs = torch.stack(test_imgs)
            test_masks = torch.stack(test_masks)

            predictions = net(test_imgs)

        # Put them in range 0 to 1
        predictions = torch.sigmoid(predictions)
        # Then force to be 0 or 1 for checking accuracy
        predictions.masked_fill_(predictions.gt(0.5), 1.0)
        predictions.masked_fill_(predictions.lt(0.5), 0.0)

        matched = (predictions == test_masks).sum()
        out_of = torch.numel(predictions)
        accuracy = matched*100/out_of        

        # abs_diff = torch.abs(predictions - test_masks)
        # total_diff = torch.sum(abs_diff)
        # avg_diff = total_diff / predictions.shape[0]
        # avg_diff_per_pixel = avg_diff / (predictions.shape[1] * predictions.shape[2] * predictions.shape[3])
        # closer to zero is better for the difference

        # TODO : save the images...
        img = torchvision.transforms.functional.to_pil_image(inputs[0][0])
        # # the test_masks and predictions now have multiple channels, so merge them back into greyscale image.

        mask = to_greyscale(test_masks[0], [50, 100, 150, 200])
        pred = to_greyscale(predictions[0], [50, 100, 150, 200])

        mask = torchvision.transforms.functional.to_pil_image(mask)
        pred = torchvision.transforms.functional.to_pil_image(pred)
        # img.show()
        # mask.show()
        # pred.show()
        imgs_path = "/Users/petercoates/Development/bath-msc/dissertation/ai-msc-dissertation/code/investigations/segmentation/temp_imgs"
        img.save(f"{imgs_path}/img_{img_count}.png","PNG")
        mask.save(f"{imgs_path}/mask_{img_count}.png","PNG")
        pred.save(f"{imgs_path}/pred_{img_count}.png","PNG")

        # TODO compare predictions against test_masks.
        # _, predicted = torch.max(outputs.data, 1)
        # match = (predicted == labels).sum().item()
        
        # for img, label in inputs:
        #     actual = label['shape_type_idx']
        #     output = net(transformer(img))
        #     predicted = torch.argmax(output).item()
        #     # If the actual type is over 0.5 then it's the most likely.
        #     if actual == predicted:
        #         match += 1

        # print statistics
        time_now = time.strftime('%H:%M:%S')
        # print(f'{time_now} [epoch:{epoch}] total images = {img_count} : Running loss for last {batch_size} images = {running_loss:.3f}. Test matched {match} of {len(inputs)} ')
        print(f'{time_now} [epoch:{epoch}] total images = {img_count} : matched {matched} out of {out_of}. Accuracy = {accuracy:0.3f}.')
        running_loss = 0.0

    print('Finished training.')
    torch.save(net.state_dict(), './basic_classifier_net_2.pth')
    # terminate any remaining shape builders
    shape_builder_train.join(timeout=10)
    inputs = q_train.get(timeout=10)
    shape_builder_test.join(timeout=10)
    inputs = q_test.get(timeout=10)