# Generate images that contain multiple simple objects along with semantic templates
from collections.abc import Callable, Iterable, Mapping
import multiprocessing as mp
from typing import Any

import mujoco
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from PIL import Image

class ShapeBuilder(mp.Process):

    def __init__(self, q=None, count=0):
        super().__init__()
        self.count = count
        self.q = q
        self.unique_id = 0

    def run(self):
        # Create the shapes and labels and add them to the q, then we just stop.
        shapes = [self.generate_shape(flip=False) for i in range(self.count)]
        self.q.put(shapes)

    def new_geom(self, offset=None):
        # Create a new geom for the mujoco xml
        # Generate disc, cube, or sphere
        # By default the image is flipped to give shape of '3, 480, 640'
        # Set flip=False to get an array of pixels instead. i.e. '480, 640, 3'

        shape_types = ['sphere', 'box', 'cylinder', 'disc']
        shape_type_idx = np.random.randint(0, len(shape_types))
        shape_type = shape_types[shape_type_idx]
        s_type = shape_type

        # Get x,y,z sizes between 0.01 and 0.02
        size_xyz = np.random.randint(10, 21, 3) / 1000
                
        # Rotate the shape about the z axis
        rotate = "0 0 {0}".format(np.random.randint(0, 90))
        offset_z = size_xyz[2]

        if shape_type == 'sphere':
            # for sphere x,y and z are equal
            size_xyz[0] = size_xyz[2]
            size_xyz[1] = size_xyz[2]
            shape_size = f"{size_xyz[2]:.3f}"
        if shape_type == 'cylinder':
            # for cylinder x and y are equal
            size_xyz[0] = size_xyz[1]
            shape_size = f"{size_xyz[0]:.3f} {size_xyz[2]:.3f}"
            # allow cylinder to lie flat or stand up
            if np.random.randint(0, 2) == 1:
                rotate = "90 0 {0}".format(np.random.randint(0, 90))
                offset_z = size_xyz[0]
        if shape_type == 'box':
            shape_size = f"{size_xyz[0]:.3f} {size_xyz[1]:.3f} {size_xyz[2]:.3f}"
        elif shape_type == 'disc':
            # a widened and flattened cylinder
            shape_type = 'cylinder'
            size_xyz[0] = 2*size_xyz[0]
            size_xyz[1] = size_xyz[0]
            size_xyz[2] = size_xyz[2] / 20
            shape_size = f"{size_xyz[0]:.3f} {size_xyz[2]:.3f}"
            offset_z = size_xyz[2]

        rgb = np.random.randint(2, 6, 3)/10
        # boost one of the rgb values to make it the predominant colour
        colour_main = np.random.randint(0, 3)
        rgb[colour_main] += 0.5

        rgb_1 = '{0:.2f} {1:.2f} {2:.2f}'.format(rgb[0], rgb[1], rgb[2])
        rgb_mark = '{0:.2f} {1:.2f} {2:.2f}'.format(rgb[0]-0.1, rgb[1]-0.1, rgb[2]-0.1)
        rgb_2 = '{0:.2f} {1:.2f} {2:.2f}'.format(rgb[0]-0.2, rgb[1]-0.2, rgb[2]-0.2)


        if offset is None:
            pos_offset = f"0.0 0.0 {offset_z:.3f}"
        else:
            # rho is 'radius' of object we're offsetting from. Add the 'radius' of the new geom to it.
            rho, phi = offset
            rho += np.hypot(size_xyz[0], size_xyz[1])
            pos_offset = f"{rho * np.cos(phi):.3f} {rho * np.sin(phi):.3f} {offset_z:.3f}"

        label = {'colour': ['red','green','blue'][colour_main],
                'colour_idx': colour_main,
                'shape_type': s_type,
                'shape_type_idx': shape_type_idx,
                'offset': pos_offset}
        
        self.unique_id += 1
        asset = f"""<texture name="gt_{self.unique_id}" type="cube" builtin="gradient" mark="random" width="128" height="128" rgb1="{rgb_1}" rgb2="{rgb_2}" markrgb="{rgb_mark}" random="0.05"/>
            <material name="gm_{self.unique_id}" texture="gt_{self.unique_id}" specular="0.0" texuniform="false"/>
        """
        body = f"""<geom name="obj{self.unique_id}" type="{shape_type}" size="{shape_size}" material="gm_{self.unique_id}" euler="{rotate}" pos="{pos_offset}"/>
        """

        return asset, body, label, size_xyz
    
    def pick_offset_angle(self, available_angles):
        angle_selected = np.random.randint(0, len(available_angles))
        offset_angle = available_angles[angle_selected]
        # remove a sector around the angle selected so that the next selection is unlikely to overlap
        low = 360 + (offset_angle - 30)
        high = 360 + (offset_angle + 30)
        available_angles = [angle for angle in available_angles if angle+360 < low or angle+360 > high]
        return offset_angle, available_angles
            
    def generate_shape(self, img_size_y=480, img_size_x=640, flip=True):
        # Generate disc, cube, or sphere
        # By default the image is flipped to give shape of '3, 480, 640'
        # Set flip=False to get an array of pixels instead. i.e. '480, 640, 3'

        # position the shape, x and y both between -0.025 and 0.025
        pos_xy = np.random.randint(-5, 5, 2) / 1000
        body_pos = f"{pos_xy[0]:.3f} {pos_xy[1]:.3f} 0.0"

        # position the light, x and y both between -0.025 and 0.025
        pos_xy = np.random.randint(-25, 25, 2) / 100
        light_pos = "{0:.3f} {1:.3f} 0.3".format(pos_xy[0], pos_xy[1])
        
        # camera pos, vary the z and y. z can go from 0.01 with y from -0.2 to -0.1, up to 
        # 0.2 with y from -0.2 to 0.0
        cam_z = np.random.randint(1, 31)
        # y is adjusted based on z (i.e. how high the camera is)
        cam_y = -np.random.randint((50-cam_z)//2, 51) / 100
        cam_z = cam_z / 100
        cam_pos = "0 {0:.2f} {1:.2f}".format(cam_y, cam_z)

        # drop the first one in the middle
        asset, body, label, size_xyz = self.new_geom()
        assets = [asset]
        bodies = [body]
        labels = [label]
        # determine the 'radius' of the shape from the x and y
        # TODO : handle laying down cyclinders as they need to use x and z.
        rho = np.hypot(size_xyz[0], size_xyz[1])
            
        # offset subsequent geom shapes so that it doesn't overlap the first.
        available_angles = [i for i in range(360)]
        num_extra_geoms = np.random.randint(0, 7)
        for i in range(num_extra_geoms):
            offset_angle, available_angles = self.pick_offset_angle(available_angles)
            # as well as the angle, nudge the next shape out by a random amount
            nudge = 1.0 + (np.random.randint(0, 40)/10)
            asset, body, label, size_xyz = self.new_geom([rho * nudge, np.radians(offset_angle)])
            assets.append(asset)
            bodies.append(body)
            labels.append(label)
        
        shape_xml = f"""
        <mujoco model="shape">

        <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".95 .95 .95" rgb2=".9 .9 .9" width="50" height="50"/>
            <material name="grid" texture="grid" texrepeat="128 128"/>
            {'`n'.join(assets)}
        </asset>

        <worldbody>
            <geom size=".6 .6 .01" type="plane" material="grid"/>
            <light diffuse=".5 .5 .5" pos="{light_pos}" mode="targetbody" target="camera_target"/>
            <camera name="closeup" pos="{cam_pos}" mode="targetbody" target="camera_target"/>
            <body name="camera_target" pos="0 0 0"/>
            <body name="shape" pos="{body_pos}">
                {'`n'.join(bodies)}
            </body>
        </worldbody>

        </mujoco>
        """

        # print(shape_xml)
        # print('{0} {1}'.format(['red','green','blue'][colour_main], s_type))
        # TODO : work out how to set up the label with sensible data
        label = {'cam_pos': cam_pos, 'labels': labels}

        shape_model = mujoco.MjModel.from_xml_string(shape_xml)
        renderer = mujoco.Renderer(shape_model, img_size_y, img_size_x)
        shape_data = mujoco.MjData(shape_model)
        mujoco.mj_forward(shape_model, shape_data)
        renderer.update_scene(shape_data, camera="closeup")
        shape_img = renderer.render()
        if flip:
            # flip the pixel's rgb from last to first
            shape_img = np.moveaxis(shape_img, -1, 0)
        return shape_img, label


if __name__ == "__main__":
    # main function

    # Shape builder can run in separate thread
    # q = mp.Queue()
    # shape_builder = ShapeBuilder()
    # shape_builder.daemon = True
    # shape_builder.start()

    # print('join the shape_builder')
    # shape_builder.join(timeout=5)
    # print('get shapes from queue')
    # shapes = q.get(timeout=5)

    # Or just use it to create shapes like so...
    shape_builder = ShapeBuilder()
    shapes = [shape_builder.generate_shape(flip=False) for i in range(4)]
    
    # create a shape
    # shape_img, shape_label = shapes[0]

    # look at some details of what is returned
    print(f'number of shapes = {len(shapes)}')
    # print(shape_label)
    # print(shape_img.shape)
    # print(f'{shape_img[0][0][0]}, {shape_img[1][0][0]}, {shape_img[2][0][0]}')
    # media.show_image(shape_img)
    for shape_img, shape_label in shapes:
        img = Image.fromarray(shape_img)
        img.show()
        print(shape_label)
    # import psutil
    # import gc
    # print(psutil.virtual_memory())
    # mem_available_init = psutil.virtual_memory().available
    # for i in range(20):
    #     shape_builder = ShapeBuilder(q, 100)
    #     shape_builder.daemon = True
    #     shape_builder.start()

    #     shape_builder.join(timeout=5)
    #     shapes = q.get(timeout=5)
    #     # batch = create_batch(100)
    #     del shapes
    #     gc.collect()
    #     # print(f'batch len : {len(shapes)}')
    #     mem_available = psutil.virtual_memory().available - mem_available_init
    #     print(f'% mem used = {psutil.virtual_memory().percent}%. Change in mem available since start = {mem_available/1e6:,.3f} MB')

    # for shape_img, shape_label in batch:
    #     print(shape_label)
