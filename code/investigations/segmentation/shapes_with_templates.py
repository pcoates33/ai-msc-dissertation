# Generate images that contain multiple simple objects along with semantic templates
from collections.abc import Callable, Iterable, Mapping
import multiprocessing as mp
from typing import Any

import mujoco
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from PIL import Image

class XmlBuilder():

    def __init__(self, assets=None, bodies=None, masks=None):
        self.assets = assets
        self.bodies = bodies
        self.masks = masks

        self.light_pos = None
        self.cam_pos = None
        self.body_pos = None
        self.bg_texture = None  
        self.bg_colour = 0.94
        
        self.with_new_background()
        self.with_new_camera_angle()
        self.with_new_body_pos()
        self.with_new_light_pos()

    def with_new_background(self):
        # modify the background colour slightly

        # Cycle through flat, checker and gradient.
        # TODO : maybe add more options for backgrounds.
        # if self.bg_texture == 'flat':
        #     self.bg_texture = 'checker'
        # elif self.bg_texture == 'checker':
        #     self.bg_texture = 'gradient'
        # else:
        #     self.bg_texture = 'flat'
        self.bg_colour -= 0.05

    def with_initial_background(self):
        self.bg_texture = 'flat'
        self.bg_colour = 0.94

    def with_new_camera_angle(self): 
        # set new camera angle for next time
        # Vary the z and y. z can go from 0.01 with y from -0.2 to -0.1, up to 
        # 0.2 with y from -0.2 to 0.0
        cam_z = np.random.randint(1, 31)
        # y is adjusted based on z (i.e. how high the camera is)
        cam_y = -np.random.randint((50-cam_z)//2, 51) / 100
        cam_z = cam_z / 100
        self.cam_pos = f"0 {cam_y:.2f} {cam_z:.2f}"

    def with_new_body_pos(self):
        # position the shape, x and y both between -0.025 and 0.025
        pos_xy = np.random.randint(-5, 5, 2) / 1000
        self.body_pos = f"{pos_xy[0]:.3f} {pos_xy[1]:.3f} 0.0"

    def with_new_light_pos(self):
        # position the light, x and y both between -0.025 and 0.025
        pos_xy = np.random.randint(-25, 25, 2) / 100
        self.light_pos = "{0:.3f} {1:.3f} 1.0".format(pos_xy[0], pos_xy[1])

    def build_shape_xml(self):
        bg_rgb1 = f"{self.bg_colour:0.3f} {self.bg_colour:0.3f} {self.bg_colour:0.3f}"
        bg_rgb2 = f"{self.bg_colour-0.03:0.3f} {self.bg_colour-0.03:0.3f} {self.bg_colour-0.03:0.3f}"
        return f"""
        <mujoco model="shape">
        <visual>
            <headlight active="1" ambient="0.2 0.2 0.2" diffuse="0.1 0.1 0.1" specular="0.0 0.0 0.0"/>
        </visual>

        <asset>
            <texture name="grid" type="2d" builtin="{self.bg_texture}" rgb1="{bg_rgb1}" rgb2="{bg_rgb2}" width="500" height="500"/>
            <material name="grid" texture="grid" texrepeat="15 15" />
            {''.join(self.assets)}
        </asset>

        <worldbody>
            <geom size=".6 .6 .01" type="plane" material="grid"/>
            <light active="true" diffuse=".3 .3 .3" pos="{self.light_pos}" mode="targetbody" target="camera_target" cutoff="90"/>
            <light active="true" diffuse=".2 .2 .2" pos="0 0 4.0" mode="targetbody" target="camera_target" castshadow="false" cutoff="45"/>
            <camera name="closeup" pos="{self.cam_pos}" mode="targetbody" target="camera_target"/>
            <body name="camera_target" pos="0 0 0"/>
            <body name="shape" pos="{self.body_pos}">
                {''.join(self.bodies)}
            </body>
        </worldbody>

        </mujoco>
        """
    
    def build_mask_xml(self):
        light_distance = "2.0"
        return f"""
        <mujoco model="mask">
        <visual>
            <headlight active="0" ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" specular="0.0 0.0 0.0"/>
        </visual>

        <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".95 .95 .95" rgb2=".9 .9 .9" width="50" height="50"/>
            <material name="grid" texture="grid" texrepeat="128 128"/>
            {''.join(self.masks)}
        </asset>

        <worldbody>
            <!-- lights from the sides -->
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="{light_distance} {light_distance} 0.0" mode="fixed" dir="-{light_distance} -{light_distance} 0" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="0.0 {light_distance} 0.0" mode="fixed" dir="0 -{light_distance} 0" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="{light_distance} -{light_distance} 0.0" mode="fixed" dir="-{light_distance} {light_distance} 0" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="0.0 -{light_distance} 0.0" mode="fixed" dir="0 {light_distance} 0" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-{light_distance} {light_distance} 0.0" mode="fixed" dir="{light_distance} -{light_distance} 0" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-{light_distance} 0.0 0.0" mode="fixed" dir="{light_distance} 0 0" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-{light_distance} -{light_distance} 0.0" mode="fixed" dir="{light_distance} {light_distance} 0" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-{light_distance} 0.0 0.0" mode="fixed" dir="{light_distance} 0 0" castshadow="false"/>

            <!-- lights from above -->
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="0.5 0.5 {light_distance}" mode="fixed" dir="0 0 -{light_distance}" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-0.5 0.5 {light_distance}" mode="fixed" dir="0 0 -{light_distance}" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="0.5 -0.5 {light_distance}" mode="fixed" dir="0 0 -{light_distance}" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-0.5 -0.5 {light_distance}" mode="fixed" dir="0 0 -{light_distance}" castshadow="false"/>
            
            <!-- lights from below -->
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="0.5 0.5 -{light_distance}" mode="fixed" dir="0 0 {light_distance}" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-0.5 0.5 -{light_distance}" mode="fixed" dir="0 0 {light_distance}" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="0.5 -0.5 -{light_distance}" mode="fixed" dir="0 0 {light_distance}" castshadow="false"/>
            <light ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" pos="-0.5 -0.5 -{light_distance}" mode="fixed" dir="0 0 {light_distance}" castshadow="false"/>
        
            <camera name="closeup" pos="{self.cam_pos}" mode="targetbody" target="camera_target"/>
            <body name="camera_target" pos="0 0 0"/>
            <body name="shape" pos="{self.body_pos}">
                {''.join(self.bodies)}
            </body>
        </worldbody>

        </mujoco>
        """

class ShapeBuilder(mp.Process):

    def __init__(self, q=None, count=0, min_shapes=1, max_shapes=5, backgrounds=2, camera_angles=5):
        super().__init__()
        self.count = count
        self.q = q
        self.unique_id = 0
        self.min_shapes = min_shapes
        # To easily space out the shapes, it's limited to 7 in total.
        self.max_shapes = min(7, max_shapes)
        self.backgrounds = backgrounds

        self.camera_angles = camera_angles

    def run(self):
        # Create the shapes and labels and add them to the q, then we just stop.
        shapes = []
        while len(shapes) < self.count:
            shapes += self.generate_shapes()
        self.q.put(shapes[0: self.count])

    def new_geom(self, offset=None):
        # Create a new geom for the mujoco xml
        # Generate disc, cube, or sphere
        # By default the image is flipped to give shape of '3, 480, 640'
        # Set flip=False to get an array of pixels instead. i.e. '480, 640, 3'

        # shape_types = ['sphere', 'box', 'cylinder', 'disc']
        shape_types = ['box', 'disc', 'sphere', 'cylinder']
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

        # Different values for different shapes
        mask_val = [0.8, 0.6, 0.4, 0.2][shape_type_idx]
        mask_rgb = '{0:.2f} {1:.2f} {2:.2f}'.format(mask_val, mask_val, mask_val)
        
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
            <material name="gm_{self.unique_id}" texture="gt_{self.unique_id}" specular="0.1" texuniform="false"/>
        """
        mask = f"""<texture name="gt_{self.unique_id}" type="cube" builtin="flat" mark="edge" width="128" height="128" rgb1="{mask_rgb}" rgb2="{mask_rgb}" markrgb="{mask_rgb}"/>
            <material name="gm_{self.unique_id}" texture="gt_{self.unique_id}" specular="0.0" texuniform="false"/>
        """
        
        body = f"""<geom name="obj{self.unique_id}" type="{shape_type}" size="{shape_size}" material="gm_{self.unique_id}" euler="{rotate}" pos="{pos_offset}"/>
        """

        return asset, mask, body, label, size_xyz
    
    def pick_offset_angle(self, available_angles):
        angle_selected = np.random.randint(0, len(available_angles))
        offset_angle = available_angles[angle_selected]
        # remove a sector around the angle selected so that the next selection is unlikely to overlap
        low = 360 + (offset_angle - 30)
        high = 360 + (offset_angle + 30)
        available_angles = [angle for angle in available_angles if angle+360 < low or angle+360 > high]
        return offset_angle, available_angles
            
    def generate_shapes(self, img_size_y=480, img_size_x=640):
        # Returns list of (img, mask, label). The number in the list depends on number of backgrounds
        # and camera angles defined in the ShapeBuilder. And is equal to the product of the two.
        label = {'to_do': 'set to something useful'} # TODO : generate a useful label.

        xml_builder = self.generate_xml_builder()

        shapes = []        
        # 
        for _ in range(self.camera_angles):
            mask_xml = xml_builder.build_mask_xml()
            shape_mask = self.render_xml(mask_xml, img_size_y, img_size_x)
            shape_mask = np.moveaxis(shape_mask, -1, 0)
            # Smoothing gives consistent grey to a shape. Just take single layer of the 
            # mask image as rgb will all be the same.
            shape_mask = self.smooth_mask(shape_mask[0])
            xml_builder.with_initial_background()

            for _ in range(self.backgrounds):
                shape_xml = xml_builder.build_shape_xml()
                shape_img = self.render_xml(shape_xml, img_size_y, img_size_x)
                shapes.append((shape_img, shape_mask, label))
                xml_builder.with_new_background()

            xml_builder.with_new_camera_angle() # set new camera angle for next time

        return shapes

    def generate_xml_builder(self):
        # Generate object geom (disc, cube, sphere or cylinder)
        # drop the first one in the middle
        asset, mask, body, label, size_xyz = self.new_geom()
        assets = [asset]
        masks = [mask]
        bodies = [body]
        labels = [label]
        # determine the 'radius' of the shape from the x and y
        rho = np.hypot(size_xyz[0], size_xyz[1])
            
        # offset subsequent geom shapes so that it doesn't overlap the first.
        available_angles = [i for i in range(360)]
        num_extra_geoms = np.random.randint(self.min_shapes-1, self.max_shapes)
        for i in range(num_extra_geoms):
            offset_angle, available_angles = self.pick_offset_angle(available_angles)
            # as well as the angle, nudge the next shape out by a random amount
            nudge = 1.0 + (np.random.randint(0, 40)/10)
            asset, mask, body, label, size_xyz = self.new_geom([rho * nudge, np.radians(offset_angle)])
            assets.append(asset)
            masks.append(mask)
            bodies.append(body)
            labels.append(label)

        return XmlBuilder(assets, bodies, masks)
    
    def smooth_mask(self, mask):
        # Now reduce it down to single values per shape. Despite the lighting, Mujoco renders a bit of gradient and shadow.
        # We can smooth this out a bit so that each shape uses a single value for each pixel in the mask.
        # The mask rgb used 0.2, 0.4, 0.6 and 0.8. These map to roughly 50, 100, 150, and 200 when scaled up to 255
        # which seems to be what the mujoco rendering does.
        # TODO : should be able to make this more efficient.
        # build array with smoothed values.
        smoothed_values = np.zeros(256, dtype=np.uint8)
        for target in [50, 100, 150, 200]:
            smoothed_values[target] = target
            for i in range(15):
                smoothed_values[target-i] = target
                smoothed_values[target+i] = target
            
        for col in range(mask.shape[0]):
            for row in range(mask.shape[1]):
                mask_val = mask[col, row]
                if mask_val != 0:
                    mask[col, row] = smoothed_values[mask_val]

        # Remove pixels that don't have at least 3 neighbours the same.
        for col in range(mask.shape[0]):
            for row in range(mask.shape[0]):
                mask_val = mask[col, row]
                if mask_val != 0:
                    neighbours = 0
                    left = max(col-1, 0)
                    right = min(col+2, mask.shape[0])
                    top = max(row-1, 0)
                    bottom = min(row+2, mask.shape[1])
                    for n_col in range(left, right):
                        for n_row in range(top, bottom):
                            if mask[n_col, n_row] == mask_val:
                                neighbours += 1
                    if neighbours < 4: # count includes itself, so 4 would be 3 neighbours
                        mask[col, row] = 0
        
        return mask
    
    def render_xml(self, xml, img_size_y, img_size_x):
        shape_model = mujoco.MjModel.from_xml_string(xml)
        renderer = mujoco.Renderer(shape_model, img_size_y, img_size_x)
        shape_data = mujoco.MjData(shape_model)
        mujoco.mj_forward(shape_model, shape_data)
        renderer.update_scene(shape_data, camera="closeup")
        img = renderer.render()
        return img


if __name__ == "__main__":
    # main function

    # Shape builder can run in separate thread
    # q = mp.Queue()
    # shape_builder = ShapeBuilder()
    # shape_builder.daemon = True
    # shape_builder.start()

    # print('join the shape_builder')
    # shape_builder.join(timeout=5)``
    # print('get shapes from queue')
    # shapes = q.get(timeout=5)

    # Or just use it to create shapes like so...
    shape_builder = ShapeBuilder(min_shapes=1, max_shapes=7, backgrounds=2, camera_angles=2)
    shapes = shape_builder.generate_shapes()
    
    # create a shape
    # shape_img, shape_label = shapes[0]

    # look at some details of what is returned
    print(f'number of shapes = {len(shapes)}')
    # print(shape_label)
    # print(shape_img.shape)
    # print(f'{shape_img[0][0][0]}, {shape_img[1][0][0]}, {shape_img[2][0][0]}')
    # media.show_image(shape_img)
    for shape_img, shape_mask, shape_label in shapes:
        print(shape_mask.shape)
        img = Image.fromarray(shape_img)
        mask = Image.fromarray(shape_mask)
        img.show()
        mask.show()
        print(shape_label)

        # See how much the pixel value in the mask varies
        pixel_values = {}
        for col in shape_mask:
            for pixel in col:
                if pixel not in pixel_values:
                    pixel_values[pixel] = 0
                pixel_values[pixel] += 1

        print(pixel_values)
                



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
