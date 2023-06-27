# Attempt to generate a fairly basic energy based model.

import mujoco
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt

def generate_shape(img_size_y=480, img_size_x=640, flip=True):
    # Generate disc, cube, or sphere
    # By default the image is flipped to give shape of '3, 480, 640'
    # Set flip=False to get an array of pixels instead. i.e. '480, 640, 3'

    shape_types = ['sphere', 'box', 'disc']
    shape_type_idx = np.random.randint(0, 3)
    shape_type = shape_types[shape_type_idx]
    s_type = shape_type

    # Get x,y,z sizes between 0.01 and 0.02
    size_xyz = np.random.randint(10, 21, 3) / 1000
    centre_z = "{0:.3f}".format(size_xyz[2])

    size = centre_z
    if shape_type == 'box':
        size = "{0:.3f} {1:.3f} {2:.3f}".format(size_xyz[0], size_xyz[1], size_xyz[2])
    elif shape_type == 'disc':
        # a flattened cylinder
        shape_type = 'cylinder'
        centre_z = size_xyz[2] / 20
        size = "{0:.3f} {1}".format(2*size_xyz[0], centre_z)

    # position the shape, x and y both between -0.025 and 0.025
    pos_xy = np.random.randint(-25, 25, 2) / 1000
    shape_pos = "{0:.3f} {1:.3f} {2}".format(pos_xy[0], pos_xy[1], centre_z)

    # position the light, x and y both between -0.025 and 0.025
    pos_xy = np.random.randint(-25, 25, 2) / 100
    light_pos = "{0:.3f} {1:.3f} 0.3".format(pos_xy[0], pos_xy[1])

    rgb = np.random.randint(2, 6, 3)/10
    # boost one of the rgb values to make it the predominant colour
    colour_main = np.random.randint(0, 3)
    rgb[colour_main] += 0.5

    rgb_1 = '{0:.2f} {1:.2f} {2:.2f}'.format(rgb[0], rgb[1], rgb[2])
    rgb_mark = '{0:.2f} {1:.2f} {2:.2f}'.format(rgb[0]-0.1, rgb[1]-0.1, rgb[2]-0.1)
    rgb_2 = '{0:.2f} {1:.2f} {2:.2f}'.format(rgb[0]-0.2, rgb[1]-0.2, rgb[2]-0.2)

    # Rotate the shape about the z axis
    rotate = "0 0 {0}".format(np.random.randint(0, 90))

    # camera pos, vary the z and y. z can go from 0.01 with y from -0.2 to -0.1, up to 
    # 0.2 with y from -0.2 to 0.0
    cam_z = np.random.randint(1, 21)
    # y is adjusted based on z (i.e. how high the camera is)
    cam_y = -np.random.randint((20-cam_z)//2, 21) / 100
    cam_z = cam_z / 100
    cam_pos = "0 {0:.2f} {1:.2f}".format(cam_y, cam_z)

    shape_xml = """
    <mujoco model="shape">

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".95 .95 .95" rgb2=".9 .9 .9" width="50" height="50"/>
        <material name="grid" texture="grid" texrepeat="128 128"/>
        <texture name="geom_texture" type="cube" builtin="gradient" mark="random" width="128" height="128" rgb1="{3}" rgb2="{4}" markrgb="{5}" random="0.05"/>
        <material name="geom_material" texture="geom_texture" specular="0.0" texuniform="false"/>
    </asset>

    <worldbody>
        <geom size=".4 .4 .01" type="plane" material="grid"/>
        <light diffuse=".2 .2 .2" pos="{7}" mode="targetbody" target="camera_target"/>
        <camera name="closeup" pos="{8}" mode="targetbody" target="camera_target"/>
        <body name="camera_target" pos="0 0 0"/>
        <body name="shape" pos="{2}">
        <geom name="one" type="{0}" size="{1}" material="geom_material" euler="{6}"/>
        </body>
    </worldbody>

    </mujoco>
    """.format(shape_type, size, shape_pos, rgb_1, rgb_2, rgb_mark, rotate, light_pos, cam_pos)

    # print(shape_xml)
    # print('{0} {1}'.format(['red','green','blue'][colour_main], s_type))
    label = {'colour': ['red','green','blue'][colour_main],
             'colour_idx': colour_main,
             'shape_type': s_type,
             'shape_type_idx': shape_type_idx}

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


def create_batch(size=1):
    batch = [generate_shape(flip=False) for i in range(size)]
    return batch


if __name__ == "__main__":
    # main function

    # create a shape
    shape_img, shape_label = generate_shape()

    # look at some details of what is returned
    print(shape_label)
    print(shape_img.shape)
    print(f'{shape_img[0][0][0]}, {shape_img[1][0][0]}, {shape_img[2][0][0]}')
    # media.show_image(shape_img)

    batch = create_batch(2)
    print(f'batch len : {len(batch)}')
    for shape_img, shape_label in batch:
        print(shape_label)