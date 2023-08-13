import argparse
import os, sys
import numpy as np
import skimage.transform as sktf
import imageio

# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--scene_index_file', default='', help='path to scannet_train.txt (required for scene traversal)')
parser.add_argument('--output_image_width', type=int, default=640, help='export image width')
parser.add_argument('--output_image_height', type=int, default=480, help='export image height')

opt = parser.parse_args()
assert opt.label_map_file != ''
print(opt)


def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

# from https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/2d_helpers/convert_scannet_label_image.py
def map_label_image(image, label_mapping):
    out = np.zeros_like(image)
    for k,v in label_mapping.items():
        out[image==k] = v
    return out.astype(np.uint8)

def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    scene_index_file = open(opt.scene_index_file,'r')
    scenes = scene_index_file.readlines()

    print('Found %d scenes' % len(scenes))
    for i in range(len(scenes)):
        scenes[i] = scenes[i][:-1]
        instance_path = os.path.join(opt.scannet_path, scenes[i]+'_2d-instance','instance')
        if not os.path.isdir(instance_path):
            print_error('Error: instance path %s does not exist' % instance_path)
        num_frames = len(os.listdir(instance_path))
        output_instance_path = os.path.join(opt.output_path, scenes[i], 'instance')
        if not os.path.isdir(output_instance_path):
            os.makedirs(output_instance_path)

        for f in range(0, num_frames, opt.frame_skip):
            instance_file = os.path.join(instance_path, str(f) + '.png')
            image = np.array(imageio.imread(instance_file))
            image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0, preserve_range=True)
            imageio.imwrite(os.path.join(output_instance_path, str(f) + '.png'), image)
    print('')


if __name__ == '__main__':
    main()

