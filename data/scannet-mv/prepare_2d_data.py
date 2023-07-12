# pre-process ScanNet 2D data
# note: depends on the sens file reader from ScanNet:
#       https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# if export_label_images flag is on:
#   - depends on https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py
#   - also assumes that label images are unzipped as scene*/label*/*.png
# expected file structure:
#  - prepare_2d_data.py
#  - https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py
#  - https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
#
# example usage:
#    python prepare_2d_data.py --scannet_path data/scannetv2 --output_path data/scannetv2_images --export_label_images

import argparse
import os, sys
import numpy as np
import skimage.transform as sktf
import imageio

try:
    from SensorData import SensorData
except:
    print('Failed to import SensorData (from ScanNet code toolbox)')
    sys.exit(-1)
try:
    import util
except:
    print('Failed to import ScanNet code toolbox util')
    sys.exit(-1)


# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--label_map_file', default='', help='path to scannetv2-labels.combined.tsv (required for label export only)')
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

    label_mapping = None
    label_map = util.read_label_mapping(opt.label_map_file, label_from='id', label_to='nyu40id')

    scene_index_file = open(opt.scene_index_file,'r')
    scenes = scene_index_file.readlines()

    print('Found %d scenes' % len(scenes))
    for i in range(len(scenes)):
        scenes[i] = scenes[i][:-1]
        sens_file = os.path.join(opt.scannet_path, scenes[i] + '.sens')
        label_path = os.path.join(opt.scannet_path, scenes[i]+'_2d-label','label')
        instance_path = os.path.join(opt.scannet_path, scenes[i]+'_2d-instance','instance')
        if not os.path.isdir(label_path):
            print_error('Error: using export_label_images option but label path %s does not exist' % label_path)
        if not os.path.isdir(instance_path):
            print_error('Error: instance path %s does not exist' % instance_path)
        output_color_path = os.path.join(opt.output_path, scenes[i], 'color')
        if not os.path.isdir(output_color_path):
            os.makedirs(output_color_path)
        output_depth_path = os.path.join(opt.output_path, scenes[i], 'depth')
        if not os.path.isdir(output_depth_path):
            os.makedirs(output_depth_path)
        output_pose_path = os.path.join(opt.output_path, scenes[i], 'pose')
        if not os.path.isdir(output_pose_path):
            os.makedirs(output_pose_path)
        output_label_path = os.path.join(opt.output_path, scenes[i], 'label')
        if not os.path.isdir(output_label_path):
            os.makedirs(output_label_path)
        output_instance_path = os.path.join(opt.output_path, scenes[i], 'instance')
        if not os.path.isdir(output_instance_path):
            os.makedirs(output_instance_path)

        # read and export
        sys.stdout.write('\r[ %d | %d ] %s\tloading...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        sd = SensorData(sens_file)
        sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        sd.export_color_images(output_color_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)
        sd.export_depth_images(output_depth_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)
        sd.export_poses(output_pose_path, frame_skip=opt.frame_skip)

        for f in range(0, len(sd.frames), opt.frame_skip):
            label_file = os.path.join(label_path, str(f) + '.png')
            image = np.array(imageio.imread(label_file))
            image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0, preserve_range=True)
            mapped_image = map_label_image(image, label_map)
            imageio.imwrite(os.path.join(output_label_path, str(f) + '.png'), mapped_image)

        for f in range(0, len(sd.frames), opt.frame_skip):
            instance_file = os.path.join(instance_path, str(f) + '.png')
            image = np.array(imageio.imread(instance_file))
            image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0, preserve_range=True)
            imageio.imwrite(os.path.join(output_instance_path, str(f) + '.png'), image)
    print('')


if __name__ == '__main__':
    main()

