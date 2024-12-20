import argparse
import os
import time
import shutil
import cv2
import numpy as np
import torch
from torch import jit

from action_detect.detect import action_detect
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose
from val import normalize, pad_width
from PIL import Image


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(video_source, net, action_net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    delay = 1
    i = 0

    save_dir = '/home/kls/data/human_pose/'
    video_name = video_source.split('/')[-1][:-4]
    save_path = save_dir + video_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # create video folder
    img_path = f'data/{video_name}/'
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    tag2 = 'both_hand_raise/'
    tag1 = 'one_hand_raise/'
    tag0 = 'no_hand_raise/'

    for img in image_provider:
        orig_img = img.copy()

        if i % 1 == 0:
            heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                         total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18], (img.shape[0], img.shape[1]))
                # current_poses.append(pose)

                if len(pose.getKeyPoints()) >= 13:
                    current_poses.append(pose)

            # if track:
            #     track_poses(previous_poses, current_poses, smooth=smooth)
            #     previous_poses = current_poses
            # for pose in current_poses:
            #     pose.draw(img)

            for pose in current_poses:

                pose.img_pose = pose.draw(img, save_path, is_save=False)

                # crown_proportion = pose.bbox[2] / pose.bbox[3]  # 宽高比
                pose = action_detect(action_net, pose)

                if pose.pose_action == 'nice':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), color)
                cv2.putText(img, '{}: {}'.format(pose.pose_action, pose.possible_rate),
                            (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

                # estimate whether raise hands
                r_hand_raise = l_hand_raise = one_hand_raise = False
                if pose.keypoints[4][1] < pose.keypoints[3][1] < pose.keypoints[2][1]:
                    r_hand_raise = True
                if pose.keypoints[7][1] < pose.keypoints[6][1] < pose.keypoints[5][1]:
                    l_hand_raise = True
                if (r_hand_raise and not l_hand_raise) or (not r_hand_raise and l_hand_raise):
                    one_hand_raise = True

                # positioning
                position = pose.get_position(pose.bbox)
                # print(position)

                if position < 120:
                    if r_hand_raise and l_hand_raise:
                        if not os.path.exists(os.path.join(img_path, tag2)):
                            os.mkdir(os.path.join(img_path, tag2))
                        if os.path.exists(os.path.join(img_path, tag1)):
                            shutil.rmtree(os.path.join(img_path, tag1))
                        if os.path.exists(os.path.join(img_path, tag0)):
                            shutil.rmtree(os.path.join(img_path, tag0))
                        cv2.imwrite(os.path.join(img_path, tag2) + f'{str(i).zfill(6)}.jpg', img)

                    if one_hand_raise:
                        if not os.path.exists(img_path + tag1) and not os.path.exists(img_path + tag2):
                            os.mkdir(img_path + tag1)
                        if os.path.exists(img_path + tag1):
                            cv2.imwrite(img_path + tag1 + f'{str(i).zfill(6)}.jpg', img)

                    if not r_hand_raise and not l_hand_raise:
                        if not os.path.exists(img_path + tag0) and \
                                (not os.path.exists(img_path + tag2) or os.path.exists(img_path + tag1)):
                            os.mkdir(img_path + tag0)
                        if os.path.exists(img_path + tag0):
                            cv2.imwrite(img_path + tag0 + f'{str(i).zfill(6)}.jpg', img)

            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            cv2.namedWindow('Lightweight Human Pose Estimation Python Demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Lightweight Human Pose Estimation Python Demo', 640, 480)
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)

            key = cv2.waitKey(delay)
            if key == 27:  # esc
                return
            elif key == 112:  # 'p'
                if delay == 1:
                    delay = 0
                else:
                    delay = 1
        i += 1
    cv2.destroyAllWindows()

    for file in os.listdir(img_path):
        imgs_list = next(os.walk(os.path.join(img_path, file)))[-1]
        imgs_list.sort()
        middle_element = imgs_list[len(imgs_list) // 2]
        print(os.path.join(img_path + file, middle_element))
        im = Image.open(os.path.join(img_path + file, middle_element))
        im.show()


def detect_main(video_source='', image_source='', video_name=''):
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default='weights/checkpoint_iter_370000.pth',
                        required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    args.video = video_source
    args.images = image_source
    args.video_name = video_name
    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    # net = PoseEstimationWithMobileNet()
    # checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    # load_state(net, checkpoint)
    net = jit.load(r'scripts/openpose.jit')

    action_net = jit.load(r'action_detect/checkpoint/pose.jit')

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        # args.track = 0
        images_dir = []
        if os.path.isdir(args.images):
            for img_dir in os.listdir(args.images):
                images_dir.append(os.path.join(args.images, img_dir))
            frame_provider = ImageReader(images_dir)
        else:
            img = cv2.imread(args.images, cv2.IMREAD_COLOR)
            frame_provider = [img]

    # run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
    run_demo(video_source, net, action_net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)


if __name__ == '__main__':
    detect_main(video_source=r'/mnt/marathon/staticresource/person_tracking/videos_person_tracking/姿势'
                             r'/000002_000000_FN001280_TID000037.avi',
                video_name='video1')
