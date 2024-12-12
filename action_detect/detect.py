import numpy as np
from torch import from_numpy, argmax

DEVICE = "cpu"


def action_detect(net, pose):
    # img = cv2.cvtColor(pose.img_pose,cv2.IMREAD_GRAYSCALE)

    img = pose.img_pose.reshape(-1)
    img = img / 255  # 把数据转成[0,1]之间的数据

    img = np.float32(img)

    img = from_numpy(img[None, :]).cpu()

    predect = net(img)

    action_id = int(argmax(predect, dim=1).cpu().detach().item())
    if action_id == 0:
        pose.pose_action = 'nice'
    else:
        pose.pose_action = 'not_nice'

    possible_rate = predect[:, action_id]

    pose.possible_rate = possible_rate.detach().numpy()[0]


    return pose
