from dataset import OnePoseAllJointsDataset
import torch
from plot import PlotPoseOnImage
import matplotlib.pyplot as plt
#set default tensor type to 32-bit float
torch.set_default_tensor_type('torch.FloatTensor')


dataset = OnePoseAllJointsDataset('/home/billy/Documents/MotionTracking/data/images',
                                  '/home/billy/Documents/MotionTracking/data/annotates', 
                                   'mpii_human_pose_v1_u12_1.mat', train = True)


#print(dataset[1])
i = 6
img = dataset[i]['img']
x = dataset[i]['annot']['x']
y = dataset[i]['annot']['y']
id = dataset[i]['annot']['id']
poses = [{'joints_x':x, 'joints_y':y , 'id':id}]
PlotPoseOnImage(img, poses)
plt.show()




# x_ordered = [x[i] for i in id]
# y_ordered = [y[i] for i in id]

# joint_names = ['r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
#                        'pelvis', 'thorax', 'upper_neck', 'head_top', 'r_wrist', 
#                        'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

# bones_str = [('r_ankle', 'r_knee'),('r_hip', 'r_knee'), ('l_hip', 'l_knee'), 
#          ('l_ankle', 'l_knee'),('l_hip', 'pelvis'), ('r_hip', 'pelvis'), 
#          ('pelvis', 'thorax'), ('thorax', 'upper_neck'), ('upper_neck', 'head_top'),
#          ('thorax', 'l_shoulder'), ('thorax', 'r_shoulder'), ('l_shoulder', 'l_elbow'),
#          ('l_elbow', 'l_wrist'), ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist') ]

# bones_idx = [(joint_names.index(b[0]),joint_names.index(b[1])) for b in bones_str]
# plt.imshow(img.permute(1,2,0))

# j = 0
# plt.scatter(x_ordered[j], y_ordered[j], cmap = 'hsv')
# plt.show()

