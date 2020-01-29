import torch 
import matplotlib.pyplot as plt
from gau_trans import gau_trans
img = torch.load('img.pt')
x = torch.load('x.pt')
y = torch.load('y.pt')
id = torch.load('id.pt')

print(x)
print(y)
print(id)
gau_img = gau_trans(img, x, y, id, sigma=10)
print(gau_img.size())
print(torch.max(gau_img))
plt.imshow(gau_img.permute(1,2,0))
plt.show()

# x_ordered = [x[i] for i in id]
# y_ordered = [y[i] for i in id]

# joint_names = ['r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
#                        'pelvis', 'thorax', 'upper_neck', 'head_top', 'r_wrist', 
#                        'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

joint_names = ['l_hip', 'r_hip', 'r_knee', 'r_ankle', 'pelvis', 'thorax', 'upper_neck',
               'head_top', 'l_ankle', 'l_knee', 'l_wrist', 'l_elbow', 'l_shoulder', 'r_shoulder',
               'r_elbow', 'r_wrist']
bones_str = [('r_ankle', 'r_knee'),('r_hip', 'r_knee'), ('l_hip', 'l_knee'), 
         ('l_ankle', 'l_knee'),('l_hip', 'pelvis'), ('r_hip', 'pelvis'), 
         ('pelvis', 'thorax'), ('thorax', 'upper_neck'), ('upper_neck', 'head_top'),
         ('thorax', 'l_shoulder'), ('thorax', 'r_shoulder'), ('l_shoulder', 'l_elbow'),
         ('l_elbow', 'l_wrist'), ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist') ]

# #bones_idx = [(joint_names.index(b[0]),joint_names.index(b[1])) for b in bones_str]
# plt.imshow(img.permute(1,2,0))

# print(id)
# j = 5
# plt.scatter(x_ordered[j], y_ordered[j], cmap = 'hsv')
# plt.show()

