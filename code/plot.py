import matplotlib.pyplot as plt
import torch 
import torchvision

#0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,
#9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist

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

bones_idx = [(joint_names.index(b[0]),joint_names.index(b[1])) for b in bones_str]
def plot_skeleton(ax, joints_x, joints_y, id, linewidth=2, linestyle = '-'):
    x_ordered = [joints_x[i] for i in id]
    y_ordered = [joints_y[i] for i in id]
    cmap = plt.get_cmap('hsv')
    for bone in bones_idx:
        color = cmap(bone[1] * cmap.N // len(joint_names)) # color according to second joint index
        ax.plot([x_ordered[bone[0]], x_ordered[bone[1]]], [y_ordered[bone[0]], y_ordered[bone[1]]], linestyle=linestyle, color=color, linewidth = linewidth)
    return

def PlotPoseOnImage(img, poses, ax=plt):
    '''
    poses: may contain multiple set of (joints_x, joints_y, id), so that we can plot multiple poses 
    onto one image
    '''
    if img.is_cuda:
        img_cpu = img.cpu().permute(1,2,0)
    else:
        img_cpu = img.permute(1,2,0)
    #print(img_cpu.size())
    ax.imshow(img_cpu)
    linestyles = ['-', '--', '-.', ':']
    if type(poses) is not list:
        poses = [poses]
    for i,p in enumerate(poses):
        joints_x = p['joints_x']
        joints_y = p['joints_y']
        id = p['id']
        linestyle = linestyles[i%len(linestyles)]
        plot_skeleton(ax, joints_x, joints_y, id, linewidth = 2, linestyle=linestyle)
    
