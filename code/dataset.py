import torch
import numpy as numpy
import torch.utils.data as data
import scipy.io as sio
import os
import skimage.io as skio

class OnePoseAllJointsDataset(data.Dataset): #right now only accept train = True
    def __init__(self, imagedir, annotdir, annot_mat_file_name, train= True):
        self.imagedir = imagedir
        self.annot_mat = os.path.join(annotdir, annot_mat_file_name)
        self.image_names = []
        self.annot_mat = sio.loadmat(self.annot_mat, struct_as_record=False)
        self.annots = {}
        self.train_label = train
        self.idx_train = self.annot_mat['RELEASE'][0,0].__dict__['img_train'][0].tolist()
        self._data_cleaning_()
        self._annot_init_()
    def _data_cleaning_(self): 
        '''
        prepare the list of image names, to be indexed
        '''
        anno_len = self.annot_mat['RELEASE'][0,0].__dict__['annolist'].shape[1]
        for i in range(anno_len):
            self.image_names.append(self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['image'][0,0].__dict__['name'].item())
            
        if self.train_label == True:
            _img_names = [x for x, y in zip(self.image_names, self.idx_train) if y == 1]
        else: 
            _img_names = [x for x, y in zip(self.image_names, self.idx_train) if y == 0]

        _names_ = os.listdir(self.imagedir)
        for name in _img_names:
            if name not in _names_:
                _img_names.remove(name)
        self.image_names = _img_names        
        return             

    def _annot_init_(self):
        '''
            Extract names, joint positions from the .mat annotation file into a dictionary
        '''
        anno_len = self.annot_mat['RELEASE'][0,0].__dict__['annolist'].shape[1]
        for i in range(anno_len):
            img_fn = self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['image'][0,0].__dict__['name'].item()
            if img_fn not in self.image_names:
                #Ignore those images without 
                continue
            pose_num = self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'].shape[1]
            if pose_num != 1: #Ignore those pictures with multiple human poses
                try:
                    self.image_names.remove(img_fn)
                except ValueError:
                    pass #In case some file names in the annotation doesn't have a corresponding image file
                continue
            try:
                joints_num =  self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['annopoints'][0,0].__dict__['point'].shape[1]
            except:
                try:
                    self.image_names.remove(img_fn)
                except ValueError:
                    pass #In case some file names in the annotation doesn't have a corresponding image file
                continue

            if joints_num != 16: #Ignore those figures with incomplete joint information
                try:
                   self.image_names.remove(img_fn)
                except ValueError:
                    pass 
                continue

            try:
                scale =  self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['scale'][0,0]
                #rough human position
                obj_pos_x =  self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['objpos'][0,0].__dict__['x'][0,0]
                obj_pos_y =  self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['objpos'][0,0].__dict__['y'][0,0]
                #person-centric body joint annotations
                joints_x = []
                joints_y = []
                joints_id = []
                joints_visible = []
                for j in range(joints_num):
                    joints_x.append(self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['annopoints'][0,0].__dict__['point'][0,j].__dict__['x'][0,0])
                    joints_y.append(self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['annopoints'][0,0].__dict__['point'][0,j].__dict__['y'][0,0])
                    joints_id.append(self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['annopoints'][0,0].__dict__['point'][0,j].__dict__['id'][0,0])
                    try:
                        joints_visible.append(self.annot_mat['RELEASE'][0,0].__dict__['annolist'][0,i].__dict__['annorect'][0,0].__dict__['annopoints'][0,0].__dict__['point'][0,j].__dict__['is_visible'][0,0])
                    except:
                        joints_visible.append(0)
                annot = {img_fn: {'img_fn':img_fn, 'scale': scale, 'objpos_x':obj_pos_x, 'objpos_y':obj_pos_y, 'x':joints_x, 'y': joints_y, 'id': joints_id, 'is_visible': joints_visible}}
                self.annots.update(annot)
            except:
                try:
                   self.image_names.remove(img_fn)
                except ValueError:
                    pass 
                continue
        return 

    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        fn = self.image_names[idx]
        fn_ = os.path.join(self.imagedir, fn)
        img = torch.from_numpy(skio.imread(fn_)).permute(2,0,1)
        return {'img':img.float()/255, 'annot': self.annots[fn]}

