# -*- coding: utf-8 -*-
import os, numpy as np
import torch
import torch.utils.data as data
from scipy.misc import imread, imresize
from scipy.sparse import csr_matrix
from PIL import Image
import xml.etree.ElementTree as ET

class DataLoader(data.Dataset):
    def __init__(self,data_path,trainval,transform,random_crops=0):
        self.data_path = data_path
        self.transform = transform
        self.random_crops = random_crops
        self.trainval = trainval
        self.__init_classes()
        self.names, self.labels = self.__dataset_info()
    
    def __getitem__(self, index):
        x = imread(self.data_path+'/JPEGImages/'+self.names[index]+'.jpg',mode='RGB')
        x = Image.fromarray(x)
        
        scale = np.random.rand()*2+0.25
        w = int(x.size[0]*scale)
        h = int(x.size[1]*scale)
        if min(w,h)<227:
            scale = 227/min(w,h)
            w = int(x.size[0]*scale)
            h = int(x.size[1]*scale)
        #x = x.resize((w,h), Image.BILINEAR) # Random scale
        if self.random_crops==0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)
        
        y = self.labels[index]

        return x, y
    
    def __len__(self):
        return len(self.names)
    
    def __dataset_info(self):
        #annotation_files = os.listdir(self.data_path+'/Annotations')
        # with open(self.data_path+'/ImageSets/Main/'+self.trainval+'.txt') as f:
        with open(self.data_path + '/ImageSets/Segmentation/' + self.trainval + '.txt') as f:
            annotations = f.readlines()
        
        annotations = [n[:-1] for n in annotations]
        # print(annotations)
        names  = []
        labels = []
        for af in annotations:
            label = {}
            if len(af)!=11:
                continue
            filename = os.path.join(self.data_path,'Annotations',af)
            tree = ET.parse(filename+'.xml')
            objs = tree.findall('object')
            num_objs = len(objs)
            boxes = np.zeros((num_objs, 4), dtype=np.float16)
            boxes_cl = np.zeros((num_objs), dtype=np.int64)
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                cls = self.class_to_ind[obj.find('name').text.lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                boxes_cl[ix] = cls



            names.append(af)
            # boxes_list.append(np.array(boxes).astype(np.float32))
            label['labels'] = torch.as_tensor(boxes_cl).type(torch.int64)
            label['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            # boxes = torch.rand(4, 11, 4)
            # labels = torch.randint(1, 91, (4, 11))
            labels.append(label)

        return np.array(names), labels
    
    def __init_classes(self):
        self.classes = ('__background__','aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes  = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        

