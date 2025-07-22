import os
import tifffile
from torch.utils.data import Dataset
import torch
import numpy as np
import natsort 

def clipping_per_channel(x, bottom=1, top=99):
    bc = np.nanpercentile(x,bottom)
    tc = np.nanpercentile(x,top)
    return (bc,tc),(x-bc)/(tc - bc)


def clipping_input_output(input, output, bottom=1, top=99):
    
    # x (H,W,BGR)
    BC, B = clipping_per_channel(input[...,0],bottom, top)
    GC, G = clipping_per_channel(input[...,1],bottom, top)
    RC, R = clipping_per_channel(input[...,2],bottom, top)
    input = np.clip(np.dstack((B,G,R)), 0, 1)
    
    # clip output
    output = output.copy()
    output[...,0] = (output[...,0] - BC[0])/(BC[1] - BC[0])
    output[...,1] = (output[...,1] - GC[0])/(GC[1] - GC[0])
    output[...,2] = (output[...,2] - RC[0])/(RC[1] - RC[0])
    output = np.clip(output,0,1)

    return (BC,GC,RC), input, output


class SpaceNet_dataset(Dataset):
    def __init__(self, root, split):
        super(SpaceNet_dataset).__init__()

        # load input_img paths
        self.root_path = root
        self.split = split

        if split == 'Train':
            self.input_folder_path = os.path.join(self.root_path, split,'Input')
            self.gt_folder_path = os.path.join(self.root_path, split, 'GT')
            self.gt_P_folder_path = os.path.join(self.root_path, split,'gt_P')
            self.input_img_list = natsort.natsorted(os.listdir(self.input_folder_path))
            self.gt_img_list = natsort.natsorted(os.listdir(self.gt_folder_path))
            self.gt_P_list = natsort.natsorted(os.listdir(self.gt_P_folder_path))

        else:
            self.input_folder_path = os.path.join(self.root_path, split, 'Input')
            self.gt_folder_path = os.path.join(self.root_path, split, 'GT')
            self.input_img_list = natsort.natsorted(os.listdir(self.input_folder_path))
            self.gt_img_list = natsort.natsorted(os.listdir(self.gt_folder_path))    

        assert len(self.input_img_list) == len(self.gt_img_list)   

    def __len__(self):
        return len(self.input_img_list)

    def __getitem__(self, idx):

        input_img_path = os.path.join(self.input_folder_path, self.input_img_list[idx])
        gt_path = os.path.join(self.gt_folder_path, self.gt_img_list[idx])
        
        assert self.input_img_list[idx] == self.gt_img_list[idx]

        raw_img = tifffile.imread(input_img_path).astype(np.float32)  # BGR
        gt_img = tifffile.imread(gt_path).astype(np.float32) # BGR 
        img_name = self.input_img_list[idx][:-4]

        if self.split == 'Train':
            assert self.gt_img_list[idx][:-4] == self.gt_P_list[idx][:-6]
            gt_P_path = os.path.join(self.gt_P_folder_path, self.gt_P_list[idx])
            gt_P = torch.from_numpy(np.load(gt_P_path).astype(np.float32)).permute(2, 0, 1) # H x W x C -> C x H x W 

        # Clipping and Normalization: [0, 1] range 
        clips, input_img, gt_img = clipping_input_output(raw_img, gt_img)
        input_img = torch.from_numpy(input_img)
        input_img = input_img.permute(2, 0, 1)

        gt_img = torch.from_numpy(gt_img)
        gt_img = gt_img.permute(2, 0, 1)

        if self.split == 'Train':
            return {"input_img":input_img, "gt_img":gt_img, "gt_P":gt_P,
                 "img_name":img_name, "clips":clips}
        else:
            return {"input_img":input_img, "gt_img":gt_img, "img_name":img_name, "clips":clips, "raw_img":raw_img}