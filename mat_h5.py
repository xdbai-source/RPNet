import scipy.io as sio
import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
data_path = ''
label_list = glob.glob(os.path.join(data_path, '*.mat'))
for index in range(len(label_list)):
    label_path = label_list[index]
    mat = sio.loadmat(label_path)
    density = mat['density_map'] 
    att = density > 0.001
    att = att.astype(np.float32)
    with h5py.File(label_path.replace('.mat', '.h5').replace('gtdens','new_data'), 'w') as hf:
        hf['density'] = density
        hf['attention'] = att
        hf['dot'] = mat['dot_map']