"""from monai.transforms import LoadImage
from scipy.ndimage import zoom
import torch
import os
import time
from multiprocessing import Process, Queue
import numpy as np
import torch.nn.functional as F

def read_data(subj_id, load_root, save_root, count, queue=None, scaling_method=None, fill_zeroback=False, min_val=None, max_val=None):
    filename = f"Sub{subj_id:03d}/ssym_Filtered_4DVolume.nii"
    print("processing: " + filename, flush=True)
    path = os.path.join(load_root, filename)
    try:
        # load each nifti file
        data = LoadImage()(path)
    except Exception as e:
        print(e)
        return None
    
    subj_name = f"0040{subj_id:03d}"
    save_dir = os.path.join(save_root, subj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data = data[:, :, 1:, :]
    
    if min_val is not None and max_val is not None:
        data = np.clip(data, min_val, max_val)
        data[data == min_val] = 0
        
    background = data == 0
    
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = torch.as_tensor(np.array(data_temp[~background].min())) if not fill_zeroback else 0
    data_global[~background] = torch.as_tensor(np.array(data_temp[~background]))

    data_global = data_global.type(torch.float64)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        img = TR.clone()
        data_gpu = img.to('cuda')
        print("shape:", data_gpu.shape)
        data_gpu = data_gpu.reshape(1, 1, 61, 73, 60)
        upsampled_data = F.interpolate(data_gpu, size=(96, 96, 96), mode='trilinear', align_corners=False)

        torch.save(upsampled_data.reshape(96, 96, 96, 1).to('cpu'), os.path.join(save_dir, f"frame_{i}.pt"))


def main():
    dataset_name = 'COBRE'
    load_root = # source root
    save_root = # save root
    scaling_method = 'z-norm'

    os.makedirs(os.path.join(save_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok=True)
    save_root = os.path.join(save_root, 'img')
    
    subjects = [int(subj[3:]) for subj in os.listdir(load_root) if subj.startswith('Sub')]
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    
    # These work well visually in fsleyes
    min_val = 5
    max_val = 50
    
    # Just reduce the low value noise
    
    #min_val = 5
    #max_val = 1000

    for subj_id in sorted(subjects):
        subj_name = f"0040{subj_id:03d}"
        
        expected_seq_length = 32
        
        if (subj_name not in finished_samples) or (len(os.listdir(os.path.join(save_root, subj_name))) < expected_seq_length):
            read_data(subj_id, load_root, save_root, count, queue, scaling_method, min_val=min_val, max_val=max_val)
            # try:
            #     count += 1
            #     p = Process(target=read_data, args=(subj_id, load_root, save_root, count, queue, scaling_method))
            #     p.start()
            #     if count % 32 == 0:
            #         p.join()
            # except Exception as e:
            #     print(f'encountered problem with Sub{subj_id:03d}')
            #     print(e)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
"""

from monai.transforms import LoadImage
from scipy.ndimage import zoom
import torch
import os
import time
from multiprocessing import Process, Queue
import numpy as np
import torch.nn.functional as F

def read_data(subj_id, load_root, save_root, count, queue=None, scaling_method=None, fill_zeroback=False, min_val=None, max_val=None):
    filename = f"Sub{subj_id:03d}/ssym_Filtered_4DVolume.nii"
    print("processing: " + filename, flush=True)
    path = os.path.join(load_root, filename)
    try:
        # load each nifti file
        data = LoadImage()(path)
    except Exception as e:
        print(e)
        return None
    
    subj_name = f"0040{subj_id:03d}"
    save_dir = os.path.join(save_root, subj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data = data[:, :, 1:, :]
    
    if min_val is not None and max_val is not None:
        data = np.clip(data, min_val, max_val)
        data[data == min_val] = 0
        
    background = data == 0
    
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = torch.as_tensor(np.array(data_temp[~background].min())) if not fill_zeroback else 0
    data_global[~background] = torch.as_tensor(np.array(data_temp[~background]))

    data_global = data_global.type(torch.float64)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        img = TR.clone()
        data_gpu = img.to('cuda')
        print("shape:", data_gpu.shape)
        data_gpu = data_gpu.reshape(1, 1, 61, 73, 60)
        upsampled_data = F.interpolate(data_gpu, size=(96, 96, 96), mode='trilinear', align_corners=False)

        torch.save(upsampled_data.reshape(96, 96, 96, 1).to('cpu'), os.path.join(save_dir, f"frame_{i}.pt"))


def main():
    dataset_name = 'COBRE'
    load_root = # source root
    save_root = # save root
    scaling_method = 'z-norm'

    os.makedirs(os.path.join(save_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok=True)
    save_root = os.path.join(save_root, 'img')
    
    subjects = [int(subj[3:]) for subj in os.listdir(load_root) if subj.startswith('Sub')]
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    
    # These work well visually in fsleyes
    min_val = 5
    max_val = 50
    
    # Just reduce the low value noise
    
    #min_val = 5
    #max_val = 1000

    for subj_id in sorted(subjects):
        subj_name = f"0040{subj_id:03d}"
        
        expected_seq_length = 32
        
        if (subj_name not in finished_samples) or (len(os.listdir(os.path.join(save_root, subj_name))) < expected_seq_length):
            read_data(subj_id, load_root, save_root, count, queue, scaling_method, min_val=min_val, max_val=max_val)
            # try:
            #     count += 1
            #     p = Process(target=read_data, args=(subj_id, load_root, save_root, count, queue, scaling_method))
            #     p.start()
            #     if count % 32 == 0:
            #         p.join()
            # except Exception as e:
            #     print(f'encountered problem with Sub{subj_id:03d}')
            #     print(e)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
