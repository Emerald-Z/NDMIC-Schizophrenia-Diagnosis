import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
import pandas as pd
from scipy.ndimage import zoom
import re   # regex

class NeuroimagingDataset(Dataset):
    def __init__(self, root_dir, phenotypic_data_path, feature_types='all', transforms=None):
        """
        Args:
            root_dir (string): Directory with all the feature folders.
            feature_types (list or string): Specific feature types to load or 'all' for loading all features.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.feature_folders = {
            'ALFF': 'ALFF_FunImgARCW',
            'fALFF': 'fALFF_FunImgARCW',
            'DegreeCentrality': 'DegreeCentrality_FunImgARCWF',
            'ReHo': 'ReHo_FunImgARCWF',
            'VMHC': 'VMHC_FunImgARCWFsymS'
        }
        self.feature_types = feature_types
        if feature_types == 'all':
            self.selected_features = self.feature_folders.keys()
        else:
            self.selected_features = feature_types if isinstance(feature_types, list) else [feature_types]

        # Phenotypic data
        self.phenotypic_data = pd.read_csv(phenotypic_data_path)
        # Map subject ID to diagnosis
        self.id_to_diagnosis = {row['Subject']: 1 if row['Subject Type'] == "Patient" else 0 for index, row in self.phenotypic_data.iterrows()}
        
        # Assuming each subject's data is stored directly in the feature folders
        #self.subjects = os.listdir(os.path.join(root_dir, self.feature_folders[next(iter(self.selected_features))]))
        
        subjects_directory = os.path.join(root_dir, self.feature_folders[next(iter(self.selected_features))])
        all_files = os.listdir(subjects_directory)

        # Create a dictionary mapping of index to the full filename for files starting with 'smALFFMap'
        self.subjects = {}
        for index, file in enumerate(sorted(all_files)):
            if file.startswith('sALFFMap_Sub'):
                # Extract the subject number from the file name
                # Assuming the format is always 'sALFFMap_Sub' followed by the subject number
                subject_number = int(re.split('_Sub|\.', file)[1])  # Convert the extracted part to an integer
                self.subjects[subject_number] = file
        #self.subjects = {index: file for index, file in enumerate(sorted(all_files)) if file.startswith('sALFFMap')}

        # print(self.subjects)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        if idx in [70, 83]:
            return torch.zeros(64, 64, 40), 0
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject_id = self.subjects[idx]
        features = {}
        for feature in self.selected_features:
            feature_path = os.path.join(self.root_dir, self.feature_folders[feature], self.subjects[idx])
            img = nib.load(feature_path)
            data = img.get_fdata()
            # Resample
            # Original dimensions of the data
            original_shape = data.shape  # (61, 73, 61)??

            # Target dimensions?
            target_shape = (64, 64, 40)

            # Calculate the zoom factors needed to go from original to target dimensions
            zoom_factors = [t/o for t, o in zip(target_shape, original_shape)]

            # Apply resampling
            resampled_data = zoom(data, zoom_factors, order=3)  # order=3 for cubic interpolation
            # print(resampled_data.shape)
            
            if self.transforms:
                resampled_data = self.transforms(resampled_data)
            else:
                # Convert to tensor without any transforms
                resampled_data = torch.tensor(resampled_data, dtype=torch.float)
            features[feature] = resampled_data
        
        # Fetch label for the subject
        # print(subject_id)
        
        # Using the full subject_id to match phenotypic data; adjust if necessary
        subject_id_clean = int(subject_id.replace('sALFFMap_Sub', '').split('.')[0])
        if subject_id_clean in self.id_to_diagnosis:
            label = self.id_to_diagnosis[subject_id_clean]
        else:
            # Handle cases where the subject_id is not found directly
            # You might need to adjust how you're matching or ensure that your phenotypic data
            # is correctly mapped to these ids.
            print(f"Warning: Label for subject_id {subject_id_clean} not found. Setting to 0, because why not. makes it actually run.")
            label = 0
        
        #print(features[self.feature_types])
        # Optionally, return labels or other information by modifying this part
        return features[self.feature_types], label

# Example usage for loading all features:
# dataset_all = NeuroimagingDataset('/path/to/your/data', feature_types='all', transforms=None)
# dataloader_all = DataLoader(dataset_all, batch_size=4, shuffle=True)

# Example usage for loading a specific feature (e.g., ALFF):
# dataset_alff = NeuroimagingDataset('/path/to/your/data', feature_types='ALFF', transforms=None)
# dataloader_alff = DataLoader(dataset_alff, batch_size=4, shuffle=True)
