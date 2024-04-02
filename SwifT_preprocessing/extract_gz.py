import os
import shutil
import gzip

# Input and output directories
input_dir = 'COBRE'  # Directory containing the .nii.gz files
output_dir = 'COBRE_test'  # Output directory for extracted files

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Traverse the input directory
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('rest.nii.gz'):
            # Construct the input and output file paths
            input_path = os.path.join(root, file)
            subject_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
            output_path = os.path.join(output_dir, f"{subject_name}.nii.gz")
            
            # Extract the .gz file
            shutil.copyfile(input_path, output_path)

            # with gzip.open(input_path, 'rb') as f_in:
            #     with open(output_path, 'wb') as f_out:
            #         shutil.copyfileobj(f_in, f_out)
                    
            print(f"Extracted {input_path} to {output_path}")