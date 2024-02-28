import os
import shutil
import gzip
import glob

def unzip_gz_file(gz_file, output_dir):
    """Unzip a .gz file."""
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_dir, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def process_subject(subject_dir, output_dir):
    """Process a subject directory."""
    subject_id = os.path.basename(subject_dir)
    anat_dir = os.path.join(subject_dir, 'session_1', 'anat_1')
    rest_dir = os.path.join(subject_dir, 'session_1', 'rest_1')

    # Process anat directory
    if os.path.exists(anat_dir):
        mprage_gz_files = glob.glob(os.path.join(anat_dir, '*mprage.nii.gz'))
        for gz_file in mprage_gz_files:
            unzip_gz_file(gz_file, os.path.join(output_dir, f'anat_{subject_id}.nii'))

    # Process rest directory
    if os.path.exists(rest_dir):
        rest_gz_files = glob.glob(os.path.join(rest_dir, '*rest.nii.gz'))
        for gz_file in rest_gz_files:
            unzip_gz_file(gz_file, os.path.join(output_dir, f'rest_{subject_id}.nii'))

def main(data_dir, output_dir):
    """Main function to traverse directories and process data."""
    os.makedirs(output_dir, exist_ok=True)
    for subject_id in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject_id)
        if os.path.isdir(subject_dir):
            process_subject(subject_dir, output_dir)

if __name__ == "__main__":
    data_dir = 'path/to/data'  # Specify the directory containing subject directories
    output_dir = 'COBRE_labeled_data'  # Specify the output directory
    main(data_dir, output_dir)
