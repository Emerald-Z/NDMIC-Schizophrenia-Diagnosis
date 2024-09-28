import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

from scipy.ndimage import zoom

import nibabel as nib

# from mayavi import mlab
# mlab.options.offscreen = True


def visualize(lrp, name): 
    # name is name of file
    # lrp is lrp result
    """
    # Plot heatmap
    img=lrp.detach().numpy()
    plt.imshow(img.squeeze(), cmap='viridis', interpolation='nearest')
    plt.imshow(slice, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add colorbar to show scale
    plt.show()
    """
    
    """
    img=lrp.detach().numpy()
    selected_slice = img[0, :, :, :, 0] # first time point 
    print('selected slice dims: ', len(selected_slice))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice
    axial_slice = selected_slice[:, :, 48]  # Middle slice along z-axis
    # axial_slice = selected_slice[:, :, 10]  # Middle slice along z-axis
    axial = axes[0].imshow(axial_slice, cmap='viridis')
    axes[0].set_title('Axial Slice')
    fig.colorbar(axial, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)  # Add color bar

    # Coronal slice
    coronal_slice = selected_slice[:, 48, :]  # Middle slice along y-axis
    coronal = axes[1].imshow(coronal_slice, cmap='viridis')
    axes[1].set_title('Coronal Slice')
    fig.colorbar(coronal, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)  # Add color bar

    # Sagittal slice
    sagittal_slice = selected_slice[48, :, :]  # Middle slice along x-axis
    sagittal = axes[2].imshow(sagittal_slice, cmap='viridis')
    axes[2].set_title('Sagittal Slice')
    fig.colorbar(sagittal, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)  # Add color bar

    plt.tight_layout()

    plt.show()
    """
    
    img=lrp.detach().cpu().squeeze().numpy()
    print(img.shape)
    
    # Define the mni shape
    mni_shape = (182, 218, 182, 20) # this is stupid and slow but i couldn't figure out how to do it in fsl
    zoom_factors = [o / i for o, i in zip(mni_shape, img.shape)]
    print("interpolating")
    interpolated_img = zoom(img, zoom_factors, order=3)  # Using cubic interpolation (order=3)
    print('creating image')
    # Create a NIfTI image
    nii_img = nib.Nifti1Image(interpolated_img, affine=np.eye(4))
    
    #nii_img = nib.Nifti1Image(img, affine=np.eye(4))

    # Save the NIfTI image to a file
    nib.save(nii_img, name + '.nii.gz')
    
    # Save for other visualization
    # np.save('lrp_numy.npy', img)
    
    # Create a list to store each frame (as an image file)
    filenames = []
    
    # Determine global min and max vals across all frames
    #img_log = np.log(img - img.min() + 1e-7)
    
    #vmin = img_log.min()
    #vmax = img_log.max()

    # Loop through each time point to generate a slice image
    for t in range(img.shape[3]):
        fig, ax = plt.subplots()
        slice = img[img.shape[0] // 2, :, :, t]  # Take the middle axial slice at each time point
        
        """# Normalize the slice (TODO: get rid of this)
        slice_min = slice.min()
        slice_max = slice.max()
        normalized_slice = (slice - slice_min) / (slice_max - slice_min) if slice_max > slice_min else slice"""
        #slice = np.log(slice - slice.min() + 1e-7)
        #slice_img = ax.imshow(slice, cmap= 'inferno', vmin = vmin, vmax = vmax)
        slice_img = ax.pcolormesh(slice, cmap='inferno')#, vmin = vmin, vmax = vmax)
        ax.set_aspect('equal')
        ax.axis('off')  # Turn of axis

        # Save the plot as an image file
        filename = f'slice_{t}.png'
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)

    # Build GIF
    with imageio.get_writer('my_slices.gif', mode='I', duration=2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove temp files
    for filename in filenames:
        os.remove(filename)

    print("GIF created successfully!")