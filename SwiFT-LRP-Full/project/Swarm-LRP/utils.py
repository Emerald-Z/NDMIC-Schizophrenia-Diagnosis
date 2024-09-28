import torch

def generate_feature_mask():
    height, width, depth, channels = 96, 96, 96, 20
    num_blocks = 10000  # Number of unique values to fill the tensor

    # Create an empty tensor with the desired shape
    tensor = torch.zeros((height, width, depth, channels), dtype=torch.int32)

    # Calculate the number of blocks per dimension
    block_size = int((height * width * depth * channels) / num_blocks)

    # Initialize the value for each block
    value = 1

    # Fill the tensor with block values
    for idx in range(num_blocks):
        # Define the starting and ending indices for the block
        start_idx = idx * block_size
        end_idx = start_idx + block_size
        
        # Flatten the tensor and assign the block values
        flattened_tensor = tensor.view(-1)
        flattened_tensor[start_idx:end_idx] = value
        
        # Increment value for the next block
        value += 1

    return tensor