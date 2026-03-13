import torch
import sys

def occupy_vram(gb_to_occupy):
    if not torch.cuda.is_available():
        print("GPU not found.")
        return

    # 1 GB of float32 is roughly 1024^3 / 4 elements
    # We use float32 (4 bytes per element)
    elements = int((gb_to_occupy * 1024**3) / 4)
    
    print(f"Attempting to occupy {gb_to_occupy} GB of VRAM...")

    try:
        # Allocate a large tensor of zeros
        # This stays in memory until the script is closed or the variable is deleted
        dummy_tensor = torch.zeros(elements, device='cuda', dtype=torch.float32)
        
        print(f"Successfully locked ~{gb_to_occupy} GB on {torch.cuda.get_device_name(0)}.")
        print("Press Ctrl+C to release memory and exit.")
        
        # Keep the script alive so the tensor stays in VRAM
        while True:
            pass

    except RuntimeError as e:
        print(f"Error: Could not allocate memory. You might be asking for more than available. \n{e}")
    except KeyboardInterrupt:
        print("\nReleasing memory and exiting...")

if __name__ == "__main__":
    # Change this number to the amount of GB you want to occupy
    target_gb = 40
    occupy_vram(target_gb)