import os
import shutil

def copy_retargeted_clean_datasets(src_root, dst_root):
    """
    Copies only the 'retargeted_clean' subdirectories from src_root to dst_root,
    preserving the structure.
    """
    if not os.path.exists(src_root):
        print(f"Error: Source directory {src_root} does not exist.")
        return

    # Iterate through immediate subdirectories in src_root
    for item in os.listdir(src_root):
        src_path = os.path.join(src_root, item)
        
        # We only care about directories (e.g., huanghao, xuanyu)
        if os.path.isdir(src_path):
            target_subdir_src = os.path.join(src_path, "retargeted_clean")
            
            # Check if 'retargeted_clean' exists in this subdirectory
            if os.path.exists(target_subdir_src) and os.path.isdir(target_subdir_src):
                # Construct destination path
                dst_path = os.path.join(dst_root, item)
                target_subdir_dst = os.path.join(dst_path, "retargeted_clean")
                
                print(f"Found {target_subdir_src}, copying to {target_subdir_dst}...")
                
                # Copy the directory tree
                if os.path.exists(target_subdir_dst):
                    print(f"Warning: Destination {target_subdir_dst} already exists. Skipping or merging.")
                    # shutil.copytree(target_subdir_src, target_subdir_dst, dirs_exist_ok=True) # Optional: merge
                else:
                    try:
                        shutil.copytree(target_subdir_src, target_subdir_dst)
                        print(f"Successfully copied to {target_subdir_dst}")
                    except Exception as e:
                        print(f"Failed to copy {target_subdir_src}: {e}")
            else:
                 print(f"Skipping {item}: 'retargeted_clean' not found.")

if __name__ == "__main__":
    src_dir = "/home/huanghao/source/datasets/twist2_pico"
    dst_dir = "/home/huanghao/source/datasets/twist2_pico_clean"
    
    print(f"Starting copy from {src_dir} to {dst_dir}...")
    copy_retargeted_clean_datasets(src_dir, dst_dir)
    print("Copy process completed.")
