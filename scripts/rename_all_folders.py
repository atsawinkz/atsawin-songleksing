import os

def rename_files_in_folders():
    base_directory = r'c:\Users\COLORFUL\Downloads\rice_leaf_diseases_v5'
    folders = [
        'brown_spot',
        'healthy_rice_leaf',
        'leaf_blast',
        'leaf_scald',
        'sheath_blight',
        'tungro'
    ]
    
    for foldername in folders:
        directory = os.path.join(base_directory, foldername)
        if not os.path.exists(directory):
            print(f"Directory {foldername} not found, skipping.")
            continue
            
        prefix = f"{foldername}_"
        
        # Get all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # Sort files to ensure consistent ordering
        files.sort()
        
        print(f"Renaming files in {foldername} ({len(files)} files)...")
        
        for i, filename in enumerate(files, 1):
            extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}{i:04d}{extension}"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            # In case the new name is the same as the old name, skip to avoid error
            if old_path == new_path:
                continue
                
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                print(f"Error renaming {filename} to {new_name} in {foldername}: {e}")
        
        print(f"Finished {foldername}.")

if __name__ == "__main__":
    rename_files_in_folders()
