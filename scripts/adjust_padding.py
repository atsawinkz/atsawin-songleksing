import os

def adjust_padding():
    base_directory = r'c:\Users\COLORFUL\Downloads\rice_leaf_diseases_v5'
    folders = [
        'healthy_rice_leaf',
        'leaf_scald',
        'sheath_blight'
    ]
    
    for foldername in folders:
        directory = os.path.join(base_directory, foldername)
        if not os.path.exists(directory):
            print(f"Directory {foldername} not found, skipping.")
            continue
            
        print(f"Adjusting padding in {foldername}...")
        
        # Get all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # Sort files to ensure we match the previous order
        files.sort()
        
        for i, filename in enumerate(files, 1):
            extension = os.path.splitext(filename)[1]
            # Use 3 digits instead of 4
            new_name = f"{foldername}_{i:03d}{extension}"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            if old_path == new_path:
                continue
                
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                print(f"Error renaming {filename} to {new_name}: {e}")
        
        print(f"Finished {foldername}.")

if __name__ == "__main__":
    adjust_padding()
