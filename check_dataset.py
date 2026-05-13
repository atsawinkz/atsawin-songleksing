from PIL import Image
import os

path = r"c:\Users\COLORFUL\Downloads\rice_leaf_diseases_v5"

corrupted_files = []
sizes = {}

print("Starting analysis...")
for root, _, files in os.walk(path):
    for f in files:
        if f.lower().endswith(('.jpg','.png')):
            filepath = os.path.join(root, f)
            try:
                with Image.open(filepath) as img:
                    img.verify() # verify that it is, in fact, an image
            except Exception as e:
                corrupted_files.append((filepath, str(e)))
                continue
            
            # Now actually get the size (verify doesn't always load the header fully depending on format, but open does)
            try:
                with Image.open(filepath) as img:
                    w, h = img.size
                    sizes[(w, h)] = sizes.get((w, h), 0) + 1
            except Exception as e:
                # Catch any errors on full load
                corrupted_files.append((filepath, "Load error: " + str(e)))

print("\n--- Analysis Results ---")
if corrupted_files:
    print(f"Found {len(corrupted_files)} corrupted files:")
    for cf, err in corrupted_files:
        print(f"{cf} - {err}")
else:
    print("No corrupted files found.")

print("\nTop 10 Image Resolutions (Width x Height):")
for (w, h), count in sorted(sizes.items(), key=lambda x: -x[1])[:10]:
    print(f"{w}x{h}: {count} images")
