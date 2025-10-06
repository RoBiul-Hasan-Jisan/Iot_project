import os
import shutil
import random


original_path = r"D:\iot\data"

new_path = r"D:\iot\leaf_dataset"

class_map = {
    "Healthy": ["Healthy leaf","Pepper_bell_healthy","Potato_healthy","Tomato_healthy"],
    "Dry": ["Dry_leaf"],
    "Disease": [
        "Alternaria leaf spot","Brown spot","Gray spot","Pepper_bell_Bacterial_spot",
        "Potato_Early_blight","Potato_Late_blight","Rust","Tomato_Target_Spot",
        "Tomato_Tomato_mosaic_virus","Tomato_Tomato_YellowLeaf_Curl_Virus",
        "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
        "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite"
    ]
}

# Split ratio
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1


for split in ['train','val','test']:
    for cls in class_map.keys():
        os.makedirs(os.path.join(new_path, split, cls), exist_ok=True)


for cls_name, folders in class_map.items():
    for folder in folders:
        folder_path = os.path.join(original_path, folder)
        if not os.path.exists(folder_path):
            print(f" Folder not found: {folder_path}")
            continue
        images = os.listdir(folder_path)
        if len(images) == 0:
            print(f" No images found in: {folder_path}")
            continue

        random.shuffle(images)
        n = len(images)
        train_cut = int(n*train_ratio)
        val_cut = int(n*(train_ratio+val_ratio))

        for i, img in enumerate(images):
            src = os.path.join(folder_path,img)
            if i < train_cut:
                dst = os.path.join(new_path,'train',cls_name,img)
            elif i < val_cut:
                dst = os.path.join(new_path,'val',cls_name,img)
            else:
                dst = os.path.join(new_path,'test',cls_name,img)
            shutil.copy(src,dst)

        print(f" Copied {len(images)} images from '{folder}' to '{cls_name}'")

print(" Dataset split completed successfully!")
