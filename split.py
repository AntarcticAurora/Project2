import argparse
import os
import shutil
import random


def split_dataset(source_folder,train_ratio):
    assert (1 >= train_ratio > 0)
    img_abs_list = []
    label_abs_list = []
    old_folder_list = []
    num_img = 0
    num_label=0
    for dirpath, dirnames, filenames in os.walk(source_folder):
        for dirname in dirnames:
            font_dir = os.path.join(dirpath, dirname)
            old_folder_list.append(font_dir)
            for file_name in os.listdir(font_dir):
                file_path = os.path.join(font_dir, file_name)
                if os.path.isfile(file_path):
                    if file_name.endswith('.png'):
                        img_abs_list.append(file_path)
                        num_img += 1
                    elif file_name.endswith('.txt'):
                        label_abs_list.append(file_path)
                        num_label += 1

    print(f"image number detected: {num_img}")
    print(f"label number detected: {num_label}")

    train_img_des= os.path.join(source_folder,"images/train")
    val_img_des = os.path.join(source_folder, "images/val")
    train_label_des = os.path.join(source_folder, "labels/train")
    val_label_des = os.path.join(source_folder, "labels/val")
    os.makedirs(train_img_des, exist_ok=True)
    os.makedirs(val_img_des, exist_ok=True)
    os.makedirs(train_label_des, exist_ok=True)
    os.makedirs(val_label_des, exist_ok=True)

    # splitting
    random.shuffle(img_abs_list)
    train_size = int(train_ratio * len(img_abs_list))

    train_img_abs_list = img_abs_list[:train_size]
    val_img_abs_list = img_abs_list[train_size:]

    for train_img_path in train_img_abs_list:
        train_label_path = train_img_path.replace('.png', '.txt')
        if train_label_path not in label_abs_list:
            print(f"Can't find train label: {train_label_path}")
            continue
        train_img_name = os.path.basename(train_img_path)
        train_label_name = os.path.basename(train_label_path)
        shutil.move(train_img_path, os.path.join(train_img_des, train_img_name))
        shutil.move(train_label_path, os.path.join(train_label_des, train_label_name))

    for val_img_path in val_img_abs_list:
        val_label_path = val_img_path.replace('.png', '.txt')
        if val_label_path not in label_abs_list:
            print(f"Can't find val label: {val_label_path}")
            continue
        val_img_name = os.path.basename(val_img_path)
        val_label_name = os.path.basename(val_label_path)
        shutil.move(val_img_path, os.path.join(val_img_des, val_img_name))
        shutil.move(val_label_path, os.path.join(val_label_des, val_label_name))

    for folder in old_folder_list:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Successfully removed folder: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source_folder="D:/pythonProject2/dataset"
    split_dataset(source_folder, 0.8)



