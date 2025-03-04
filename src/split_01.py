import os
import shutil
import argparse
import yaml
from get_data import get_data

def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['raw_data']['data_src']
    dest = config['load_data']['preprocessed_data']
    
    # Create destination directories (train & test)
    os.makedirs(os.path.join(dest, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'test'), exist_ok=True)

    # Mapping original class names to generic class labels
    class_mapping = {
        'no_tumor': 'class_0',
        'pituitary_tumor': 'class_1',
        'meningioma_tumor': 'class_2',
        'glioma_tumor': 'class_3'
    }

    # Create class directories in train and test
    for class_label in class_mapping.values():
        os.makedirs(os.path.join(dest, 'train', class_label), exist_ok=True)
        os.makedirs(os.path.join(dest, 'test', class_label), exist_ok=True)

    # Function to copy images
    def copy_images(src_parent, dst_parent):
        for original_class, new_class in class_mapping.items():
            src_dir = os.path.join(src_parent, original_class)
            dst_dir = os.path.join(dst_parent, new_class)
            
            if not os.path.exists(src_dir):
                print(f"Warning: Directory {src_dir} does not exist. Skipping...")
                continue
            
            files = os.listdir(src_dir)
            print(f"{original_class} -> {new_class} ({len(files)} images)")

            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

            print(f"Done copying {original_class} -> {new_class}")

    # Copy training and testing data
    copy_images(os.path.join(root_dir, 'Training'), os.path.join(dest, 'train'))
    copy_images(os.path.join(root_dir, 'Testing'), os.path.join(dest, 'test'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    passed_args = args.parse_args()
    train_and_test(config_file=passed_args.config)
