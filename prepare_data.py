import cv2
import os
from tqdm import tqdm
import numpy as np
from utils import crop_image_using_labels
from hog import compute_gradients, get_window_descriptor


def prepare_data(
    positive_img_path,
    positive_labels_path,
    negative_img_path,
    detection_win_size=(64, 128),
    cell_size=(8, 8),
    unsigned_grad=True,
    num_bins=9,
    block_size=(2, 2),
):

    pos_dataset = []
    neg_dataset = []

    # for positive images:
    for imfile in tqdm(os.listdir(positive_img_path)):
        label_file = imfile[:-4] + ".txt"
        image = cv2.imread(
            os.path.join(positive_img_path, imfile)
        )
        labels = np.loadtxt(os.path.join(positive_labels_path, label_file), ndmin=2)

        cropped_imgs = crop_image_using_labels(image, labels)

        for img in cropped_imgs:
            img = cv2.resize(
                img, detection_win_size, interpolation=cv2.INTER_AREA
            ).astype(np.float64)
            grad_magnitude, grad_angle = compute_gradients(img)
            img_descriptors = get_window_descriptor(
                grad_magnitude,
                grad_angle,
                cell_size=cell_size,
                unsigned_grad=unsigned_grad,
                num_bins=num_bins,
                block_size=block_size,
            )
            pos_dataset.append(img_descriptors)
    pos_class_id = np.ones(len(pos_dataset))

    # for negative images:
    for imfile in tqdm(os.listdir(negative_img_path)):
        image = cv2.imread(
            os.path.join(negative_image_path, imfile)
        )
        img = cv2.resize(image, detection_win_size).astype(np.float64)
        grad_magnitude, grad_angle = compute_gradients(img)
        img_descriptors = get_window_descriptor(
            grad_magnitude,
            grad_angle,
            cell_size=cell_size,
            unsigned_grad=unsigned_grad,
            num_bins=num_bins,
            block_size=block_size,
        )
        neg_dataset.append(img_descriptors)
    neg_class_id = np.zeros(len(neg_dataset))

    return np.vstack((pos_dataset, neg_dataset)), np.hstack(
        (pos_class_id, neg_class_id)
    )


## Prepare and save data for all splits

for split in ["train", "valid", "test"]:

    print(f"Preparing {split} data...")
    base_folder_path = os.path.dirname(os.path.abspath("__main__"))
    positive_image_path = os.path.join(f"inria/{split}/images")
    positive_labels_path = os.path.join(f"inria/{split}/labels")
    negative_image_path = os.path.join(f"inria_neg/{split}/images")

    X, y = prepare_data(positive_image_path, positive_labels_path, negative_image_path)

    if not os.path.exists(f"data/{split}"):
        os.makedirs(f"data/{split}", exist_ok=False)
    np.save(f"data/{split}/features.npy", X)
    np.save(f"data/{split}/labels.npy", y)
