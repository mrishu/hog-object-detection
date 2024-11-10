import cv2
import numpy as np


def draw_bounding_boxes(
    image, labels, add_margin=False, final_window_size=(64, 128), margin=16
):
    """
    Draws bounding boxes on an image based on YOLO-style labels.

    Parameters:
    - image: The image on which to draw (NumPy array).
    - labels: List of labels, each a dictionary with 'class_id', 'cx', 'cy', 'width', and 'height'.
    - image_width: Width of the image in pixels.
    - image_height: Height of the image in pixels.
    """
    h, w = image.shape[:2]
    for label in labels:
        class_id = label[0]
        # Denormalize the label coordinates
        _, x_center, y_center, box_width, box_height = label
        x_center, y_center = int(x_center * w), int(y_center * h)
        box_width, box_height = int(box_width * w), int(box_height * h)
        
        if add_margin:
            # Calculate the required pre-resize box size to include a 16-pixel margin after resizing
            target_width, target_height = (
                final_window_size[0] - 2*margin,
                final_window_size[1] - 2*margin,
            )  # Core object area within 64x128 after resizing
            scale_width = final_window_size[0] / target_width
            scale_height = final_window_size[1] / target_height
        else:
            scale_width, scale_height = (1,1)

        # Expand the box to include margin that will become 16 pixels post-resize
        expanded_width = int(box_width * scale_width)
        expanded_height = int(box_height * scale_height)

        # Calculate the bounding box coordinates with the expanded box
        x1 = max(0, x_center - expanded_width // 2)
        y1 = max(0, y_center - expanded_height // 2)
        x2 = min(w, x_center + expanded_width // 2)
        y2 = min(h, y_center + expanded_height // 2)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(class_id),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def crop_image_using_labels(image, labels, add_margin=True, final_window_size=(64, 128), margin=16):
    """
    Extract and resize labeled regions from an image, adding a 16-pixel margin after resizing.

    Parameters:
    - image: np.array, the input image
    - labels: np.array of shape (n, 4), each row is a label in normalized coordinates
      [x_center, y_center, width, height] with values ranging from 0 to 1.

    Returns:
    - List of cropped and resized 64x128 images with a 16-pixel margin around the object.
    """
    h, w = image.shape[:2]  # Image height and width
    resized_crops = []

    for label in labels:
        # Denormalize the label coordinates
        _, x_center, y_center, box_width, box_height = label
        x_center, y_center = int(x_center * w), int(y_center * h)
        box_width, box_height = int(box_width * w), int(box_height * h)

        if add_margin:
            # Calculate the required pre-resize box size to include a 16-pixel margin after resizing
            target_width, target_height = (
                final_window_size[0] - 2*margin,
                final_window_size[1] - 2*margin,
            )  # Core object area within 64x128 after resizing
            scale_width = final_window_size[0] / target_width
            scale_height = final_window_size[1] / target_height
        else:
            scale_width, scale_height = (1,1)

        # Expand the box to include margin that will become 16 pixels post-resize
        expanded_width = int(box_width * scale_width)
        expanded_height = int(box_height * scale_height)

        # Calculate the bounding box coordinates with the expanded box
        x1 = max(0, x_center - expanded_width // 2)
        y1 = max(0, y_center - expanded_height // 2)
        x2 = min(w, x_center + expanded_width // 2)
        y2 = min(h, y_center + expanded_height // 2)

        # Crop and resize the region to 64x128
        cropped = image[y1:y2, x1:x2]
        resized_crop = cv2.resize(cropped, (64, 128), interpolation=cv2.INTER_AREA)
        resized_crops.append(resized_crop)

    return resized_crops
