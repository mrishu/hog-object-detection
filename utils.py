import cv2


def draw_bounding_boxes(image, labels):
    """
    Draws bounding boxes on an image based on YOLO-style labels.

    Parameters:
    - image: The image on which to draw (NumPy array).
    - labels: List of labels, each a dictionary with 'class_id', 'cx', 'cy', 'width', and 'height'.
    - image_width: Width of the image in pixels.
    - image_height: Height of the image in pixels.
    """
    if labels.ndim == 1:
        labels = labels.reshape(-1, len(labels))
    for label in labels:
        class_id = label[0]

        image_height, image_width = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        cx = int(label[1] * image_width)
        cy = int(label[2] * image_height)
        width = int(label[3] * image_width)
        height = int(label[4] * image_height)

        # Calculate top-left and bottom-right corners of the bounding box
        x_min = int(cx - width / 2)
        y_min = int(cy - height / 2)
        x_max = int(cx + width / 2)
        y_max = int(cy + height / 2)

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(class_id),
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def crop_image_using_labels(image, labels):
    """
    Crops images based on YOLO-style labels.

    Parameters:
    - image: The input image (NumPy array).
    - labels: List of labels, each a dictionary with 'class_id', 'cx', 'cy', 'width', and 'height'.

    Returns:
    - A list of cropped images.
    """
    cropped_images = []
    image_height, image_width = image.shape[:2]

    if labels.ndim == 1:
        labels = labels.reshape(-1, len(labels))
    for label in labels:
        # Get bounding box parameters
        # class_id = label[0]
        cx = int(label[1] * image_width)
        cy = int(label[2] * image_height)
        width = int(label[3] * image_width)
        height = int(label[4] * image_height)

        # Calculate bounding box coordinates
        x_min = int(cx - width / 2)
        y_min = int(cy - height / 2)
        x_max = int(cx + width / 2)
        y_max = int(cy + height / 2)

        # Ensure the coordinates are within the image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width, x_max)
        y_max = min(image_height, y_max)

        # Crop the image using the bounding box coordinates
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_image)

    return cropped_images
