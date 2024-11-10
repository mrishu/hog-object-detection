import numpy as np
import cv2

from hog import compute_gradients, get_window_descriptor

DETECTION_WIN_SIZE = (64, 128)  # Standard window size for detection (width, height)
STEP_SIZE = 16  # Step size in pixels for sliding the window
SCALES = [2.0, 2.5, 3.0, 3.5]  # Different scales for the detection window


def non_max_suppression(labels, img_width, img_height, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression on bounding boxes.

    Args:
        labels (np.ndarray): Nx5 array where each row is [confidence, x_center, y_center, width, height].
                             The coordinates are in normalized format (0 to 1).
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.
        iou_threshold (float): Threshold for Intersection over Union (IoU) to suppress boxes.

    Returns:
        np.ndarray: Filtered array of bounding boxes after applying NMS, in the same format as input labels.
    """

    # Convert normalized coordinates to pixel coordinates
    confidences = labels[:, 0]
    x_centers = labels[:, 1] * img_width
    y_centers = labels[:, 2] * img_height
    widths = labels[:, 3] * img_width
    heights = labels[:, 4] * img_height

    x1 = x_centers - widths / 2
    y1 = y_centers - heights / 2
    x2 = x_centers + widths / 2
    y2 = y_centers + heights / 2

    boxes = np.stack((confidences, x1, y1, x2, y2), axis=1)

    # Sort boxes by confidence score in descending order
    indices = np.argsort(-boxes[:, 0])
    boxes = boxes[indices]

    selected_boxes = []

    while len(boxes) > 0:
        # Select the box with the highest confidence
        current_box = boxes[0]
        selected_boxes.append(current_box)

        # Compute IoU between the current box and the rest
        x1_inter = np.maximum(current_box[1], boxes[1:, 1])
        y1_inter = np.maximum(current_box[2], boxes[1:, 2])
        x2_inter = np.minimum(current_box[3], boxes[1:, 3])
        y2_inter = np.minimum(current_box[4], boxes[1:, 4])

        # Calculate intersection area
        inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(
            0, y2_inter - y1_inter
        )

        # Calculate union area
        box_area = (current_box[3] - current_box[1]) * (current_box[4] - current_box[2])
        boxes_area = (boxes[1:, 3] - boxes[1:, 1]) * (boxes[1:, 4] - boxes[1:, 2])
        union_area = box_area + boxes_area - inter_area

        # Compute IoU
        iou = inter_area / (
            union_area + 1e-6
        )  # Add small epsilon to avoid division by zero

        # Suppress boxes with IoU above the threshold
        boxes = boxes[1:][iou < iou_threshold]

    # Convert back to normalized format for the output
    selected_boxes = np.array(selected_boxes)
    confidences = selected_boxes[:, 0]
    x_centers = ((selected_boxes[:, 1] + selected_boxes[:, 3]) / 2) / img_width
    y_centers = ((selected_boxes[:, 2] + selected_boxes[:, 4]) / 2) / img_height
    widths = (selected_boxes[:, 3] - selected_boxes[:, 1]) / img_width
    heights = (selected_boxes[:, 4] - selected_boxes[:, 2]) / img_height

    # Combine results into final Nx5 array
    result_labels = np.stack(
        (confidences, x_centers, y_centers, widths, heights), axis=1
    )

    return result_labels


def sliding_window_detection_multi_scale(
    image,
    classifier,
    detection_win_size=DETECTION_WIN_SIZE,
    step_size=STEP_SIZE,
    scales=SCALES,
    top_k=10,
    iou_threshold_nms=0.5,
):
    """
    Slides a detection window over the image at different scales and uses the classifier to detect humans.
    Stores the top k detections with the highest confidence.

    Parameters:
        - image (np.array): The input image.
        - classifier: A trained SVM classifier.
        - detection_win_size (tuple): Standard detection window size (width, height).
        - step_size (int): Number of pixels to shift the window at each step.
        - scales (list of float): List of scales for resizing the window.
        - top_k (int): Number of top detections to store based on confidence.

    Returns:
        list: List of top k detections in (0, center_x, center_y, width, height) normalized format.
    """
    img_height, img_width = image.shape[:2]

    window_labels = []
    window_descriptors = []

    # Slide the window at different scales
    for scale in scales:
        win_width = int(detection_win_size[0] * scale)
        win_height = int(detection_win_size[1] * scale)
        for y in range(0, img_height - win_height, step_size):
            for x in range(0, img_width - win_width, step_size):
                # Extract and resize the detection window
                window = image[y : y + win_height, x : x + win_width]
                window_resized = cv2.resize(
                    window, (64, 128), interpolation=cv2.INTER_AREA
                )

                # Find HOG descriptor of (resized) detection window
                grad_magnitude, grad_angle = compute_gradients(window_resized)
                descriptor_vector = get_window_descriptor(
                    grad_magnitude,
                    grad_angle,
                    cell_size=(8, 8),
                    unsigned_grad=True,
                    num_bins=9,
                    block_size=(2, 2),
                ).reshape(1, -1)
                window_descriptors.append(descriptor_vector)

                # Calculate normalized center_x, center_y, width, height
                center_x = (x + win_width / 2) / img_width
                center_y = (y + win_height / 2) / img_height
                norm_width = win_width / img_width
                norm_height = win_height / img_height
                win_label = [center_x, center_y, norm_width, norm_height]
                window_labels.append(win_label)

    confidences = np.round(
        classifier.decision_function(np.vstack(window_descriptors)), decimals=2
    )
    top_kth_confidence = np.sort(confidences)[::-1][top_k - 1]

    detected_confidences = confidences[confidences >= top_kth_confidence].reshape(-1, 1)
    detected_labels = np.array(window_labels)[confidences >= top_kth_confidence]
    detected_labels = np.concatenate((detected_confidences, detected_labels), axis=1)

    return non_max_suppression(
        detected_labels, img_width, img_height, iou_threshold=iou_threshold_nms
    ), len(window_descriptors)

