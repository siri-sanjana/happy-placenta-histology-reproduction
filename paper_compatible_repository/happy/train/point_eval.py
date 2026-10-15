# HAPPY_LOCAL_OUTPUT_PATCH
from scipy.spatial import distance_matrix
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch

from happy.microscopefile.prediction_saver import PredictionSaver


@torch.no_grad()
def evaluate_points_over_dataset(
    dataloader, model, device, score_threshold, max_detections, valid_distance
):
    model.eval()
    num_true_positive = 0
    num_false_positive = 0
    num_false_negative = 0
    prediction_records = []
    image_index = 0
    for data in tqdm(dataloader):
        scale = data["scale"]

        scores, _, boxes = model(data["img"].to(device).float(), device)
        scores = scores.cpu().numpy()
        boxes = boxes.cpu().numpy()
        boxes /= scale[0]

        filtered_preds = PredictionSaver.filter_by_score(
            max_detections, score_threshold, scores, boxes
        )
        predicted_points = convert_boxes_to_points(filtered_preds)

        gt_predictions = data["annot"].numpy()[0][:, :4]
        gt_predictions /= scale[0]
        ground_truth_points = convert_boxes_to_points(gt_predictions)
        empty_image = bool(data["annot"][0][0][-1].numpy() == -1)
        prediction_records.append({"image_index": image_index, "empty_image": empty_image, "ground_truth_points": [] if empty_image else np.asarray(ground_truth_points).reshape(-1, 2).tolist(), "predicted_points": np.asarray(predicted_points).reshape(-1, 2).tolist()})
        image_index += 1

        # For batches with all empty images
        if empty_image:
            num_false_positive += len(predicted_points)
            continue

        # If there are no predicted points in the image
        if len(predicted_points) == 0:
            num_false_negative += len(ground_truth_points)
            continue

        true_positive, false_positive, false_negative = evaluate_points_in_image(
            ground_truth_points, predicted_points, valid_distance
        )
        num_true_positive += true_positive
        num_false_positive += false_positive
        num_false_negative += false_negative

    output_path = globals().get("HAPPY_LOCAL_PREDICTION_PATH")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(prediction_records) + "\n")
    # If all images in the dataset are empty
    if num_true_positive == 0 and num_false_negative == 0:
        return np.nan, np.nan, np.nan, num_false_positive

    if num_true_positive + num_false_positive == 0:
        precision = 0
    else:
        precision = round(num_true_positive / (num_true_positive + num_false_positive),
                          3)
    if num_true_positive + num_false_negative == 0:
        recall = 0
    else:
        recall = round(num_true_positive / (num_true_positive + num_false_negative), 3)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = round(2 * ((precision * recall) / (precision + recall)), 3)
    return precision, recall, f1, num_false_positive


def evaluate_points_in_image(ground_truth_points, predicted_points, valid_distance):
    all_distances = distance_matrix(ground_truth_points, predicted_points)
    sorted_indicies = np.argsort(all_distances)
    sorted_distances = np.sort(all_distances)

    # True positives
    accepted_gt = np.nonzero(sorted_distances[:, 0] <= valid_distance)[0]
    paired_pred_indicies = sorted_indicies[:, 0][accepted_gt]

    # False positives
    pred_indicies = list(range(0, len(predicted_points)))
    extra_predicted_indicies = np.setdiff1d(pred_indicies, paired_pred_indicies)

    # False negative
    gt_indicies = list(range(0, len(ground_truth_points)))
    unpredicted_indicies = np.setdiff1d(gt_indicies, accepted_gt)

    true_positive_count = len(accepted_gt)
    false_positive_count = len(extra_predicted_indicies)
    false_negative_count = len(unpredicted_indicies)
    return true_positive_count, false_positive_count, false_negative_count


def convert_boxes_to_points(boxes):
    # If there are no predictions for that image
    if boxes.size == 0:
        return boxes

    boxes = np.array(boxes)
    x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xs = ((x2s + x1s) / 2).astype(int)
    ys = ((y2s + y1s) / 2).astype(int)
    return list(zip(xs, ys))
