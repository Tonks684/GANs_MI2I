import math
import statistics
import numpy as np
import pandas as pd
import skimage.io
import skimage.morphology
import skimage.segmentation
from skimage.metrics import structural_similarity as ssim
from tifffile import imread
from tqdm import tqdm

def psnr_scores(yhats: str, ys: str, bit: int) -> list:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) scores between predicted images and ground truth images.

    Args:
        yhats (str): Path to the predicted images.
        ys (str): Path to the ground truth images.
        bit (int): Number of bits used to represent the pixel values (8 or 16).

    Returns:
        list: List of PSNR scores for each pair of predicted and ground truth images.

    Raises:
        None

    """
    psnr_score_f = []
    for r, g in zip(ys, yhats):
        img_g, img_r = imread(g), imread(r)
        mse = np.mean((img_r - img_g) ** 2)
        if mse == 0:
            psnr_score_f.append(np.float(100))
        elif bit == 8:
            pixel_max = 255.0
            psnr = 20 * (math.log10(pixel_max / math.sqrt(mse)))
            psnr_score_f.append(np.float(psnr))
        elif bit == 16:
            pixel_max = np.max(img_r) - np.min(img_r)
            psnr = 20 * (math.log10(pixel_max / math.sqrt(mse)))
            psnr_score_f.append(np.float(psnr))
    return psnr_score_f

def intersection_over_union(ground_truth, prediction):
    """
    Calculate the Intersection over Union (IoU) score between the ground truth and prediction masks.

    Parameters:
    ground_truth (ndarray): Ground truth mask.
    prediction (ndarray): Predicted mask.

    Returns:
    ndarray: Intersection over Union (IoU) score.

    """
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    # Compute intersection
    h = np.histogram2d(
        ground_truth.flatten(),
        prediction.flatten(),
        bins=(true_objects, pred_objects),
    )
    intersection = h[0]
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection / union
    return IOU

def measures_at(threshold, IOU):
    """
    Calculates various measures at a given threshold.

    Parameters:
    threshold (float): The threshold value for matching objects.
    IOU (numpy.ndarray): The array of IOU values between objects.

    Returns:
    float: The F1 score.
    int: The number of true positives.
    int: The number of false positives.
    int: The number of false negatives.
    """
    matches = IOU > threshold
    true_positives = np.sum(matches, axis=1) == 1
    false_positives = np.sum(matches, axis=0) == 0
    false_negatives = np.sum(matches, axis=1) == 0
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return f1, tp, fp, fn

def compute_af1_results(
    model,
    ground_truth,
    prediction,
    results: pd.DataFrame,
    image_name,
    multi=False,
):
    """
    Computes the AF1 (Average F1) results for a given model, ground truth, and prediction.

    Args:
        model (str): The name of the model.
        ground_truth (array-like): The ground truth data.
        prediction (array-like): The predicted data.
        results (pd.DataFrame): The DataFrame to store the results.
        image_name (str): The name of the image.
        multi (bool, optional): Whether to calculate F1 score at all thresholds. Defaults to False.

    Returns:
        pd.DataFrame: The updated DataFrame with the computed results.
    """
  
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    if IOU.shape[0] > 0:
        jaccard = np.max(IOU, axis=0).mean()
    else:
        jaccard = 0.0
    if multi:
        # Calculate F1 score at all thresholds
        for t in np.arange(0.5, 0.95, 0.05):
            f1, tp, fp, fn = measures_at(t, IOU)
            # Remove those images where no object are detected
            if tp + fn == 0:
                res = {
                    'Model': model, 'Image': image_name, 'GT_Cell_Count': tp + fn,
                    'Threshold': t, 'F1': np.NaN, 'IoU': np.NaN, 'TP': np.NaN,
                    'FP': np.NaN, 'FN': np.NaN,
                }
            else:
                res = \
                    {
                        'Model': model, 'Image': image_name, 'GT_Cell_Count': tp + fn,
                        'Threshold': t, 'F1': f1, 'IoU': jaccard, 'TP': tp,
                        'FP': fp, 'FN': fn,
                    }
            row = len(results)
            results.loc[row] = res
    else:
        # Calculate F1 score at all threshold
        f1, tp, fp, fn = measures_at(.7, IOU)
        # Calculate precision
        precision = tp / (tp + fp)
        # Calculate recall
        recall = tp / (tp + fn)
        # Remove those images where no object are detected
        if tp + fn == 0:
            res = {
                'Model': model, 'Image': image_name, 'GT_Cell_Count': tp + fn,
                'Threshold': .7, 'F1': np.NaN, 'Jaccard': np.NaN, 'TP': np.NaN,
                'FP': np.NaN, 'FN': np.NaN,
                'Precision': precision, 'Recall': recall
            }
        else:
            res = \
                {
                    'Model': model, 'Image': image_name, 'GT_Cell_Count': tp + fn,
                    'Threshold': .7, 'F1': f1, 'Jaccard': jaccard, 'TP': tp,
                    'FP': fp, 'FN': fn, 'Precision': precision,
                    'Recall': recall
                }
        row = len(results)
        results.loc[row] = res
    return results

def get_false_negatives(
    ground_truth,
    prediction,
    results,
    image_name,
    threshold=0.7,
):
    """
    Calculates the number of false negatives at a given IoU threshold.

    Args:
        ground_truth (array-like): Ground truth annotations.
        prediction (array-like): Predicted annotations.
        results (pandas.DataFrame): Existing results dataframe.
        image_name (str): Name of the image.
        threshold (float, optional): IoU threshold. Defaults to 0.7.

    Returns:
        pandas.DataFrame: Updated results dataframe with false negatives information.
    """
 
    # Count number of False Negatives at 0.7 IoU
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    true_objects = len(np.unique(ground_truth))
    if true_objects <= 1:
        return results
    area_true = np.histogram(ground_truth, bins=true_objects)[0][1:]
    true_objects -= 1
    # Identify False Negatives
    matches = IOU > threshold
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    data = np.asarray(
        [
            area_true.copy(),
            np.array(false_negatives, dtype=np.int32),
        ],
    )
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                data=data.T,
                columns=['Area', 'False_Negative'],
            ),
        ],
        sort=False,
    )
    return results

def get_splits_and_merges(ground_truth, prediction, results, image_name):
    """
    Computes the splits and merges based on the ground truth and prediction masks.

    Args:
        ground_truth (numpy.ndarray): The ground truth mask.
        prediction (numpy.ndarray): The predicted mask.
        results (pandas.DataFrame): The DataFrame to store the results.
        image_name (str): The name of the image.

    Returns:
        pandas.DataFrame: The updated DataFrame with the splits and merges information.
    """

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    f1, tp, fp, fn = measures_at(0.7, IOU)
    matches = IOU > 0.1
    merges = np.sum(matches, axis=0) > 1
    splits = np.sum(matches, axis=1) > 1
    r = {
        'Image_Name': image_name,
        'Merges': np.sum(merges),
        'Splits': np.sum(splits),
        'GT_Cell_Count': tp + fn,
    }
    results.loc[len(results) + 1] = r
    return results



def gen_segmentation_scores(
    image_sets: list,
    results: list,
    false_negatives: list=None,
    splits_merges: list=None,
    final_scores_output: str=None,
    multi=False,
):
    """
    Generate segmentation scores for a list of images.

    Args:
        image_sets (list): A list of image sets containing ground truth and predicted masks.
        results (list): A list to store the computed evaluation metrics.
        false_negatives (list): A list to store the false negatives.
        splits_merges (list): A list to store the splits and merges.
        final_scores_output (str): The output directory to save the results.
        multi (bool, optional): Flag indicating whether the evaluation is multi-class or not. Defaults to False.

    Returns:
        tuple: A tuple containing the computed evaluation metrics, false negatives, and splits and merges.
    """

    for index, item in enumerate(tqdm(image_sets)):
        image_name = item[0]
        # Load ground truth data
        y_mask = item[2]
        if y_mask.shape == 3:
            y_mask = y_mask[:, :, 0]
        # Transform ground truth to label matrix
            y_mask = skimage.morphology.label(y_mask)
        # Load prediction
        yhat_mask = item[3]
        if yhat_mask.shape == 3:
            yhat_mask = yhat_mask[:, :, 0]
        # Transform prediction to label matrix
            yhat_mask = skimage.morphology.label(yhat_mask)
        # Compute incremental list for each binary mask
        y_mask = skimage.segmentation.relabel_sequential(y_mask)[0]
        yhat_mask = skimage.segmentation.relabel_sequential(yhat_mask)[0]
        # Compute evaluation metrics
        results = compute_af1_results(
            item[1],
            y_mask,
            yhat_mask,
            results,
            image_name,
            multi=multi,
        )
        if false_negatives is not None:
            false_negatives = get_false_negatives(
                y_mask,
                yhat_mask,
                false_negatives,
                image_name,
            )
        if splits_merges is not None:
            splits_merges = get_splits_and_merges(
                y_mask,
                yhat_mask,
                splits_merges,
                image_name,
            )
    # Double check for removal of blank masks
        results_zero_obj_removed = results[results['GT_Cell_Count'] != 0]
        results_zero_obj_removed.to_csv(final_scores_output + '/results.csv')
    # Print out results
    print(
        f'{results_zero_obj_removed.shape[0]} images successfully saved in '
        f'{final_scores_output}/results.csv'
    )
    false_negatives.to_csv(final_scores_output + '/false_negatives.csv')
    print(
        f'{false_negatives.shape[0]} images successfully saved in '
        f'{final_scores_output}/false_negatives.csv'
    )
    splits_merges.to_csv(final_scores_output + '/splits_merges.csv')
    print(
        f'{splits_merges.shape[0]} images successfully saved in '
        f'{final_scores_output}/splits_merges.csv'
    )
    return results_zero_obj_removed, false_negatives, splits_merges
 