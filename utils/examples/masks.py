import torch
import numpy as np
import scipy.misc
from PIL import Image


def read_masks(filepath):
    masks = np.empty((512, 512))

    mask = scipy.misc.imread(filepath)
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks[mask == 3] = 0  # (Cyan: 011) Urban land
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land
    masks[mask == 5] = 2  # (Purple: 101) Rangeland
    masks[mask == 2] = 3  # (Green: 010) Forest land
    masks[mask == 1] = 4  # (Blue: 001) Water
    masks[mask == 7] = 5  # (White: 111) Barren land
    masks[mask == 0] = 6  # (Black: 000) Unknown
    masks[mask == 4] = 6  # (Red: 100) Unknown

    return masks


def write_masks(model_output, output_path):
    pred_max = torch.max(model_output, dim=1)[1].squeeze()
    pred_max = pred_max.cpu().detach().numpy()
    masks = np.empty((512, 512, 3))

    masks[pred_max == 0] = np.array([0, 255, 255])
    masks[pred_max == 1] = np.array([255, 255, 0])
    masks[pred_max == 2] = np.array([255, 0, 255])
    masks[pred_max == 3] = np.array([0, 255, 0])
    masks[pred_max == 4] = np.array([0, 0, 255])
    masks[pred_max == 5] = np.array([255, 255, 255])
    masks[pred_max == 6] = np.array([0, 0, 0])

    img = Image.fromarray(masks.astype(np.uint8))
    img.save(output_path)
