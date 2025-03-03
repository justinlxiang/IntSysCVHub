import ast
import os

import cv2
import pandas as pd
from tqdm import tqdm
from enum import Enum

import detectron2
from detectron2.structures import BoxMode

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")

def get_detectron_dicts(
        raw_data_path=os.path.join(DATA_PATH, '../2025_targets'), 
        ground_truth_path=os.path.join(DATA_PATH, '../annotations2025.csv')):
    raw_data_path = os.path.abspath(raw_data_path)
    df = pd.read_csv(ground_truth_path)
    indices = df["img_name"].unique()

    df["top_left"] = df[["top_left_x", "top_left_y"]].values.tolist()
    df["bottom_right_x"] = df["top_left_x"] + df["width"]
    df["bottom_right_y"] = df["top_left_y"] + df["height"]
    df["bottom_right"] = df.apply(lambda row: [row["bottom_right_x"], row["bottom_right_y"]], axis=1).tolist()

    data = [None] * len(indices)

    prog_bar = tqdm(range(len(indices)))
    for idx in prog_bar:
        img_filepath = indices[idx]
        file_name = os.path.join(raw_data_path, img_filepath)

        height, width = cv2.imread(file_name).shape[:2]

        df_targets_in_img = df.loc[df["img_name"] == img_filepath]
        annotations = []
        for i, row in df_targets_in_img.iterrows():
            x1, y1 = row["top_left"]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = row["bottom_right"]
            x2, y2 = min(x2, width), min(y2, width)

        data[idx] = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "image_id": idx,
            "annotations": annotations,
        }
    return data


if __name__ == "__main__":
    RAW_DATA_PATH = "./real_runway_imgs"
    GROUND_TRUTH_PATH = "./annotations.csv"
    d = get_detectron_dicts(RAW_DATA_PATH, GROUND_TRUTH_PATH)
