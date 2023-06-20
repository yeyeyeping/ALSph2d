import numpy as np
import os
import json
import SimpleITK as sitk
import cv2
import json
from os.path import basename
from scipy import ndimage
import pandas as pd
from skimage import exposure


def autofit_contrast_cdf(image, bins=40):
    cdf, intensity = exposure.cumulative_distribution(image=image, nbins=bins)
    idx = len(cdf) - 1
    while idx >= 0 and cdf[idx] >= 0.999:
        idx -= 1
    high = intensity[idx]
    idx = 0
    while idx < len(cdf) and cdf[idx] <= 0.001:
        idx += 1
    low = intensity[idx]
    if low > high:
        low = intensity[0]
        high = intensity[-1]
    return np.clip(image, low, high)


def autofit_contrast(img_numpy: np.ndarray, bins=40, cutoff=0.001):
    # reimplementation of automatically fitting constrast  algorithm in itksnap
    frequency, intensity = np.histogram(img_numpy, bins=bins)
    goal = img_numpy.size * cutoff
    front_idx, accum = 0, 0
    while front_idx <= len(frequency):
        if accum + frequency[front_idx] < goal:
            accum += frequency[front_idx]
        else:
            break
        front_idx += 1
    low = round(intensity[front_idx], 1)

    end_idx, accum = len(frequency) - 1, 0
    while end_idx >= 0:
        if accum + frequency[end_idx] < goal:
            accum += frequency[end_idx]
        else:
            break
        end_idx -= 1

    high = round(intensity[end_idx + 1], 1)
    if low >= high:
        low = round(intensity[0], 1)
        high = round(intensity[-1], 1)
    return np.clip(img_numpy, a_min=low, a_max=high)


def intensity_norm(img: np.ndarray):
    return (img - img.min()) / (img.max() - img.min())


def preprocess_train(train_files, folder):
    for filename in train_files:
        img, label = sitk.ReadImage(folder + "/imagesTr/" + filename), sitk.ReadImage(
            folder + "/labelsTr/" + filename)
        start, end = df.loc[filename, "start"], df.loc[filename, "end"]
        img_numpy, label_numpy = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)
        img_numpy = autofit_contrast(img_numpy)
        img_numpy = intensity_norm(img_numpy)
        for i in range(start, end + 1):
            img_path, mask_path = f"preprocessed/train/image/{filename.split('.')[0]}_slice{i}.png", \
                f"preprocessed/train/label/{filename.split('.')[0]}_slice{i}.png"
            img, mask = np.asanyarray(img_numpy[i] * 255, dtype=np.uint8), np.asanyarray(label_numpy[i] * 255,
                                                                                         dtype=np.uint8)
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)


def preprocess_val(val_files, folder):
    for filename in val_files:
        img, label = sitk.ReadImage(folder + "/imagesTr/" + filename), sitk.ReadImage(
            folder + "/labelsTr/" + filename)
        start, end = df.loc[filename, "start"], df.loc[filename, "end"]
        origin, direction, spacing = img.GetOrigin(), img.GetDirection(), img.GetSpacing()
        img_numpy, label_numpy = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)
        img_numpy = autofit_contrast(img_numpy)
        img_numpy = intensity_norm(img_numpy)

        print(start, end)
        out_obj = sitk.GetImageFromArray(img_numpy[start:end + 1])
        out_obj.SetOrigin(origin)
        out_obj.SetSpacing(spacing)
        out_obj.SetDirection(direction)

        out_lab_obj = sitk.GetImageFromArray(label_numpy[start:end + 1])
        out_lab_obj.SetOrigin(origin)
        out_lab_obj.SetSpacing(spacing)
        out_lab_obj.SetDirection(direction)
        output_img_name, output_lab_name = f"preprocessed/test/image/{filename}", f"preprocessed/test/label/{filename}"
        sitk.WriteImage(out_obj, output_img_name)
        sitk.WriteImage(out_lab_obj, output_lab_name)


def preprocess_test(test_files, folder):
    for filename in test_files:
        img, label = sitk.ReadImage(folder + "/imagesTr/" + filename), sitk.ReadImage(
            folder + "/labelsTr/" + filename)
        start, end = df.loc[filename, "start"], df.loc[filename, "end"]
        origin, direction, spacing = img.GetOrigin(), img.GetDirection(), img.GetSpacing()
        img_numpy, label_numpy = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)
        img_numpy = autofit_contrast(img_numpy)
        img_numpy = intensity_norm(img_numpy)

        print(start, end)
        out_obj = sitk.GetImageFromArray(img_numpy[start:end + 1])
        out_obj.SetOrigin(origin)
        out_obj.SetSpacing(spacing)
        out_obj.SetDirection(direction)

        out_lab_obj = sitk.GetImageFromArray(label_numpy[start:end + 1])
        out_lab_obj.SetOrigin(origin)
        out_lab_obj.SetSpacing(spacing)
        out_lab_obj.SetDirection(direction)
        output_img_name, output_lab_name = f"preprocessed/test/image/{filename}", f"preprocessed/test/label/{filename}"
        sitk.WriteImage(out_obj, output_img_name)
        sitk.WriteImage(out_lab_obj, output_lab_name)




df = pd.read_excel("/home/yeep/project/py/ALSph2d/data/1.xlsx", index_col="image")

base_folder = "/home/yeep/dataset/3d/SPH"

with open(base_folder + "/data.json") as fp:
    des = json.load(fp)

train, val, test = [], [], []
for item in des["train"]:
    train.append(basename(item["image"]))

for item in des["test"]:
    val.append(basename(item["image"]))

for item in des["test"]:
    test.append(basename(item["image"]))
preprocess_test(test)
preprocess_val(val)
