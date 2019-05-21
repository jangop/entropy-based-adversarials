#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import random
random.seed(42)

from PIL import Image
import numpy as np
import skimage.io
import pandas
import csv

from keras.applications.inception_v3 import InceptionV3
import foolbox
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LinfinityBasicIterativeAttack

from foolbox_keras_model_entropy import FoolboxKerasModelEntropy

import utils


NUM_CLASSES = 999
TARGET_CLASS = 42

def list_dir(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def list_files(path, ext='.JPEG'):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isfile(os.path.join(path, d)) and d.endswith(ext)]


def load_model():
    return InceptionV3()

def compute_entropy(file_in, binary=True):
    img = skimage.io.imread(file_in)
    gray = skimage.color.rgb2gray(img)
    mask = skimage.filters.rank.entropy(gray, skimage.morphology.disk(3))
    
    if binary is True:
        low = mask < 4.2
        high = mask >= 4.2

        mask[low] = 0.0
        mask[high] = 1.0
    
    return np.mean(mask)


def compute_saturation(file_in):
    img = skimage.io.imread(file_in)
    hsv = skimage.color.rgb2hsv(img)
    
    return np.mean(hsv[:,1])


def compute_entropy_per_image(data_in_path, file_out):
    # Enumerate all image files
    files = list_files(data_in_path)

    # Compute and store entropy of each image
    df = pandas.DataFrame(columns=['file', 'entropy'])

    m = len(files)
    i = 0
    for file_in in files:
        i += 1
        print("{0}/{1}".format(i, m))
        df = df.append({'file': file_in, 'entropy': compute_entropy(file_in)}, ignore_index=True)

    df.to_csv(file_out)


def compute_saturation_per_image(data_in_path, file_out):
    # Enumerate all image files
    files = list_files(data_in_path)

    # Compute and store entropy of each image
    df = pandas.DataFrame(columns=['file', 'saturation'])

    m = len(files)
    i = 0
    for file_in in files:
        i += 1
        print("{0}/{1}".format(i, m))
        df = df.append({'file': file_in, 'entropy': compute_saturation(file_in)}, ignore_index=True)

    df.to_csv(file_out)


def attack(x, model, method, label_adv, label_true, entropy_masking=False, confidence=0.99):
    _model = FoolboxKerasModelEntropy(model, bounds=(0,1), entropy_mask=entropy_masking, cache_grad_mask=True)
    if entropy_masking is True:
        _model.compute_gradient_mask(x)  # Precompute and cache gradient mask of image

    label_adv = TARGET_CLASS
    criterion = TargetClassProbability(label_adv, p=confidence)  # Targeted attack

    attacker = None
    img_adv = None
    if method == "BIM":
        attacker = LinfinityBasicIterativeAttack(_model, criterion, distance=foolbox.distances.Linfinity)

        img_adv = attacker(x, label_true, binary_search=False, epsilon=1.0, stepsize=0.004, iterations=1000)
    else:
        raise "Unkown attack!"

    return img_adv

def sample_from_dataset(num_per_label=10, num_different_labels=1000, data_in_path=None, data_out_path=None, bad_images=None, entropy_in=None):
    # Read labels
    file_labels = os.path.join(data_in_path, 'val.txt')
    labels_true = []
    with open(file_labels, mode='r') as file_in:
        labels_true = file_in.readlines()
        labels_true = [int(y.strip().split()[1]) for y in labels_true]

    # Enumerate all image files
    files = list_files(data_in_path)
    files.sort()

    # Select samples
    classes = random.sample(range(0, len(np.unique(labels_true))), num_different_labels)
    
    df = pandas.DataFrame(list(zip(files, labels_true)), columns=['file', 'label'])

    df_entropy = pandas.read_csv(entropy_in)    # Add mean entropy of each image
    df["entropy"] = df_entropy["entropy"]

    df_final = pandas.DataFrame(columns=['file', 'label', 'entropy'])

    if bad_images is not None:  # Remove "bad images" from the dataframe
        with open(bad_images, 'r') as f_in:
            bad_files = f_in.readlines()
            bad_files = [data_in_path  + "/" + os.path.basename(i.strip()) for i in bad_files]
            df = df[~df['file'].isin(bad_files)]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # Shuffle rows

    for y in classes:
       df_final = df_final.append(df[df['label'] == y].head(num_per_label), ignore_index=True)

    # Create data.csv file
    with open(os.path.join(data_out_path, 'data.csv'), mode='w') as file_out_csv:
        header = ['original', 'original2', 'bim', 'bim_entropy', 'cw', 'cw_entropy', 'label_true', 'label_adv', 'entropy']
        writer = csv.DictWriter(file_out_csv, fieldnames=header)
        writer.writeheader()

        for _, row in df_final.iterrows():
            # Load original image and apply some preprocessing (e.g. resizing)
            file_img = row.file
            label_true = row.label
            img_original = utils.open_image_as_tensor(file_img)
            _img_original = utils.tensor2array(img_original)
            #_img_original = utils.open_image_properly(file_img, arch='inception')

            # Generate random file names
            file_name = os.path.splitext(os.path.basename(file_img))[0]
            x = random.randint(42, 4242)
            file_original_out = os.path.join(data_out_path, file_name + str(x) + '.png')
            file_original2_out = os.path.join(data_out_path, file_name + str(x + 2) + '.png')
            file_bim_out = os.path.join(data_out_path, file_name + str(x - 1) + '.png')
            file_bim_entropy_out = os.path.join(data_out_path, file_name + str(x + 1) + '.png')
            file_cw_out = os.path.join(data_out_path, file_name + str(x + 3) + '.png')
            file_cw_entropy_out = os.path.join(data_out_path, file_name + str(x + 4) + '.png')

            # Save image
            skimage.io.imsave(file_original_out, _img_original)
            skimage.io.imsave(file_original2_out, _img_original)
    
            # Generate random target label
            label_adv = random.randint(0, NUM_CLASSES)
            while label_adv == label_true:
                label_adv = random.randint(0, NUM_CLASSES)

            # Create new entry in data.csv
            writer.writerow({'original': file_original_out, 'original2': file_original2_out, 'bim': file_bim_out, 'bim_entropy': file_bim_entropy_out, 'cw': file_cw_out, 'cw_entropy': file_cw_entropy_out, 'label_true': label_true, 'label_adv': label_adv, 'entropy': row.entropy})


def create_mturk_batch(file_data_in, file_batch_out, file_ground_truth_out):
    header = ['image_a', 'image_b', 'entropy', 'method']
    df_out = pandas.DataFrame(columns=header)

    # Read input and select important columns
    df = pandas.read_csv(file_data_in)
    df = df[['original', 'original2', 'bim', 'bim_entropy', 'cw', 'cw_entropy', 'entropy']]

    # Generate HITs (permute columns)
    for _, row in df.iterrows():
        df_out = df_out.append(pandas.Series(random.sample([row.original, row.bim], 2) + [row.entropy, "BIM"], index=header), ignore_index=True)
        df_out = df_out.append(pandas.Series(random.sample([row.original, row.bim_entropy], 2) + [row.entropy, "BIM_ENTROPY"], index=header), ignore_index=True)
        df_out = df_out.append(pandas.Series(random.sample([row.original, row.original2], 2) + [row.entropy, "NONE"], index=header), ignore_index=True)
    
    # Permute rows
    df_out = df_out.sample(frac=1.0)

    # Create ground truth
    with open(file_ground_truth_out, mode='w') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(["image_a_is_original", "entropy"])

        for _, row in df_out.iterrows():
            if (len(df[df['original'] == row.image_a]) > 0 and len(df[df['original2'] == row.image_b]) > 0) or (len(df[df['original2'] == row.image_a]) > 0 and len(df[df['original'] == row.image_b]) > 0):
                writer.writerow([-1, row.entropy, row.method])
            elif len(df[df['original'] == row.image_a]) > 0:
                writer.writerow([1, row.entropy, row.method])
            else:
                writer.writerow([0, row.entropy, row.method])

    # Remove entropy column
    df_out = df_out.drop("entropy", axis=1)

    # Save data as .csv file
    with open(file_batch_out, mode='w') as file_out:
        df_out.to_csv(file_out, header=True, sep=',', index=False)

def create_adversarial(file_in, file_bim_out, file_bim_entropy_out, file_cw_out, file_cw_entropy_out, label_adv, label_true, model):
    # Load original image
    img_original = utils.open_image_as_tensor(file_in)
    _img_original = utils.tensor2array(img_original)
    #_img_original = utils.open_image_properly(file_in, arch='inception')

    labels_adv = []
    while len(labels_adv) < 4:   # If many different labels failed, we skip this sample
        # Try a different label
        if len(labels_adv) > 0:
            label_adv = random.randint(0, NUM_CLASSES)
            while label_adv == label_true or label_adv in labels_adv:
                label_adv = random.randint(0, NUM_CLASSES)
            
            labels_adv.append(label_adv)
        else:
            labels_adv = [label_adv]

        # Perform adversarial attack
        img_bim = attack(_img_original, model, "BIM", label_adv, label_true, entropy_masking=False)
        if img_bim is None:
            continue
        img_bim_entropy = attack(_img_original, model, "BIM", label_adv, label_true, entropy_masking=True)
        if img_bim_entropy is None:
            continue

        # Save adversarial images
        skimage.io.imsave(file_bim_out, img_bim)
        skimage.io.imsave(file_bim_entropy_out, img_bim_entropy)

        break

def process_data(file_data_in, index_start=0, index_end=0, model=None):
    df = pandas.read_csv(file_data_in)

    if model is None:
        model = load_model()

    # Compute end
    index_end = len(df) if index_end == None else index_end

    # Work on requested rows
    i = 0
    m = index_end - index_start
    for _, row in df.loc[range(index_start, index_end)].iterrows():
        i += 1
        print("{0}/{1}".format(i, m))
        create_adversarial(row.original, row.bim, row.bim_entropy, row.cw, row.cw_entropy, row.label_adv, row.label_true, model)

def find_bad_images(data_in_path, file_out):
    files = list_files(data_in_path)
    
    bad_images = []
    for file_img in files:
        img_original = np.asarray(Image.open(file_img))

        if len(img_original.shape) != 3:
            bad_images.append(file_img)
        else:
            if img_original.shape[0] == 1 or img_original.shape[1] == 1 or img_original.shape[2] == 1:
                bad_images.append(file_img)
    
    with open(file_out, 'w') as f_out:
        f_out.write('\n'.join(bad_images))

