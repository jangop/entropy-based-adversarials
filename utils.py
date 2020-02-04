import math
import os
import random

import numpy as np
import skimage.color
import skimage.draw
import skimage.exposure
import skimage.filters
import skimage.filters.rank
import skimage.io
import skimage.morphology
import skimage.transform
import skimage.util
from PIL import Image

import torch
from torchvision import transforms, utils

INCEPTION_SIZE = 299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transforms_imagenet_inception = transforms.Compose([
    transforms.Resize(INCEPTION_SIZE),
    transforms.CenterCrop(INCEPTION_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def center_crop(array):
    width, height = array.shape[0], array.shape[1]
    if width == height:
        return array
    if width < height:
        diff = height - width
        start = math.floor(diff / 2)
        end = math.ceil(diff / 2)
        return array[start:-end, :]
    if width > height:
        diff = width - height
        start = math.floor(diff / 2)
        end = math.ceil(diff / 2)
        return array[:, start:-end]


def open_image_properly(path, arch):
    # Open image. Results in values between 0 and 255.
    array = skimage.io.imread(path)

    # Crop.
    array = center_crop(array)

    # Resize.
    if arch == 'inception':
        length = INCEPTION_SIZE
    else:
        raise ValueError()
    array = skimage.transform.resize(array, (length, length))

    # Convert to float. Results in values between 0 and 1.
    array = skimage.util.img_as_float(array)

    return array


def array_to_tensor(array):
    # Apply imagenet normalization.
    array = array.copy()
    array = array - IMAGENET_MEAN
    array = array / IMAGENET_STD

    # Rearrange.
    array = array.transpose([2, 0, 1])

    # Add dimension.
    array = np.expand_dims(array, 0)

    # Convert to tensor and return.
    tensor = torch.Tensor(array)

    return tensor


def image2tensor(image, arch='inception'):
    '''
    Turns a PIL image into a pytorch tensor for the
    pretrained inception network trained on imagenet.
    '''

    if arch == 'inception':
        tensor = transforms_imagenet_inception(image)
    else:
        raise ValueError()

    tensor.unsqueeze_(0)
    return tensor


def tensor2array(tensor):
    '''
    Turns a pytorch tensor, prepared for the pretrained
    inception network trained on imagenet into a numpy array
    with values between zero and one.
    '''

    # Turn tensor into numpy array.
    array = tensor.clone().detach()[0].numpy()

    # Rearrange from (c, w, h) to (w, h, c).
    array = array.transpose([1, 2, 0])

    # Denormalize.
    array *= IMAGENET_STD
    array += IMAGENET_MEAN

    # Clip.
    array = array.clip(0, 1)
    return array


def array2tensor(array):
    '''
    Turns a numpy array with values between zero and one
    into a pytorch tensor for the pretrained inception
    network trained on imagenet.
    '''

    # Clone.
    array = array.copy()

    # Add first dimension.
    dim = len(array.shape)
    if dim == 3:
        array = np.expand_dims(array, 0)
    elif dim == 4:
        pass
    else:
        raise ValueError()

    # Normalize.
    array -= IMAGENET_MEAN
    array /= IMAGENET_STD

    # Rearrange.
    array = array.transpose([0, 3, 1, 2])

    # Turn into tensor.
    tensor = torch.Tensor(array)

    return tensor


def open_image_as_tensor(path, arch='inception'):
    '''
    Opens an image using PIL and turns it into a pytorch
    tensor for the pretrained inception network trained on
    imagenet.
    '''
    image = Image.open(path)
    tensor = image2tensor(image, arch)
    return tensor


def save_tensor_as_image(tensor, path):
    '''
    Saves a pytorch tensor as an image file.
    '''
    array = tensor2array(tensor)
    skimage.io.imsave(path, array)


class ImageNetLabelDecoder():
    def __init__(self, root):
        path_classes = os.path.join(root, 'imagenet_classes.txt')
        path_synsets = os.path.join(root, 'imagenet_synsets.txt')

        # Load synsets.
        with open(path_synsets, 'r') as handle:
            synsets = handle.readlines()

        synsets = [line.strip().split() for line in synsets]
        self.names = {synset[0]: ' '.join(synset[1:]) for synset in synsets}

        # Load classes.
        with open(path_classes, 'r') as handle:
            classes = handle.readlines()

        self.classes = [line.strip() for line in classes]

    def decode(self, idx):
        try:
            return self.names[self.classes[int(idx)]]
        except TypeError:
            try:
                return [self.decode(i) for i in idx]
            except:
                raise

    def __call__(self, idx):
        return self.decode(idx)

    def print_top_predictions(self, predictions, n_predictions=5):
        # Select top scoring predictions.
        top_idx = np.argpartition(predictions, -n_predictions)[-n_predictions:]

        # Sort top scoring predictions.
        top_idx = top_idx[np.argsort(predictions[top_idx])][::-1]

        # Calculate certainties.
        sm = torch.nn.Softmax(0)
        certainties = sm(torch.Tensor(predictions))

        # Print.
        for i in top_idx:
            print('{:3d}\t{:04.2f}\t{}'.format(i, certainties[i],
                                               self.decode(i)))


def image2mask(image, size=3):
    gray = skimage.color.rgb2gray(image)
    mask = skimage.filters.rank.entropy(gray, skimage.morphology.disk(size))
    mask = skimage.exposure.equalize_hist(mask)
    mask -= mask.min()
    mask /= mask.max()
    mask = np.power(mask, 4)

    return mask


def random_mask(shape, radius):
    # Initialize empty mask.
    mask = np.zeros(shape)

    # Determine center of circle.
    (width, height) = shape
    radius = int(radius)
    x = random.randint(radius, width - radius)
    y = random.randint(radius, height - radius)

    # Draw circle.
    row_idx, col_idx = skimage.draw.circle(x, y, radius, shape)
    mask[row_idx, col_idx] = 1

    # Blur.
    mask = skimage.filters.gaussian(mask, radius * 0.1)

    return mask


def flat_idx_to_matrix_idx(idx, shape):
    (_, n_cols) = shape
    row_idx = idx // n_cols
    col_idx = idx % n_cols
    return row_idx, col_idx


def binary_mask_from_highest_values(matrix, n_ones):
    # Reduce.
    matrix = np.linalg.norm(matrix, axis=0)

    # Find indices of highest values.
    flat = matrix.flatten()
    flat = np.abs(flat)
    perm = flat.argsort()[::-1]
    highest = perm[:n_ones]
    row_idx, col_idx = flat_idx_to_matrix_idx(highest, matrix.shape)

    # Initialize empty mask and fill in ones.
    mask = np.zeros_like(matrix)
    mask[row_idx, col_idx] = 1

    return mask


def open_image_as_segments(path):
    img = skimage.io.imread(path)
    img = skimage.color.rgb2gray(img)
    segments = np.zeros_like(img, dtype=int)
    for i, val in enumerate(np.unique(img)):
        segments[img == val] = i
    return segments


def relative_total_energy_from_map(energy_map):
    energy = np.sum(energy_map) / np.prod(energy_map.shape)
    return energy
