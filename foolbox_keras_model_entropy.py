# -*- coding: utf-8 -*-
import numpy as np
import skimage.filters.rank
import skimage.morphology
import skimage.color
import skimage.exposure
import skimage.draw
import skimage.filters
import skimage.io
import matplotlib.pyplot as plt
import foolbox
from foolbox.models import DifferentiableModel, KerasModel

import utils


class FoolboxKerasModelEntropy(DifferentiableModel):
    def __init__(self, model, bounds, channel_axis=3, preprocessing=(0, 1), predicts='probabilities', entropy_mask=True, cache_grad_mask=False):
        super(FoolboxKerasModelEntropy, self).__init__(bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing)

        self.entropy_mask = entropy_mask
        self.grad_mask = None
        self.cache_grad_mask = cache_grad_mask
        self.keras_model = KerasModel(model, bounds, channel_axis, preprocessing, predicts)        

    def compute_gradient_mask(self, image):
        gray = skimage.color.rgb2gray(image)
        mask = skimage.filters.rank.entropy(gray, skimage.morphology.disk(3))
        
        low = mask < 4.2
        high = mask >= 4.2

        mask[low] = 0.0
        mask[high] = 1.0
        
        self.grad_mask = np.broadcast_to(mask.reshape(mask.shape[0]*mask.shape[1],1), (mask.shape[0]*mask.shape[1], image.shape[2])).reshape(image.shape)

    def __mask_gradient(self, grad, image):
        if self.entropy_mask is True:
            if self.cache_grad_mask is True:
                return grad * self.grad_mask
            else:
                mask = utils.image2mask(image)
                mask = np.broadcast_to(mask.reshape(mask.shape[0]*mask.shape[1],1), (mask.shape[0]*mask.shape[1], image.shape[2])).reshape(image.shape)

                return grad * mask
        else:
            return grad

    def predictions_and_gradient(self, image, label):
        """Calculates predictions for an image and the gradient of
        the cross-entropy loss w.r.t. the image.
        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        label : int
            Reference label used to calculate the gradient.
        Returns
        -------
        predictions : `numpy.ndarray`
            Vector of predictions (logits, i.e. before the softmax) with
            shape (number of classes,).
        gradient : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the image. Will
            have the same shape as the image.
        See Also
        --------
        :meth:`gradient`
        """

        pred, grad = self.keras_model.predictions_and_gradient(image, label)

        return pred, self.__mask_gradient(grad, image)

    def num_classes(self):
        return self.keras_model.num_classes()

    def batch_predictions(self, images):
        return self.keras_model.batch_predictions(images)

    def backward(self, gradient, image):
        """Backpropagates the gradient of some loss w.r.t. the logits
        through the network and returns the gradient of that loss w.r.t
        to the input image.
        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t the image.
        See Also
        --------
        :meth:`gradient`
        """

        grad = self.keras_model.backward(gradient, image)
        
        return self.__mask_gradient(grad, image)
