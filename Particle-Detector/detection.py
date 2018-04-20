from __future__ import division
from __future__ import print_function

# Handle Imports
import cv2
import numpy as np
from keras.models import model_from_json

import utils


def detect_particles(imgs, pyramid, model_path, suppression=0.5):
    json_file = open(model_path + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_path + ".h5")
    print("Loaded model from disk")

    height = model.input_shape[1]
    width = model.input_shape[2]

    for original in imgs:

        detections = []
        coordinates = []

        # Image Pyramid.
        for scale, stride in pyramid:

            # Resize the image.
            img = cv2.resize(original, (int(original.shape[1] * scale), int(original.shape[0] * scale)))

            # Store the coordinates of each slice.
            coords = []

            # Store all of the images for prediction.
            slices = []

            # Gather images and coordinates.
            for x in xrange(0, img.shape[1] - width, stride):
                for y in xrange(0, img.shape[0] - height, stride):
                    coords.append([x, y, x + width, y + height])
                    slices.append(img[y:y + height, x:x + width, ...])

            # Stack coordinates and images into numpy arrays.
            coords = np.stack(coords)
            slices = np.stack(slices)

            # Augment images for grayscale prediction.
            slices = np.expand_dims(slices, -1)

            # Classify images.
            preds = model.predict(slices, verbose=1)

            # Extract Box Coordinates.
            coords = coords[np.isin(preds[:, 1], 1)] // scale

            detections.append(slices[np.isin(preds[:, 1], 1)] * 255)

            print(len(coords))

            coordinates.append(coords)

    if len(coordinates) != 0:
        coordinates = np.concatenate(coordinates).astype(np.uint32)

    detections = np.concatenate(detections)

    if suppression:
        pick = utils.non_max_suppression_fast(coordinates, suppression)
        detections = detections[pick]
        coordinates = coordinates[pick]

    return coordinates, detections