import utils
import cv2
import numpy as np

from learnedwatershed.utils import prediction_utils, display_utils, preprocessing_utils, graph_utils
from learnedwatershed import ChopinNet


def segment_particles(detections):
    receptive_field_shape = (23, 23)

    chopin = ChopinNet.Chopin()
    chopin.build(receptive_field_shape)
    chopin.initialize_session()

    chanvese_segmentations = []
    lws_segmentations = []

    for particle in detections:

        particle = np.squeeze(particle)

        blur = cv2.GaussianBlur(particle.astype(np.uint8), (5, 5), 0)

        ret, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = np.zeros(thres.shape)
        mask[20:100, 20:100] = 1

        seg, phi, its = utils.chanvese(thres, mask, max_its=1000, alpha=1.0)

        chanvese_boundaries = utils.ids_to_segmentation(seg).astype(np.uint8)
        chanvese_boundaries[chanvese_boundaries == 1] = 255

        chanvese_segmentations.append(chanvese_boundaries)

        seeds = [np.unravel_index(phi.argmax(), phi.shape), np.unravel_index(phi.argmin(), phi.shape)]

        i_a = np.stack((particle, chanvese_boundaries), -1)

        i_a = preprocessing_utils.pad_for_window(i_a,
                                                 chopin.receptive_field_shape[0],
                                                 chopin.receptive_field_shape[1])

        graph = graph_utils.prims_initialize(particle)

        msf = prediction_utils.minimum_spanning_forest(chopin, i_a, graph, seeds)

        segmentation = display_utils.assignments(particle, msf, seeds)

        lws_segmentations.append(segmentation)

    return np.stack(chanvese_segmentations), np.stack(lws_segmentations)
