import cv2
import numpy as np


class EfficientPhys:

    def __init__(self):

        self.signal_buffer = []

        self.prev_value = None

    def predict_signal(self, roi):

        if roi is None:
            return []

        if roi.size == 0:
            return []

        roi = cv2.resize(
            roi,
            (160, 80)
        )

        green = roi[:, :, 1]

        green = cv2.GaussianBlur(
            green,
            (5, 5),
            0
        )

        value = np.mean(green)

        if self.prev_value is None:

            self.prev_value = value

        pulse = value - self.prev_value

        self.prev_value = value

        pulse = pulse * 10

        self.signal_buffer.append(pulse)

        if len(self.signal_buffer) > 300:
            self.signal_buffer.pop(0)

        return self.signal_buffer