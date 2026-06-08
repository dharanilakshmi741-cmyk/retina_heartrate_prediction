import cv2
import numpy as np

from detection.facemesh_detector import FaceMeshDetector

from cnn.efficientphys import EfficientPhys

from processing.blink_detector import calculate_ear
from processing.filter import bandpass_filter
from processing.bpm import calculate_bpm


detector = FaceMeshDetector()

model = EfficientPhys()

cap = cv2.VideoCapture(0)

stable_bpm = 72

closed_frames = 0

signal = []


def create_graph(signal):

    graph = np.zeros((100, 220, 3), dtype=np.uint8)

    if len(signal) < 5:
        return graph

    signal = np.array(signal[-220:])

    signal = signal - np.min(signal)

    max_value = np.max(signal)

    if max_value != 0:

        signal = signal / max_value

    signal = (signal * 80).astype(np.int32)

    for i in range(1, len(signal)):

        cv2.line(
            graph,
            (i - 1, 90 - signal[i - 1]),
            (i, 90 - signal[i]),
            (0, 255, 0),
            2
        )

    return graph


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (430, 320))

    canvas = np.ones((360, 760, 3), dtype=np.uint8) * 235

    bpm = stable_bpm

    frequency = round(bpm / 60, 2)

    preview = np.zeros((80, 80, 3), dtype=np.uint8)

    landmarks = detector.detect_landmarks(frame)

    if landmarks:

        left_eye, right_eye = detector.get_eye_points(
            landmarks,
            frame.shape
        )

        all_points = left_eye + right_eye

        x = [p[0] for p in all_points]

        y = [p[1] for p in all_points]

        x1 = max(min(x) - 30, 0)

        y1 = max(min(y) - 20, 0)

        x2 = min(max(x) + 30, frame.shape[1])

        y2 = min(max(y) + 25, frame.shape[0])

        eye_roi = frame[y1:y2, x1:x2]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        try:

            preview = cv2.resize(
                eye_roi,
                (80, 80)
            )

        except:
            pass

        ear = calculate_ear(left_eye)

        if ear < 0.18:

            closed_frames += 1

        else:

            closed_frames = 0

        if closed_frames > 10:

            bpm = 0

            stable_bpm = 0

            frequency = 0

            signal = []

            model.signal_buffer.clear()

        else:

            signal = model.predict_signal(
                eye_roi
            )

            filtered = bandpass_filter(signal)

            new_bpm = calculate_bpm(filtered)

            if new_bpm != 0:

                stable_bpm = int(
                    (
                        stable_bpm * 0.94
                    ) +
                    (
                        new_bpm * 0.06
                    )
                )

                if stable_bpm > 110:
                    stable_bpm = 110

                if stable_bpm < 55:
                    stable_bpm = 55

            bpm = stable_bpm

            frequency = round(
                bpm / 60,
                2
            )

            signal = filtered

    canvas[10:330, 10:440] = frame

    canvas[20:100, 500:580] = preview

    cv2.putText(
        canvas,
        f"Freq: {frequency}",
        (610, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (40, 40, 40),
        1
    )

    cv2.putText(
        canvas,
        f"Heart rate: {bpm} bpm",
        (610, 70),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (40, 40, 40),
        1
    )

    graph1 = create_graph(signal)

    canvas[120:220, 500:720] = graph1

    graph2 = cv2.GaussianBlur(
        graph1,
        (7, 7),
        0
    )

    canvas[230:330, 500:720] = graph2

    cv2.putText(
        canvas,
        "Press ESC to Exit",
        (20, 350),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (80, 80, 80),
        1
    )

    cv2.imshow(
        "RETINA HEART RATE AI",
        canvas
    )

    key = cv2.waitKey(1)

    if key == 27:
        break


cap.release()

cv2.destroyAllWindows()