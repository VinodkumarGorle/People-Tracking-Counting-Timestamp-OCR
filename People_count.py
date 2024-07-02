#!/usr/bin/env python

import collections
import sys
import time
import os
import numpy as np
import cv2
from IPython import display
import matplotlib.pyplot as plt
from openvino.runtime import Core

sys.path.append("utils")
import notebook_utils as utils

from scipy import ndimage


import pandas as pd
import cv2

import itertools

from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy

import Time_Filter as T

IN = []
OUT = []
count=0
time_delay = 2
# ## Download the Model

# A directory where the model will be downloaded.
base_model_dir = "model"
precision = "FP32"
# The name of the model from Open Model Zoo
detection_model_name = "person-detection-0202"

download_command = f"omz_downloader " \
                   f"--name {detection_model_name} " \
                   f"--precisions {precision} " \
                   f"--output_dir {base_model_dir} " \
                   f"--cache_dir {base_model_dir}"
#get_ipython().system(' $download_command')

detection_model_path = f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"


reidentification_model_name = "person-reidentification-retail-0287"

download_command = f"omz_downloader " \
                   f"--name {reidentification_model_name} " \
                   f"--precisions {precision} " \
                   f"--output_dir {base_model_dir} " \
                   f"--cache_dir {base_model_dir}"
#get_ipython().system(' $download_command')

reidentification_model_path = f"model/intel/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"


# ## Load model

ie_core = Core()


class Model:
    def __init__(self, model_path, batchsize=1, device="AUTO"):
        self.model = ie_core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = ie_core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        result = self.compiled_model(input)[self.output_layer]
        return result


detector = Model(detection_model_path)
# since the number of detection object is uncertain, the input batch size of reid model should be dynamic
extractor = Model(reidentification_model_path, -1)


# ## Data Processing

def preprocess(frame, height, width):
    resized_image = cv2.resize(frame, (width, height))
    resized_image = resized_image.transpose((2, 0, 1))
    input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return input_image


def batch_preprocess(img_crops, height, width):
    img_batch = np.concatenate([
        preprocess(img, height, width)
        for img in img_crops
    ], axis=0)
    return img_batch


def process_results(h, w, results, thresh=0.5):
    # The 'results' variable is a [1, 1, N, 7] tensor.
    detections = results.reshape(-1, 7)
    boxes = []
    labels = []
    scores = []
    for i, detection in enumerate(detections):
        _, label, score, xmin, ymin, xmax, ymax = detection
        # Filter detected objects.
        if score > thresh:
            # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
            boxes.append(
                [(xmin + xmax) / 2 * w, (ymin + ymax) / 2 * h, (xmax - xmin) * w, (ymax - ymin) * h]
            )
            labels.append(int(label))
            scores.append(float(score))

    if len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = np.array([])
    return np.array(boxes), np.array(scores), np.array(labels)

up={}
up_count = {}

down= {}
down_count = {}

time_out = {}
time_in = {}

inc = 0
outc = 0

IN = []
OUT = []

IN_time = {}
OUT_time = {}
ID_System_Time_Tuple = {}
ID_Video_Time_Tuple = {}
t=0

def draw_boxes(img, bbox, identities=None):

    global centroid
    global x, y,inc,outc,label
    Y_ref = 225
    Y_up = Y_ref-25
    Y_down = Y_ref+25

    for i, box in enumerate(bbox):

        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2
        )

        # LINES
        #red_line = cv2.line(img=img, pt1=(0, 120), pt2=(0, 120), color=(0, 0, 255), thickness=2, lineType=8, shift=0)
        blue_line = cv2.line(img=img, pt1=(0, 290), pt2=(800, 290), color=(255, 0, 0), thickness=2, lineType=8,shift=0)
        pink1_line = cv2.line(img=img, pt1=(0, Y_down), pt2=(800, Y_down), color=(180,105,225), thickness=1, lineType=8, shift=0)
        pink2_line = cv2.line(img=img, pt1=(0, Y_up), pt2=(800, Y_up), color=(180,105,225), thickness=1, lineType=8, shift=0)
        yellow_line = cv2.line(img=img, pt1=(0, Y_ref), pt2=(800, Y_ref), color=(0, 225, 225), thickness=2, lineType=8,shift=0)


        #x, y = (x1 + x2) // 2, (9*y1 + y2) // 10
        x, y = (x1 + x2) // 2, (y1 + y2) // 2

        # CENTROIDS
        if y < Y_up:
            centroid_red = cv2.circle(img=img, center=(x, y), radius=3, color=(0, 0, 225), thickness=3)
        if y > Y_down:
            centroid_green = cv2.circle(img=img, center=(x, y), radius=3, color=(0, 225, 0), thickness=3)
        if y in range(Y_up,Y_ref):
            centroid_blue = cv2.circle(img=img, center=(x, y), radius=3, color=(225, 0, 0), thickness=3)
        if y in range(Y_ref, Y_down):
            centroid_black = cv2.circle(img=img, center=(x, y), radius=3, color=(0, 0, 0), thickness=3)

        # COUNT-PEOPLE
        if (y in range(Y_up, Y_down)) :

            if (y in range(Y_up, Y_ref+1)) :

                if not label in up.keys() :
                    up[label] = []
                    up[label].append([y, time.time()])
                    up_count[label] = 0
                else :
                    if label in down.keys():
                        down.pop(label)
                        OUT.append(label)
                        down_count.pop(label)
                        outc = outc + 1


                        id_time = time.localtime()
                        id_time = time.strftime("%H:%M:%S", id_time)
                        if not label in ID_System_Time_Tuple.keys() :
                            ID_System_Time_Tuple[label] = []

                        ID_System_Time_Tuple[label].append(id_time)

                        img_file = "tmp.png"
                        cv2.imwrite(img_file, img)

                        video_time = T.run_paddle_ocr(img_file)
                        if not label in ID_Video_Time_Tuple.keys():
                            ID_Video_Time_Tuple[label] = []
                        ID_Video_Time_Tuple[label].append(video_time[0])


                        time_out[label] = []
                        if label in up.keys():
                            tf = time.time() - up[label][0][1]
                            time_out[label].append(tf)
            else :
                if (y in range(Y_ref+1, Y_down)) :
                    if not label in down.keys():
                        down[label] = []
                        down[label].append([y, time.time()])

                        down_count[label] = 0
                    else:

                        if label in up.keys():
                            up.pop(label)
                            IN.append((label))
                            up_count.pop(label)
                            inc = inc + 1

                            id_time = time.localtime()
                            id_time = time.strftime("%H:%M:%S", id_time)
                            if not label in ID_System_Time_Tuple.keys():
                                ID_System_Time_Tuple[label] = []
                            ID_System_Time_Tuple[label].append(id_time)

                            img_file = "tmp.jpg"
                            cv2.imwrite(img_file , img)

                            video_time = T.run_paddle_ocr(img_file)
                            if not label in ID_Video_Time_Tuple.keys():
                                ID_Video_Time_Tuple[label] = []
                            ID_Video_Time_Tuple[label].append(video_time[0])



                            time_in[label] = []
                            if label in down.keys():
                                tf = time.time() - down[label][0][1]
                                time_in[label].append(tf)
                            if label in time_in and label in time_out:
                                t = abs(float(time_in[label][0] - float(time_out[label][0])))
                                if label in IN and label in OUT and t < 2.0 :
                                    IN.remove(label)
                                    OUT.remove(label)
                                    #IN_time.pop(label)
                                    #del IN_time[label][0]
                                    #OUT_time.pop(label)
                                    print("removed ID : - " , label)
                                    inc = inc - 1
                                    outc = outc - 1

        else :
            if label in up.keys():
                up.pop(label)
                up_count.pop(label)

            if label in down.keys():
                down.pop(label)
                down_count.pop(label)

    print('came-IN = ', inc, "        Went-Out = ", outc)
    print("IN ids :- ", IN)
    print("OUT ids :-",OUT)
    #print("ID_System_Time_Tuple : - " , ID_System_Time_Tuple )
    print("ID_Video_Time_Tuple : - ", ID_Video_Time_Tuple)
    print()
    return img
def cosin_metric(x1, x2):
    """
    Calculate the consin distance of two vector

    Parameters
    ----------
    x1, x2: input vectors
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# ## Main Processing Function
# Main processing function to run person tracking.
def run_person_tracking(source=0, flip=False, use_popup=False, skip_first_frames=0):

    player = None
    try:
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(
            source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            frame = ndimage.rotate(frame,9)     # Rotate Frame Horizontally


            if frame is None:
                print("Source ended")
                break
            # Resize the image and change dims to fit neural network input.
            h, w = frame.shape[:2]
            input_image = preprocess(frame, detector.height, detector.width)

            # Measure processing time.
            start_time = time.time()
            # Get the results.
            output = detector.predict(input_image)
            stop_time = time.time()
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time

            # Get poses from detection results.
            bbox_xywh, score, label = process_results(h, w, results=output)

            img_crops = []
            for box in bbox_xywh:
                x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
                img = frame[y1:y2, x1:x2]
                img_crops.append(img)

            # lines
            # red_line = cv2.line(img=frame, pt1=(400, 280), pt2=(460, 280), color=(0, 0, 255), thickness=2, lineType=8, shift=0)
            # Get re-identification feature of each person.
            if img_crops:
                # preprocess
                img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
                features = extractor.predict(img_batch)
            else:
                features = np.array([])

            # Wrap the detection and re-identification results together
            bbox_tlwh = xywh_to_tlwh(bbox_xywh)
            detections = [
                Detection(bbox_tlwh[i], features[i])
                for i in range(features.shape[0])
            ]
            # predict the position of tracking target 
            tracker.predict()

            # update tracker
            tracker.update(detections)

            # update bbox identities
            outputs = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
                track_id = track.track_id
                outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int64))

            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
            # draw box for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                frame = draw_boxes(frame, bbox_xyxy, identities)


            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 2000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
            
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)


   # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()

NN_BUDGET = 100
MAX_COSINE_DISTANCE = 0.7  # threshold of matching object
metric = NearestNeighborDistanceMetric(
    "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
)
tracker = Tracker(
    metric,
    max_iou_distance=0.7,
    max_age=70,
    n_init=3
)
# ### Run Person Tracking on a Video File

video_file = "Single-Image.png"    #Image
#video_file = "orchestrated_1.mp4"    #VIDEO
#video_file = "rtsp://admin:123456@192.168.10.220:554/mode=real&idc=4&ids=2"  #RTSP

run_person_tracking(source = video_file, flip=False, use_popup=True)

