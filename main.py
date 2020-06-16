"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

INPUT_STREAM = "resources/Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = None
# MODEL = "samples/model/IR/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco_2018_01_28.xml"
MODEL = "samples/model/IR/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.xml"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=MODEL,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", type=str, default=INPUT_STREAM,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    """
    Create MQTT client

    :return: MQTT client
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_bounding_boxes(frame, result, args, width, height):
    """
    Draw bounding boxes based on model detections
    for specified confidence thresholds (default is 50%).
    
    :param frame: Pre-processed frame for inference
    :param result: Result of inference from IENetwork
    :param args: Command line arguments parsed by `build_argparser()`
    :param width: Width of pre-processed frame
    :param height: Height of pre-processed frame
    : return: frame and person 
    """
    person_count = 0
    for box in result[0][0]:
        conf = box[2]
        # draw boxes for detections above 50%
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            # new object has been detected
            person_count += 1
    return frame, person_count


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    infer_network = Network()
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    # Handle input stream
    single_image_mode = False
    if args.input == 'cam':
        args.input = 0
    elif args.input.endswith('.mp4') or args.input.endswith('.avi'):
        single_image_mode = False
    elif args.input.endswith('.jpg') or args.input.endswith('.png'):
        single_image_mode = True
    else:
        exit(1)

    total = 0
    start_time = 0
    duration = 0
    prev_person_count = 0
    frame_id = 0
    frame_tracker = 0

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    width = int(cap.get(3))
    height = int(cap.get(4))

    while cap.isOpened():
        frame_exist, frame = cap.read()
        frame_id += 1

        if not frame_exist:
            break

        if single_image_mode:
            cv2.imwrite('output_image.PNG', frame)
    
        if not single_image_mode:
            key_pressed = cv2.waitKey(60)

            # Pre-process frame
            p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            
            # Start inference on pre-processed frame
            infer_network.exec_net(p_frame)

            # Wait for inference results
            if (infer_network.wait() == 0):
                if key_pressed == 27:
                    break

                result = infer_network.get_output()
                frame, person_count = draw_bounding_boxes(frame, result, args, width, height)

                # For every new detection in frame increase the total count
                if person_count > prev_person_count:
                    start_time = time.time()
                    total = total + person_count
                    client.publish("person", json.dumps({"total": total}))
                    # keep track of frame at the point of detection
                    frame_tracker = frame_id

                # If person is detected in the next 5 frames,
                # there was a fluctuation, hence treat as false negative
                # otherwise, person has exited the frame
                if person_count < prev_person_count and frame_id - frame_tracker > 5:
                    duration = int(time.time() - start_time)
                    client.publish("person/duration", json.dumps({"duration": duration}))

                # Keep track of previous count detected
                prev_person_count = person_count
                client.publish("person", json.dumps({"count": person_count}))

                # Display meta data on frame
                cv2.putText(frame, 'Person(s) In Frame: {}'.format(person_count), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, 'Duration of Previous Person: {}'.format(duration), (0, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, 'Total Frames Processed: {}'.format(frame_id), (0, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 2)
                
            # Flush frame to FFMPEG Server
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
    
    # Close video capture object 
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    args = build_argparser().parse_args()
    client = connect_mqtt()
    infer_on_stream(args, client)
    

if __name__ == '__main__':
    main()
