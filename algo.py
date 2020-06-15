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


HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

INPUT_STREAM = "resources/Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = None
MODEL = "samples/model/IR/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco_2018_01_28.xml"
# MODEL = "samples/model/IR/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.xml"

def build_argparser():
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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.4,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def draw_bounding_boxes(frame, result, args, width, height):
    person_count = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            person_count += 1
    return frame, person_count


def infer_on_stream(args):
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    infer_network = Network()
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
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
    fluc_count = 0
    prev_person_count = 0
    frame_id = 0

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

            p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            
            infer_network.exec_net(p_frame)

            if (infer_network.wait() == 0):
                if key_pressed == 27:
                    break

                result = infer_network.get_output()
                
                frame, person_count = draw_bounding_boxes(frame, result, args, width, height)

                # Reset frame counter on every fluctuation in detection
                if person_count != prev_person_count:
                    fluc_count += 1
                    frame_id = 0

                # True positive if new person detection lasts at least 44 frames 
                if person_count > 0:
                    start_time = time.time()
                    fluc_count = 0
                    if frame_id == 44:
                        client.publish("person", json.dumps({"total": total}))
                        total = total + person_count - prev_person_count
                        
                        # duration = time.time() - start_time
                        ## for some reason this has to be here for the total to be published correctly
                        client.publish("person/duration", json.dumps({"duration": duration}))
                        # cv2.putText(frame, 'Total Count: {}'.format(total), (0, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (0, 0, 255), 2)

                if person_count < 1 and frame_id < 20:
                    duration = time.time() - start_time

                prev_person_count = person_count

                cv2.putText(frame, 'Person(s) In Frame: {}'.format(person_count), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (0, 0, 255), 2)
                cv2.putText(frame, 'Frame ID: {}'.format(frame_id), (0, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (0, 0, 255), 2)
                
                client.publish("person", json.dumps({"count": person_count}))

            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)
    

if __name__ == '__main__':
    main()
