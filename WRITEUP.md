# PEOPLE COUNTER APP

## Edge Computing

Edge computing is a distributed computing paradigm that brings computation and data storage closer to the location where it is needed, to improve response times and save bandwidth. [Read more](https://en.wikipedia.org/wiki/Edge_computing).

Today, business models often involve geographically distributed branch offices, manufacturing facilities, retail locations, or services. Edge computing systems place edge servers, Internet of Things (IoT) edge devices, and data processing at many different points, wherever resources are needed most. As a result, businesses can converge analytics, media, and networking workloads and bring them closer to the people and processes that rely on them, enabling real-time insights that can transform operations and experiences. Because distributed edge computing collects and analyzes data locally, it offers several advantages: 
meeting data locality and privacy needs, reducing transmission costs, achieving ultralow latency for real-time results. [Read more](https://www.intel.com/content/www/us/en/edge-computing/overview.html).


## What is this project about?

This project, People Counter App is a Computer Vision Application that utilizes the OpenVINO™ Toolkit (Intel's Deep Learning Toolkit, based on Convolutional Neural Networks (CNNs), that extends computer vision (CV) workloads across Intel® hardware) in the emulation of human vision, thereby providing insights such as how many people are detected in an input stream, how long were they there for and what is the average time spent. This is AI at the Edge. The [README]('./README.md') shows how this works.


## Explaining AI Models

### `Note:` Before the completion of this project, several object detection models were tested in the detection of people in a video stream. [See Model Research Section](#model-research).

[Neural networks](https://pathmind.com/wiki/neural-network) are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. Deep learning is the name used for “stacked neural networks”; that is, networks composed of several layers. The adjective "deep" in deep learning comes from the use of multiple layers.

A [Machine Learning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model) (or Deep Learning) Model is a file that has been trained to recognize certain types of patterns. Typically, you train (or say learn) a model over a set of data, providing it an algorithm(s) that it can use to reason over and learn from those data. Once you have trained the model, you can use it to reason over data that it hasn't seen before, and make predictions about those data. 

## Explaining Custom Layers

AI models are trained using Deep Learning (DL) frameworks such as Tensorflow, Caffe, Pytorch, etc. Before these models (trained using a supported DL framework) are used with the Inference Engine (IE) of the OpenVINO™ Toolkit, they have to be converted to an Intermediate Representation (IR - an abstraction of the different frameworks and a model format that the IE understands). The Model Optimizer (MO - part of the Toolkit) searches for each layer of the input model from a list of [standard layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) known to the MO before building the model's internal representation, optimizing the model, and producing the IR.

![](./images/convert_model_to_ir_workflow.png)
#### <center>Source: [OpenVINO Docs](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)</center>

When a layer of the input model is not found in the list of standard layers, it is classified as a Custom Layer. Such layers must be registered with the MO.

The process behind converting (registering) custom layers as detailed in the documentation for the OpenVINO™ Toolkit typically involves:
- Registering the custom layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. 
[More details in the docs](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.htm)

## Comparing Model Performance

- The initail [(test) models](#model-research) were downloaded from the Tensorflow Model Zoo and were significantly larger in size compared to the [Intel Pretrained Model](#selected-model) that was selected.
- The [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model was the most accurate in detection of persons in the frames.

## Model Use Cases

Some of the potential use cases of the people counter app are:
- Social Health (Covid-19)- Individual gatherings:  
  The Nigerian government declared a maximum of 25 person(s) for any social gathering (at a single time) in its major commercial areas.
  An alarm can be triggered when the number of people gets above the maximum number to alert the monitoring personnels.

- Event Centers:  
  Halls that host social events often have a max capacity; event planners can install IOT devices that have embedded in them the People Counter App for analytics purposes.

- Unmanned or Self Driving Buses:  
  Passengers should be able to board self driving buses. The People Counter App can be installed in the buses which will close the doors when the required number of persons are already onboard.

## Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

- Lighting & Visibility:  
  A computer vision app requires proper lighting to be able to make detections. A cloudy or dark frame may pass off an detection.

- Image size:  
  The size of an image can be dependent on its resolution - high res images means more accurate detections and vice versa.

## Model Research

While investigating potential people counter models, I tried each of the following three models:
  
- Model 1: ssdlite_mobilenet_v2_coco
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments 
  ```
  python3 <path-to-mo_tf.py> \
  --input_model ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb \
  --output_dir <output-directory> \
  --model_name ssdlite_mobilenet_v2_coco_2018_05_09 \
  --transformations_config <json-transformation-config> \
  --tensorflow_object_detection_api_pipeline_config ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config 
  ```
  - The model was insufficient for the app because of it's instable detections and sometimes wrong detection of(
    s) in the input stream
  - I tried to improve the model for the app by increasing the confidence treshold, but this was counter productive for cases where some person(s) had a low confidence prediction and ditto for reducing the confidence threshold.
  
- Model 2: ssd_inception_v2_coco
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments 
  ```
  python3 <path-to-mo_tf.py> \
  --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb \
  --output_dir <output-directory> \
  --model_name ssd_inception_v2_coco_2018_01_28 \
  --transformations_config <json-transformation-config> \
  --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config 
  ```
  - The model was insufficient for the app because of it's inability to detect all of the person(
    s) in the input stream
  - I also tried to improve the model for the app by reducing the confidence treshold.

- Model 3: ssd_mobilenet_v2_coco
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments 
  ```
  python3 <path-to-mo_tf.py> \
  --input_model ssd_mobilenet_v2_coco_2018_01_28/frozen_inference_graph.pb \
  --output_dir <output-directory> \
  --model_name ssd_mobilenet_v2_coco_2018_01_28 \
  --transformations_config <json-transformation-config> \
  --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_01_28/pipeline.config 
  ```
  - Even though, this had the best accuracy amongst the three, the model was insufficient for the app because of it's low to zero confidence in it's detection of person(s) in black clothings and turned backwards in the input stream provided.
  - I tried to improve the model for the app by reducing the confidence treshold and modifying the app code to notice when there is a fluctuation in detections by adjusting the number of frames for each detection. Doing this, I was able to get the total number of people in the input stream, but this was not sufficient to get the duration of each person detected.

### Selected Model

I resorted to using an [Intel Pretrained Model](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/pretrained-models.html) which proved to have the best accuracy in the dection of person(s): [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html).

The Toolkit provides a downloader that you can use to download Intel pretrained models which you can find in the toolkit directory:
`deployment_tools/open_model_zoo/tools/downloader`.

After installing all the requirements, donwload the pretrained model:
```
sudo ./downloader.py --name person-detection-retail-0013 -o <output-directory>
```

## Running the App
Check the [README.md](./README.md)
