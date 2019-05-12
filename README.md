# Shopper Gaze Monitor

| Details               |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  Python* 3.5 |
| Time to Complete:     |  30 min     |

This reference implementation is also [available in C++](https://github.com/intel-iot-devkit/reference-implementation-private/blob/master/shopper-gaze-monitor/README.md).

![images](./images/output.png)

## Introduction

This shopper gaze monitor application uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit and the Intel® Deep Learning Deployment Toolkit. It is designed for a retail shelf mounted camera system that counts the number of passers-by that look towards the display vs. the number of people that pass by the display without looking. 

It is intended to provide real-world marketing statistics for in-store shelf-space advertising.

## How it works

The application uses a video source, such as a camera, to grab frames and then uses two different Deep Neural Networks (DNNs) to process the data. The first network looks for faces and then if successful is counted as a "Shopper" 

A second neural network is then used to determine the head pose detection for each detected face. If the person's head is facing towards the camera, it is counted as a "Looker"

The shopper and looker data are sent to a local web server using the Paho* MQTT C client libraries.

![images](./images/arch_diagram.png)

The program creates two threads for concurrency:

 * Main thread that performs the video i/o, processes video frames using the trained neural network.
 * Worker thread that publishes MQTT messages.
	
		
## Requirements

### Hardware

* 6th to 8th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics

### Software

* Ubuntu 16.04

* OpenCL™ Runtime Package

  **Note:** We recommend using a 4.14+ kernel to use this software. Run the following command to determine your kernel version:

      uname -a
  
* Intel® Distribution of OpenVINO™ toolkit 2019 R1 Release

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux for more information about how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

You will need the OpenCL™ Runtime package if you plan to run inference on the GPU. It is not mandatory for CPU inference.

### Install Python* dependencies
 
    sudo apt install python3-pip

    sudo apt-get install mosquitto mosquitto-clients

    pip3 install jupyter

    pip3 install numpy

    pip3 install paho-mqtt

## Download the model

This application uses the **face-detection-adas-0001** and **head-pose-estimation-adas-0001** Intel® model, that can be downloaded using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that is used by the application. 

Steps to download **.xml** and **.bin** files:

- Go to the **model_downloader** directory using following command: 
    ```
    cd /opt/intel/openvino/deployment_tools/tools/model_downloader
    ```
- Specify which model to download with --name:

    ```
    sudo ./downloader.py --name face-detection-adas-0001
    sudo ./downloader.py --name head-pose-estimation-adas-0001
    ```
- To download the model for FP16, run the following commands:

    ```
    sudo ./downloader.py --name face-detection-adas-0001-fp16
    sudo ./downloader.py --name head-pose-estimation-adas-0001-fp16
    ```
The files will be downloaded inside the following directories:
 -   ``/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/    ``     
 -   ``/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/`` 

## Setting the Build Environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
    
    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

## Run the application

Go to `shopper-gaze-monitor-python` directory:
    
    cd <path-to-shopper-gaze-monitor-python-directory>
    
To see a list of the various options:

    python3 main.py --help
 
### Sample Video

You can download sample video by running following commands.	

    mkdir resources
    cd resources
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4 
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4 
    wget https://github.com/intel-iot-devkit/sample-videos/blob/master/head-pose-face-detection-female.mp4 	
    cd .. 

### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at 

    /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/

Though by default the application runs on CPU, this can also be explicitly specified by -d CPU command-line argument:

	python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -i resources/face-demographics-walking-and-pause.mp4

**Note**: To get the input video from the camera, use -i cam command-line argument. For example:

	python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -i cam


### Running on the GPU

To run on the integrated Intel® GPU in 32 bit mode, use the -d GPU command-line argument:

	python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml -i resources/face-demographics-walking-and-pause.mp4 -d GPU

To use GPU in 16 bit mode, use the following command:

	python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -i resources/face-demographics-walking-and-pause.mp4 -d GPU

### Running on the Intel® Neural Compute Stick
To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:

    python3 main.py -d MYRIAD -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -i resources/face-demographics-walking-and-pause.mp4

**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

### Running on the HDDL
To run on the HDDL-R, use the `-d HETERO:HDDL,CPU ` command-line argument:

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -i resources/face-demographics-walking-and-pause.mp4 -d HETERO:HDDL,CPU

**Note:** The HDDL-R can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

### Running on the FPGA

Before running the application on the FPGA,  program the AOCX (bitstream) file. Use the setup_env.sh script from [fpga_support_files.tgz](https://clicktime.symantec.com/38YrYPLxxSqYhBQLb1eruhW7Vc?u=http%3A%2F%2Fregistrationcenter-download.intel.com%2Fakdlm%2Firc_nas%2F12954%2Ffpga_support_files.tgz) to set the environment variables.<br>
For example:

    source /home/<user>/Downloads/fpga_support_files/setup_env.sh

The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams` folder.<br><br>To program the bitstream use the below command:<br>

    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNet_Clamp.aocx

For more information on programming the bitstreams, please refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11

To run the application on the FPGA with floating point precision 16 (FP16), use the `-d HETERO:FPGA,CPU` command-line argument:

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -i resources/face-demographics-walking-and-pause.mp4 -d HETERO:FPGA,CPU

# Machine to machine messaging with MQTT

If you wish to use a MQTT server to publish data, you should set the following environment variables on the terminal before running the program.
 
    export MQTT_SERVER=localhost:1883
 
    export MQTT_CLIENT_ID=cvservice

Change the MQTT_SERVER to a value that matches the MQTT server you are connecting to.

You should change the MQTT_CLIENT_ID to a unique value for each monitoring station, so you can track the data for individual locations. For example:

    export MQTT_CLIENT_ID=zone1337

If you want to monitor the MQTT messages sent to your local server, and you have the mosquitto client utilities installed, you can run the following command in new terminal while executing the code:

    mosquitto_sub -h localhost -t shopper_gaze_monitor
