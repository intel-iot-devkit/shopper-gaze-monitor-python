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

The DNN model used in this application is an Intel® optimized model that is part of the Intel® Distribution of OpenVINO™ toolkit. You can find it here:

  * /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001
  * /opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001

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

	```uname -a```
  
* Intel® Distribution of OpenVINO™ toolkit 2018 R5 release

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux for more information about how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

You will need the OpenCL™ Runtime package if you plan to run inference on the GPU. It is not mandatory for CPU inference.

### Install Python* dependencies

```sudo apt install python3-pip```

```sudo apt-get install mosquitto mosquitto-clients```

```pip3 install jupyter ```

```pip3 install numpy```

```pip3 install paho-mqtt```


### Sample Video

You can download sample video by running following commands in `shopper-gaze-monitor-python` directory on the terminal.	

```
mkdir resources
cd resources
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4 
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4 
wget https://github.com/intel-iot-devkit/sample-videos/blob/master/head-pose-face-detection-female.mp4 	
cd .. 
```

## Setting the Build Environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

 ```
 source /opt/intel/computer_vision_sdk/bin/setupvars.sh -pyver 3.5
 ```

### Follow the steps to run the code on Jupyter*:

1. Go to the `shopper-gaze-monitor-python` directory and open the Jupyter notebook.

	```jupyter notebook```
	
      ![images](./images/jupy1.png)
      
2. Click on **New** button on the right side of the Jupyter window.

3. Click on **Python 3** option from the drop down list.

4. In the first cell type **import os** and press **Shift+Enter** from the keyboard.

5. Export the below environment variables in second cell of Jupyter and press **Shift+Enter**.

   %env INPUT_FILE = resources/face-demographics-walking-and-pause.mp4<br>
   %env CPU_EXTENSION = /opt/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so<br>
   %env MODEL=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml<br> 
   %env POSEMODEL=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml<br>

6. User can set threshold for the detection (CONFIDENCE) and target device to infer on (DEVICE). Export these environment variables as given below if required else skip this step. If user skips this step, these values are set to default values.

   %env CONFIDENCE = 0.5<br>
   %env DEVICE = CPU<br> 
      
7. Copy the code from **main_jupyter.py** and paste it in the next cell and press **Shift+Enter**.

8. Alternatively, code can be run in the following way.

   i. Click on the **main_jupyter.ipynb** file in the Jupyter notebook window.

   ii. Click on the **Kernel** menu and then select **Restart & Run All** from the drop down list.

   iii. Click on Restart and Run All Cells.
   
![images](./images/jupy2.png)

**NOTE:**

1. To run the application on **GPU**:
     * Change the **%env DEVICE = CPU** to **%env DEVICE = GPU**.
     * With the floating point precision 16 (FP16), change the path of the model in the environment variable as given below:<br>
      **%env MODEL=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml**<br> 
   **%env POSEMODEL=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml**<br>
     * **CPU_EXTENSION** environment variable is not required.
   
2. To run the application on **Intel® Neural Compute Stick**: 
      * Change the **%env DEVICE = CPU** to **%env DEVICE = MYRIAD**.  
      * The Intel® Neural Compute Stick can only run FP16 models. Hence change the environment variable for the model as shown below. <br>
      **%env MODEL=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml**<br> 
   **%env POSEMODEL=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml**<br>
      * **CPU_EXTENSION** environment variable is not required.
3. To run the application using **camera stream**, change the **%env INPUT_FILE = resources/face-demographics-walking-and-pause.mp4** to  **%env INPUT_FILE = cam**.<br>

## Machine to machine messaging with MQTT

If you wish to use a MQTT server to publish data, you should set the following environment variables on the terminal before running the program.
 
   ```export MQTT_SERVER=localhost:1883```
 
   ```export MQTT_CLIENT_ID=cvservice```

Change the MQTT_SERVER to a value that matches the MQTT server you are connecting to.

You should change the MQTT_CLIENT_ID to a unique value for each monitoring station, so you can track the data for individual locations. For example:

   ```export MQTT_CLIENT_ID=zone1337```

If you want to monitor the MQTT messages sent to your local server, and you have the mosquitto client utilities installed, you can run the following command in new terminal while executing the code:

   ```mosquitto_sub -h localhost -t shopper_gaze_monitor```
