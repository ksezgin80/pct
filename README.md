# Pedestrian Counting Tool
A framework to count pedestrians (or objects) in a video

Introduction
--------------------
The Pedestrian Counting Tool (PCT) has been developed in Python as a Jupyter Notebook. The notebook was organized in a step-by-step manner with proper titles and comments to guide the user. This documents serves as a how-to-use guide to explain the folder structure and how to use PCT.

Folder Structure
--------------------
The PCT folder has the following components:  
  * requirements: list of required python packages for PCT to function properly  
  * pct_main.ipynb: PCT Jupyter Notebook  
  * sort.py: library used for object tracking (SORT algorithm)  
  * utils.py: helper functions to support PCT   
  * input (folder): the sample input videos (AVG-TownCentre, PETS09-S2L2)  
  * output (folder): processed frames and animations are created in this folder  

How does PCT work?
--------------------
The tool functions as a regular Jupyter Notebook. Here is the outline of the tool:

1. The required libraries are loaded.  
2. The project file/folder settings are input by user.  
3. The input video is converted into frames with the option of resizing the frames, if required.   
4. The cvlib library is used for detecting objects in each frame. The cv.detect_common_objects() function is used to detect common objects and filtered the detected objects for ‘person’ class. These ‘person’ detections are stored in a text file.  
5. For object tracking between frames, SORT algorithm is used. NOTE that the tracker cell needs to be run only ONCE. Otherwise, the tracker would start assigning id values starting from the last largest assignment. These tracking ids along with object bounding box coordinates are stored in a text file as well.  
6. Before starting the counting process, the user needs to input a counter line. The user clicks two points in the first frame of the video. The tool draws a line and the ‘set_counter_line’ function returns the coordinates and the coefficients of the line passing through user-clicked points.  
7. Using the counter line and tracking info, the counting routine counts each person based on which region they move into and label regions as "A" and "B".  
   
How to use?
--------------------
The user needs to enter the "file/folder settings" to specify:
  * the input video path  
  * the output folder path  
  * "frame_folder": frames from the original video  
  * "detection_folder": frames after object detection  
  * "tracking_folder": frames after object tracking  
  * "counting_folder": frames after object counting  
  * "detection_file": filename to store object detection info  
  * "tracking_file": filename to store object tracking info  
  * "counting_file": filename to store object counting info  
    
The user needs to run all the cells until "Step 3. Count Objects - set counter line".

The user need to enter the counter line:
  * the user will see the first frame of the video
  * the user will click two points which will be used to create a line for counting
  * the user will stop the interaction by clicking the icon on the top right corner
  * the user will run the next cell

The user needs to run until the end of the notebook.

NOTE: The first time you run object detection code, it will download a large file for YOYOv3 model. It will do it once and you don't need to download it again. YOLOv3 is actually a heavy model to run on CPU. If you are working with real time webcam / video feed and doesn't have GPU, try using tiny yolo which is a smaller version of the original YOLO model. It's significantly fast but less accurate.

```python
bbox, label, conf = cv.detect_common_objects(im, model='yolov3-tiny')
```

Output
--------------------
The code generates the following output:
  * 1_frames    : frames from the original video
  * 2_detections: frames after object detection  + detections.txt
  * 3_trackings": frames after object tracking   + trackings.txt
  * 4_counting  : frames after object counting   + counts.txt
  * tracking_video.avi
  * counting_video.avi
    
