[//]: # (Image References)

[imageRobot]: ./Pictures/imageRobot.png
[imageRawCameraOutput]: ./Pictures/imageRawCameraOutput.PNG
[imageStatisticalOutlierFilter]: ./Pictures/imageStatisticalOutlierFilter.PNG
[imageVoxelAndPassThroughOutput]: ./Pictures/imageVoxelAndPassThroughOutput.PNG
[imageObjectsOutput]: ./Pictures/imageObjectsOutput.PNG
[imageTableOutput]: ./Pictures/imageTableOutput.PNG
[imageClusteringOutput]: ./Pictures/imageClusteringOutput.PNG
[CountConfMatrix1]: ./Pictures/CountConfMatrix1.png
[CountConfMatrix2]: ./Pictures/CountConfMatrix2.png
[CountConfMatrix3]: ./Pictures/CountConfMatrix3.png
[NormConfMatrix1]: ./Pictures/NormConfMatrix1.png
[NormConfMatrix2]: ./Pictures/NormConfMatrix2.png
[NormConfMatrix3]: ./Pictures/NormConfMatrix3.png
[imageFinalOutputScene3]: ./Pictures/imageFinalOutputScene3.PNG
[imagePublishersSubscribers]: ./Pictures/imagePublishersSubscribers.PNG
[imagePCLCallback]: ./Pictures/imagePCLCallback.PNG

## Project: 3D Perception
### The why? 
This project is meant is meant to give an in-depth understanding of how a basic perception pipeline can be built for a standard industrial robot. 

### The what?
The robot involved in this project is the PR2 robot with two arms, which is a commonly used robot for pick and place operations. The robot has an RGBD sensor. This camera senses the color and depth information of objects in front of it (placed on a table). In this project we implement a series of image processing steps to help the PR2 robot recognize the objects placed on the table. The result of the perception step can be used in pick and place operations where the robot is responsible for picking objects from the table top and place it in an assigned bin. 

### The how?
The perception pipeline has a few major steps which will be discussed in detail below. They are summarized here:
1. Remove noise from the collected image using an outlier filter.
2. Downsample the image using Voxel grid filering and remove unwanted regions using a pass through filter.
3. Separate the table and the objects (RANSAC filter)
4. Cluster object points together (DBSCAN clustering)
5. Collect sample images and train an SVM to identify the clustered objects.
6. Output the coordinates of the identified objects and the coordinates of the bins where they must be placed to the specified format.
---

### Project set up
On launching the project in RViz, we see the PR2 robot facing a table with a bunch of objects on it. We will use this setup to test our perception pipeline. Here's a screen shot of what the simulator looks like:
![alt text][imageRobot]

A given launch file (pick_place_project.launch) is used to visualize the project setup in Rviz. The perception pipeline is implemented in a python file, which will be called during run time (project_template.py). In addition to containing the perception pipeline, the python file also contains the following subscribers and publishers (read/write data from/to the simulator):
![alt text][imagePublishersSubscribers]

Data obtained from the simulator is converted from ROS to Point Cloud using a helper function. Similarly, data being published is converted from Point Cloud to ROS format.

The rest of the write-up discusses the perception pipeline implemented in a functino called the "pcl_callback":
![alt text][imagePCLCallback]

### Overview of the perception pipeline 
The following functions are developed and applied on the image sensed by the robot. By the end of this phase, the PR2 robot will be able to recognize objects in front of it. It will also be able to compute the object centroids, which are needed for the pick and place operation. Before we get started, here's a screen shot of the raw data as seen by the camera:
![alt text][imageRawCameraOutput]

#### 1. Statistical Outlier Filter
In this step, we apply a statistics based filter to remove noise pixels. This type of filter measures the mean distance between each pixel and its neighbors and rejects the pixels that lie outside a specified standard deviation. For this application, we consider 10 neighbors for each pixel and reject the points that lie outside 0.1 times the standard deviation. The results after applying the filter are shown below:
![alt text][imageStatisticalOutlierFilter]

#### 2. Voxel grid and Pass through filtering
The image output from the camera has a high number of pixels, which may increase the processing time. For applications where the scene is not too crowded, we may not need information from all the pixels to identify the objects. Thus, we down sample the pixels, by considering only those pixels that lie on the edge of a cubic grid of a specified size (voxel). The size of the grid we consider is 0.01. As another cost saving measure, we also crop the image that is processed to only the regions of interest (the table top and the objects). We do this by applying a pass through filter. The pass through filter is set up to process the scene from (-0.5 to 0.5) in the y-axis and (0.6 to 1.1) in the z-axis. Here's the result of the image after applying the voxel and pass through filters:

![alt text][imageVoxelAndPassThroughOutput]

#### 3. RANSAC filtering
The next step in the perception pipeline is to separate the table from the objects of interest. This step is done so that the objects can then be further analyzed, and the table can be removed from further processing. RANSAC stands for Random Sample Consensus. The idea behind RANSAC is to see how well each pixel fits to a given model. Points that fit the model are considered inliers, and the rest are considered outliers. The shape of the table is a plane. Hence, we do a RANSAC plane fitting, and consider the inliers to be the table and the outliers to be the area that contains the objects. The maximum allowed deviation from the model is set to 0.015. Here's the output of the table and the objects after applying the RANSAC plan filter:
![alt text][imageObjectsOutput]

![alt text][imageTableOutput]

#### 4. Clustering using DBSCAN
We now have the pixels corresponding to the area that contain the objects of interest. The next step is to group the points belonging to each object. This process is called "clustering" and the algorithm used is called DBSCAN. The objective of the DBSCAN algorithm is to group together points based on their distance to neighboring points (or density). We set the minimum number of points needed to form a cluster to 100 and the maximum to 3000. The maximum distance between pixels in a cluster is set to 0.05. The algorithm outputs the indices of the points that belong to each cluster. All points belonging to a cluster are colored together for visualization purposes. The result of the clustering is shown below:
![alt text][imageClusteringOutput]

#### 5. Object recognition
At this point, we have successfully identified the pixels that belong to each object. The next step is to identify what that object is. In order to recognize the object, we will use sample images of each object to train an SVM. We will then use the trained model to predict the object type.

##### Step 1: Collect the training data
We run a training simulation to collect sample images of each object as seen by the camera. Each sample image is spawned at a different orientation in order to provide the SVM with as much information as possible. A set of training features are extracted from each image. These features are the inputs to the SVM and they provide relevant information about the shape and color of each object. The features include a color histogram of the HSV channel and a histogram of the normal vectors of each object surface. The HSV channel is used to obtain the color histogram, instead of the RGB channel. This is because the HSV channel information is typically insensitive to lighting conditions, thus resulting in better information about the object. In total, 50 images were used to train each object in scene 1. 125 images per object were used in scene 2. 175 images per object were used in scene 3. 

##### Step 2: Train the SVM
The collected features are then used to train the SVM to predict the objects. We use a linear kernel for this project. The pipeline is tested on three setups, each one progressively complicated than the previous. The confusion matrices for all the scenarios are given below. They represent the output of the SVM's validation:

##### Scene1
![alt text][CountConfMatrix1]
![alt text][NormConfMatrix1]

##### Scene2
![alt text][CountConfMatrix2]
![alt text][NormConfMatrix2]

##### Scene3
![alt text][CountConfMatrix3]
![alt text][NormConfMatrix3]

The accuracy of the trained SVM is as follows:
Scene 1: 93.3%
Scene 2: 94.4%
Scene 3: 95.5%

##### Step 3: Predict using the SVM
We use the trained SVM model to predict the object type in the perception pipeline. Labels are attached to each predicted object and are displayed in the image for visualization. Here's the result of prediction for scene 3, which is the toughest scene we tested:

![alt text][imageFinalOutputScene3]

In this image, 6/8 objects are classified right. The glue is misclassified as soap, and the book is not recognized. Please see the section about improvements for further suggestions on how to improve the pipeline.

#### 6. Output the centroid data
We have now completed the perception pipeline that can be used for a succesful pick and place project. The final step in this project is to output each object's type and centroid to a "YAML" format file. This YAML file is then used by the pick and place service to complete the pick and place operation. The output YAML files for the three test scenes can be found in the project repository. The centroid for each object is calculated as the mean position of all the pixels that belong to that object.

#### Potential problems and further suggestions
With this algorithm implementation, PR2 robot does a decent job of identifying objects placed in front of it. The accuracy achieved was:

Scene 1: 100%

Scene 2: 80%

Scene 3: 75%

The following are some suggestions to improve this pipeline:

1. The pipeline seems to have a tougher time with crowded scenes. Better clustering algorithms could be explored to alleviate this.
2. The number of features used to train the SVM plays a major part in the SVM's prediction accuracy. A histogram with 32 bins was used to train the SVM for scene 1 and 2. A histogram with 128 bins was used for scene 3. While more objects were recognized in scene 3, it can be seen that the glue was mis-recognized as a soap. This indicates that the SVM is trending towards over fitting, and hence the number of histogram bins could be reduced. Playing around with the number of histogram bins for the color and noraml histograms would lead to better SVM predictions.
3. The SVM was fit to a linear kernel. A more complex kernel, such as the RBF (Radial Basis Function) could lead to better predictions.
4. Finally, the project can be improved by actually passing the required output to the pick and place service, there by completing the pick and place operation. There are some additional steps involved in this, such as making sure that the robot does not think of the objects as collision points. 

#### Source code for the project

1. [Launch file for the project](./pr2_robot/launch/pick_place_project.launch)
2. [The perception pipeline](./pr2_robot/scripts/project_template.py)
3. [Launch file for the SVM training](./sensor_stick/launch/training.launch)
4. [The script used to capture training features for the SVM](./sensor_stick/scripts/capture_features.py)
5. [The script used to train the SVM](./sensor_stick/scripts/train_svm.py)
6. [The output YAML for scene 1](./output_1.yaml)
7. [The output YAML for scene 2](./output_2.yaml)
8. [The output YAML for scene 3](./output_3.yaml)
