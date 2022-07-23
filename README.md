# Real-Time-Face-Mask-Detection

The objective of this project is to detect the presence of a face mask on human faces on live videos. First collected images using webcam and OpenCV. I have used LabelImg to label the images for training the model. Then trained the model using TensorFlow object detection API more specifically used transfer learning and leveraged a pre-trained model “MobileNet SSD “.And then used OpenCV to make real-time face detection. 

## Workflow

1.&nbsp;<b><a href ="https://github.com/nikhil-047/Real-Time-Face-Mask-Detection/blob/master/Create%20Image%20to%20Train.ipynb" >Creating Dataset </a> : </b>
I have created the image dataset by myself using opencv . The dataset includes images in which I had some images with and without wearing the mask.Dataset can also be downloaded from kaggle.

Then for image annotation I have used a easy-to-use image annotation tool LabelImg to label object bounding boxes in images.The annotations are saved as XML files in PASCAL VOC format.This format is used by ImageNet for object detection.

Can download the LabelImg tool and follow the installation instructions provided here https://github.com/heartexlabs/labelImg

&emsp;&emsp;<img src="https://user-images.githubusercontent.com/43903557/180600144-bf1a529f-c16d-4aa8-abeb-942764a0b86e.png" width=40% height=40% > &emsp;&emsp;&emsp;&emsp;<img src="https://user-images.githubusercontent.com/43903557/180600137-dfc04e27-690b-4aea-a323-b1bc31b0652a.png" width=40% height=40%>
 
 
&emsp;&emsp;Image 1. Labelling face image with No Mask &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Image 2. Labelling face image with Mask


2.&nbsp;<b><a href ="https://github.com/nikhil-047/Real-Time-Face-Mask-Detection/blob/master/Real%20Time%20Face%20Mask%20Detection.ipynb" >Traning and Evaluating the Model </a> : </b>
Will download a pre trained model and applied transfer learning , have fine tuned the model as per my requirement to achieve the goal.The model was built with <a href ="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">MobileNet SSD v2</a> .

&emsp;&emsp;<img src="https://user-images.githubusercontent.com/43903557/180601586-d502cc5f-2257-4ecc-a073-a4ef6b032eaa.png" width=40% height=40% ></br>
&emsp;&emsp;Image 3. Evaluating the performance of model

2.&nbsp;<b><a href ="https://github.com/nikhil-047/Real-Time-Face-Mask-Detection/blob/master/Real%20Time%20Face%20Mask%20Detection.ipynb" >Making detections in real time using model </a> : </b> Used OpenCv for capturing and displaying the results predicted from the model in real time.
</br>

You can refer the following link for more details on Tensorflow's object detection API https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/




 
