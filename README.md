# waspAS1ObjDet

## For YOLOv3
1. Download pre-trained YOLOv3 weights on MSCOCO - https://pjreddie.com/media/files/yolov3.weights
2. Run yolo_convert.py to build and save a Keras YOLOv3 model
3. Run yolo_predict.py to detect zebras from test/zebra.jpg

## For SSD7
1. Clone the ssd_keras repo https://github.com/pierluigiferrari/ssd_keras
2. Download the specially prepared dataset (by pierluigiferrari) https://drive.google.com/open?id=1tfBFavijh4UTG4cGqIKwhcklLXUDuY0D and unzip into datasets/udacity_driving_datasets
3. Download the ssd7 model trained from scratch using the link that was emailed
4. Edit the ssd_keras path in sys.path.append in ssd7_inference.py
5. Run ssd7_predict.py, which will detect objects in a random image from the validation set
6. The prediction result can be seen in output.png

## For MobileNet
1. Setting this up is trickier for two reasons
	- The TF object detection API is purely in TF and we need to see how to put a Keras face on it (and if that's necessary)
	- I have a problem which restricts me to use TF 1.4, so I'm writing instructions for 1.4 compatibility only
2. Download/clone & checkout this specific realease of the TF object detection API - https://github.com/tensorflow/models/tree/1f34fcafc1454e0d31ab4a6cc022102a54ac0f5b/research/object_detection
	- The commit_sha for this checkout is in the URL
3. In mobilenet_predict.py, edit TF_OBDET_ROOT to point to where you have the TF object detection repo
4. Run mobilenet_predict.py to detect objects in two test images included in the TF object detection repo