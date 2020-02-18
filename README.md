# waspAS1ObjDet

## For YOLOv3
1. Download pre-trained YOLOv3 weights on MSCOCO - https://pjreddie.com/media/files/yolov3.weights
2. Run yolo_convert.py to build and save a Keras YOLOv3 model
3. Run yolo_predict.py to detect zebras from test/zebra.jpg

## For SSD7
1. Clone the ssd_keras repo https://github.com/pierluigiferrari/ssd_keras
2. Download the specially prepared dataset (by pierluigiferrari) https://drive.google.com/open?id=1tfBFavijh4UTG4cGqIKwhcklLXUDuY0D and unzip into datasets/udacity_driving_datasets
3. Download the ssd7 model which I trained using the link that I sent you
4. Edit the ssd_keras path in sys.path.append in ssd7_inference.py
5. Run ssd7_predict.py, which will detect objects in a random image from the validation set
6. The prediction result can be seen in output.png