# waspAS1ObjDet

## MobileNet transfer learning
1. Setting this up is trickier for two reasons
	- The TF object detection API is purely in TF and we need to see how to put a Keras face on it (and if that's necessary)
	- I have a problem which restricts me to use TF 1.4, so I'm writing instructions for 1.4 compatibility only
2. Download/clone & checkout this specific realease of the TF object detection API - https://github.com/tensorflow/models/tree/1f34fcafc1454e0d31ab4a6cc022102a54ac0f5b/research/object_detection
	- The commit_sha for this checkout is in the URL
	- Remember to install all the dependencies too
3. Train the model using the script transfer_learning.py
	- Note: To train you need the necessary config files and tfrecords so please ask us

## Pen detection in a live stream
1. Download the pretrained model from the link in the demo slide deck
2. Run live detection using predict_live.py
	- Note: To run this, you may need to install TF object detection dependencies

## Creating a custom dataset