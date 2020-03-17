#in tf_models/research/object_detection run the following
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#for pedestrian detection
python -i train.py --train_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pedestrians/ --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/pedestrians/ssd_mobilenet_v1_pets.config
python -i eval.py --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/pedestrians/ssd_mobilenet_v1_pets.config --checkpoint_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pedestrians --eval_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/test/mobilenet_pedestrians
python -i export_inference_graph.py --input_type=image_tensor --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/pedestrians/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pedestrians/model.ckpt-2813 --output_directory=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pedestrians_ig

#for detecting pens
python -i train.py --train_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pens \
 --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/oid_pens/ssd_mobilenet_v1_pens.config


python -i train.py --train_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/merged_mobilenet_pens \
 --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/oid_pens/merged.config

 python -i train.py --train_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/concat_mobilenet_pens \
 --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/oid_pens/concat.config

python -i eval.py --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/oid_pens/ssd_mobilenet_v1_pens.config \
--checkpoint_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pens \
--eval_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/test/mobilenet_pens

python -i export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/oid_pens/ssd_mobilenet_v1_pens.config \
--trained_checkpoint_prefix=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pens/model.ckpt-20000 \
--output_directory=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/mobilenet_pens_ig

python -i export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/oid_pens/concat.config \
--trained_checkpoint_prefix=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/concat_mobilenet_pens/model.ckpt-20000 \
--output_directory=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/concat_mobilenet_pens_ig

#custom dataset
python -i train.py --train_dir=/home/px2/courses/wasp-as1/waspAS1ObjDet/models/custom_mobilenet_pens \
 --pipeline_config_path=/home/px2/courses/wasp-as1/waspAS1ObjDet/datasets/custom_pens/ssd_mobilenet_v1_pens.config