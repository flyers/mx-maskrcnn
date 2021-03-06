export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0

TRAIN_DIR=model/res50-fpn/blender_v1/alternate/
DATASET=Blender
SET=train_v1
mkdir -p ${TRAIN_DIR}

# Train
python train_alternate_mask_fpn.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/resnet-50 \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu 0 |& tee -a ${TRAIN_DIR}/train.log

