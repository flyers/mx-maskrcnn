export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0

TRAIN_DIR=model/res50-fpn/blender_v1/alternate/
DATASET=Blender
TEST_SET=test_21

# Test
python eval_maskrcnn.py \
    --network resnet_fpn \
    --has_rpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${TRAIN_DIR}final \
    --result_path data/blender/results/pred/ \
    --epoch 0 \
    --gpu 0

