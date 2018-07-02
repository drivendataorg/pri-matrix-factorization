#!/usr/bin/env bash

export PATH="$PATH:/usr/local/cuda-8.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"

# it's possible to train each fold/model on the separate GPU/system,
# it's only necessary to run train/generate_prediction steps in sequence for particular model/fold
# export CUDA_VISIBLE_DEVICES=0
#CUDA_VISIBLE_DEVICES=0 python3.6 single_frame_cnn.py train --model resnet50_avg --fold 1
# CUDA_VISIBLE_DEVICES=1 python3.6 single_frame_cnn.py train --model resnet50_avg --fold 2
# CUDA_VISIBLE_DEVICES=2 python3.6 single_frame_cnn.py train --model resnet50_avg --fold 3
# CUDA_VISIBLE_DEVICES=3 python3.6 single_frame_cnn.py train --model resnet50_avg --fold 4

#CUDA_VISIBLE_DEVICES=0 python3.6 single_frame_cnn.py generate_prediction --model resnet50_avg --fold 1 --weights ../output/checkpoints/resnet50_avg_fold_1/checkpoint-012*
#CUDA_VISIBLE_DEVICES=1 python3.6 single_frame_cnn.py generate_prediction --model resnet50_avg --fold 2 --weights ../output/checkpoints/resnet50_avg_fold_2/checkpoint-012*
CUDA_VISIBLE_DEVICES=2 python3.6 single_frame_cnn.py generate_prediction --model resnet50_avg --fold 3 --weights ../output/checkpoints/resnet50_avg_fold_3/checkpoint-012*
#CUDA_VISIBLE_DEVICES=3 python3.6 single_frame_cnn.py generate_prediction --model resnet50_avg --fold 4 --weights ../output/checkpoints/resnet50_avg_fold_4/checkpoint-012*

#python3.6 single_frame_cnn.py find_non_blank_frames --fold 1 --model resnet50_avg
#python3.6 single_frame_cnn.py find_non_blank_frames --fold 2 --model resnet50_avg
#python3.6 single_frame_cnn.py find_non_blank_frames --fold 3 --model resnet50_avg
#python3.6 single_frame_cnn.py find_non_blank_frames --fold 4 --model resnet50_avg

# When run in parallel, finish all find_non_blank_frames before training next models

#python3.6 single_frame_cnn.py train --model xception_avg --fold 1 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model xception_avg --fold 2 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model xception_avg --fold 3 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model xception_avg --fold 4 --use_non_blank_frames --nb_epoch 14

#python3.6 single_frame_cnn.py generate_prediction --model xception_avg --fold 1 --weights ../output/checkpoints/xception_avg_fold_1/checkpoint-007*
#python3.6 single_frame_cnn.py generate_prediction --model xception_avg --fold 2 --weights ../output/checkpoints/xception_avg_fold_2/checkpoint-007*
#python3.6 single_frame_cnn.py generate_prediction --model xception_avg --fold 3 --weights ../output/checkpoints/xception_avg_fold_3/checkpoint-007*
#python3.6 single_frame_cnn.py generate_prediction --model xception_avg --fold 4 --weights ../output/checkpoints/xception_avg_fold_4/checkpoint-007*

#python3.6 single_frame_cnn.py generate_prediction --model xception_avg_ch10 --fold 1 --weights ../output/checkpoints/xception_avg_fold_1/checkpoint-012*
#python3.6 single_frame_cnn.py generate_prediction --model xception_avg_ch10 --fold 2 --weights ../output/checkpoints/xception_avg_fold_2/checkpoint-012*
#python3.6 single_frame_cnn.py generate_prediction --model xception_avg_ch10 --fold 3 --weights ../output/checkpoints/xception_avg_fold_3/checkpoint-012*
#python3.6 single_frame_cnn.py generate_prediction --model xception_avg_ch10 --fold 4 --weights ../output/checkpoints/xception_avg_fold_4/checkpoint-012*

#python3.6 single_frame_cnn.py train --model inception_v3 --fold 1 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model inception_v3 --fold 2 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model inception_v3 --fold 3 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model inception_v3 --fold 4 --use_non_blank_frames --nb_epoch 14

#python3.6 single_frame_cnn.py generate_prediction --model inception_v3 --fold 1 --weights ../output/checkpoints/inception_v3_fold_1/checkpoint-012*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v3 --fold 2 --weights ../output/checkpoints/inception_v3_fold_2/checkpoint-012*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v3 --fold 3 --weights ../output/checkpoints/inception_v3_fold_3/checkpoint-012*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v3 --fold 4 --weights ../output/checkpoints/inception_v3_fold_4/checkpoint-012*

#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 1 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 2 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 3 --use_non_blank_frames --nb_epoch 14
#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 4 --use_non_blank_frames --nb_epoch 14

#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1/checkpoint-006*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2/checkpoint-006*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3/checkpoint-006*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4/checkpoint-006*

#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_ch10 --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1/checkpoint-013*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_ch10 --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2/checkpoint-013*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_ch10 --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3/checkpoint-013*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_ch10 --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4/checkpoint-013*

#python3.6 single_frame_cnn.py train --model resnet152 --fold 1 --use_non_blank_frames --nb_epoch 16
#python3.6 single_frame_cnn.py train --model resnet152 --fold 2 --use_non_blank_frames --nb_epoch 16
#python3.6 single_frame_cnn.py train --model resnet152 --fold 3 --use_non_blank_frames --nb_epoch 16
#python3.6 single_frame_cnn.py train --model resnet152 --fold 4 --use_non_blank_frames --nb_epoch 16

#python3.6 single_frame_cnn.py generate_prediction --model resnet152 --fold 1 --weights ../output/checkpoints/resnet152_fold_1/checkpoint-015*
#python3.6 single_frame_cnn.py generate_prediction --model resnet152 --fold 2 --weights ../output/checkpoints/resnet152_fold_2/checkpoint-015*
#python3.6 single_frame_cnn.py generate_prediction --model resnet152 --fold 3 --weights ../output/checkpoints/resnet152_fold_3/checkpoint-015*
#python3.6 single_frame_cnn.py generate_prediction --model resnet152 --fold 4 --weights ../output/checkpoints/resnet152_fold_4/checkpoint-015*

#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 1 --use_non_blank_frames --use_extra_clips --nb_epoch 16
#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 2 --use_non_blank_frames --use_extra_clips --nb_epoch 16
#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 3 --use_non_blank_frames --use_extra_clips --nb_epoch 16
#python3.6 single_frame_cnn.py train --model inception_v2_resnet --fold 4 --use_non_blank_frames --use_extra_clips --nb_epoch 16

#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_extra --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1_extra/checkpoint-014*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_extra --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2_extra/checkpoint-014*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_extra --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3_extra/checkpoint-014*
#python3.6 single_frame_cnn.py generate_prediction --model inception_v2_resnet_extra --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4_extra/checkpoint-014*

# finish generate_prediction before running save_all_combined_train_results

#python3.6 single_frame_cnn.py save_all_combined_train_results

#python3.6 second_stage_nn.py train
#python3.6 second_stage.py train
