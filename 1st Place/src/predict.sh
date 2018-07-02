#!/usr/bin/env bash

set -e

# all generate_prediction_test files may be run in parallel

# checkpoint numbers must match train.sh script

python3.6 single_frame_cnn.py generate_prediction_test --model resnet50_avg --fold 1 --weights ../output/checkpoints/resnet50_avg_fold_1/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model resnet50_avg --fold 2 --weights ../output/checkpoints/resnet50_avg_fold_2/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model resnet50_avg --fold 3 --weights ../output/checkpoints/resnet50_avg_fold_3/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model resnet50_avg --fold 4 --weights ../output/checkpoints/resnet50_avg_fold_4/checkpoint-012*

python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 1 --weights ../output/checkpoints/xception_avg_fold_1/checkpoint-007*
python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 2 --weights ../output/checkpoints/xception_avg_fold_2/checkpoint-007*
python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 3 --weights ../output/checkpoints/xception_avg_fold_3/checkpoint-007*
python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 4 --weights ../output/checkpoints/xception_avg_fold_4/checkpoint-007*

python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 1 --weights ../output/checkpoints/xception_avg_fold_1/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 2 --weights ../output/checkpoints/xception_avg_fold_2/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 3 --weights ../output/checkpoints/xception_avg_fold_3/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 4 --weights ../output/checkpoints/xception_avg_fold_4/checkpoint-012*

python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 1 --weights ../output/checkpoints/inception_v3_fold_1/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 2 --weights ../output/checkpoints/inception_v3_fold_2/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 3 --weights ../output/checkpoints/inception_v3_fold_3/checkpoint-012*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 4 --weights ../output/checkpoints/inception_v3_fold_4/checkpoint-012*

python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1/checkpoint-006*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2/checkpoint-006*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3/checkpoint-006*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4/checkpoint-006*

python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1/checkpoint-013*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2/checkpoint-013*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3/checkpoint-013*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4/checkpoint-013*

python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 1 --weights ../output/checkpoints/resnet152_fold_1/checkpoint-015*
python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 2 --weights ../output/checkpoints/resnet152_fold_2/checkpoint-015*
python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 3 --weights ../output/checkpoints/resnet152_fold_3/checkpoint-015*
python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 4 --weights ../output/checkpoints/resnet152_fold_4/checkpoint-015*

python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1_extra/checkpoint-014*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2_extra/checkpoint-014*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3_extra/checkpoint-014*
python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4_extra/checkpoint-014*

python3.6 single_frame_cnn.py save_all_combined_test_results

python3.6 second_stage.py predict
python3.6 second_stage_nn.py predict
python3.6 second_stage_nn.py generate_submission

