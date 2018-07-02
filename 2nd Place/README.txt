* Software Requirements

Main requirements: Python 3.4+, keras 2.0.8, theano 0.8.2, ffmpy (for video conversion), librosa (for audio processing), imagehash (extract image hashes)
Other requirements: numpy 1.13.1+, pandas 0.20.3+, opencv-python 3.1.0+, scipy 0.19.1+, sklearn 0.18.1+, 

** Notes:
- Code is written for Theano backend. Usage with Tensorflow should be fine too, but it wasn't checked.
- Code was developed in Microsoft Windows 10, but should work fine in Linux as well.
- I included WinPython (it system independent) which I uses for development to ensure you have all the needed modules. See folder WinPython-64bit-3.4.3.7Qt5build7.

* Hardware requirements

All batch sizes for Neural nets are tuned to be used on NVIDIA GTX 1080 Ti 11 GB card. To use code with other GPUs with less memory - decrease batch size accordingly.

* How to run:

All r*.py files must be run one by one. All intermediate folders will be created automatically.

python r10_preprocess_data.py
python r11_get_metadata_table.py
python r12_extract_image_hashes.py
python r12_get_inception_v3_predictions.py
python r12_get_resnet_predictions.py
python r12_get_VGG16_predictions.py
python r21_create_keras_3D_models.py
python r21_create_keras_audio_models_v1.py
python r21_create_keras_models_based_on_inception_v1.py
python r21_create_keras_models_based_on_inception_v2.py
python r21_create_keras_models_based_on_resnet50.py
python r21_create_keras_models_based_on_VGG16.py
python r21_create_keras_models_based_on_imagehashes.py
python r31_run_validation_3D_models.py
python r31_run_validation_audio_models.py
python r31_run_validation_based_on_imagehashes.py
python r31_run_validation_based_on_inception_v1.py
python r31_run_validation_based_on_inception_v2.py
python r31_run_validation_based_on_inception_v3.py
python r31_run_validation_based_on_resnet50.py
python r31_run_validation_based_on_VGG16.py
python r40_process_test_3D_models.py
python r40_process_test_audio.py
python r40_process_test_based_on_imagehashes.py
python r40_process_test_based_on_inception_v1.py
python r40_process_test_based_on_inception_v2.py
python r40_process_test_based_on_inception_v3.py
python r40_process_test_based_on_resnet50.py
python r40_process_test_based_on_VGG16.py
python r48_create_neighbor_features.py
python r50_second_level_model_xgboost.py
python r50_second_level_model_lightgbm.py
python r50_second_level_model_keras.py
python r60_ensemble_submissions.py

Optional:
python r70_create_videos.py

* Notes about a code

1) Files with same r*.py index can be run in parallel.
2) Training of neural networks can be done in parallel as well. There are 8 networks in total. And I used 5KFold split, so 5 models for each net. In extreme you can use 40 GPUs to train in parallel with x40 speedup. See r21_* files.
3) The same can be done with inference process: see r31_* files
4) It's probably only 2 networks needed for good accuracy Inception v3 and VGG16: r21_create_keras_models_based_on_inception_v2.py and r21_create_keras_models_based_on_VGG16.py.
5) In the header of each code file you can find some notes about its functionality

* Description of folders

-- DRIVENDATA_CHIMPS_SUBMIT - folder with solution code
-- models - weights for Neural nets which I obtained on my local computer. You can use them to skip training of neural nets from scratch.
-- WinPython-64bit-3.4.3.7Qt5build7 - python interpreter with all needed modules to run the code on Windows system. It system independent. It's the copy of my local python interpreter which was used for development of solution.
-- README.txt - this file
