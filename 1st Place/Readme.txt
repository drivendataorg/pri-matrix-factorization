This package covers prediction from trained models to reproduce winning submissions.

Requirements:

Linux (tested on Ubuntu 17.04 with python 3.6 installed),
python 3.6
ffmpeg, may be necessary to install dev packetas as well for building python packets:
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev libswscale-dev libavresample-dev
It's also necessary to install cuda8 for tensorflow 1.4

python dependencies are listed in requirements.txt


Prepare input files:

test video clips should be extracted to input/raw directory
decoded frames are stored to /opt/data_fast/pri_matrix/train_img
but this directory can be chaged in config.py and convert_all_img.sh
ssd drive is reecommended for this directory with approx 250-300GB free space


bash prepare_input_data.sh

cd src
bash train.sh
bash predict.sh


Notes:
when done sequentially, prediction would take a lot of time.
all the calls like
python3.6 single_frame_cnn.py generate_prediction_test 
in predict.sh are independent and can be run on different GPUs or different computers.

It's just necessary to collect generated output/prediction_test_frames before running the second stage scripts

