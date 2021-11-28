# Readme

## install requirements

cd Toolbox

conda create -n your_env python=3.6

pip install -r requirements.txt

## run codes

**python main.py --model physnet --data_dir [path for UBFC dataset] --device [e.g:0 for using cuda:0]**

For exemple, if your UBFC data are [/path/UBFC/subject1/video+signal,/path/UBFC/subject2/video+signal,...] and you are using cuda:1

The command should be:

**python main.py --model physnet --data_dir  /path  --device 1**

