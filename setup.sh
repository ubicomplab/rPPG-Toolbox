conda remove --name rppg-toolbox --all -y
conda create -n rppg-toolbox python=3.8 -y
conda activate rppg-toolbox
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y
pip install -r requirements.txt
