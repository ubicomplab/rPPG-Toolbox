@echo off
echo UNSUPERVISED
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_UNSUPERVISED_driving_still.yaml"
echo TSCAN/DEEPPHYS
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_driving_still.yaml"
echo EFFICIENTPHYS
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_driving_still.yaml"
echo PHYSNET
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_driving_still.yaml"
echo -------------------------COMPLETED-------------------------