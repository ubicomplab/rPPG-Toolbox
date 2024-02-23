@echo off
echo DEEPPHYS
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_driving_small_motion.yaml"

echo TSCAN
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_driving_small_motion.yaml"

echo EFFICIENTPHYS
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_driving_small_motion.yaml"

echo PHYSNET
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_driving_small_motion.yaml"

echo ------------------------- COMPLETE -------------------------