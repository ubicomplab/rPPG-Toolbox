@echo off

@REM echo ------------------------- UNSUPERVISED -------------------------
@REM echo:
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_UNSUPERVISED_garage_still.yaml"
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_UNSUPERVISED_garage_small_motion.yaml"
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_UNSUPERVISED_driving_still.yaml"
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_UNSUPERVISED_driving_small_motion.yaml"

echo ------------------------- DEEPPHYS -------------------------
echo:
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_DEEPPHYS_driving_small_motion.yaml"

echo ------------------------- TSCAN ------------------------
echo:
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_TSCAN_driving_small_motion.yaml"

echo ------------------------- EFFICIENTPHYS -------------------------
echo:
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_EFFICIENTPHYS_driving_small_motion.yaml"

echo ------------------------- PHYSNET -------------------------
echo:
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_garage_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_garage_small_motion.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_driving_still.yaml"
python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_driving_small_motion.yaml"

@REM echo PHYSFORMER
@REM echo:
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_garage_still.yaml"
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_garage_small_motion.yaml"
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_driving_still.yaml"
@REM python main.py --config_file "C:\\Users\\kouts\\Desktop\\Thesis\\rPPG-Toolbox\\configs\\infer_configs\\MR-NIRP_PHYSNET_driving_small_motion.yaml"

echo ------------------------- COMPLETE -------------------------
