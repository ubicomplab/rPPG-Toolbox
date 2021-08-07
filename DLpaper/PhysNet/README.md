# PhysNet
Main code of **BMVC2019 paper 'Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks'** 

How to train it?

    #1. Inference the model
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
    rPPG, x_visual, x_visual3232, x_visual1616 = model(inputs)

    #2. Normalized the Predicted rPPG signal and GroundTruth BVP signal
    rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize
    BVP_label = (BVP_label-torch.mean(BVP_label)) /torch.std(BVP_label)	 	# normalize

    #3. Calculate the loss
    loss_ecg = Neg_Pearson(rPPG, BVP_label)

It is just for **research purpose**, and commercial use is not allowed.

If you use the PhysNet please cite: 

>@inproceedings{yu2019remote,  
>&nbsp;&nbsp;&nbsp;&nbsp;title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},      
>&nbsp;&nbsp;&nbsp;&nbsp;author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},  
>&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {Proc. BMVC},  
>&nbsp;&nbsp;&nbsp;&nbsp;year = {2019}  
>}  


