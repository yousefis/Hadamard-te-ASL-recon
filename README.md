# Dynamic Angiography and Perfusion Reconstruction from Hadamard-te Arterial Spin Labeling of rank 8
You are more than welcome to use this code. Please include the below paper in your work. 
    
    @inproceedings{yousefi2019fast,
    title={Fast Dynamic Perfusion and Angiography Reconstruction using an end-to-end 3D Convolutional Neural Network},
    author={Yousefi, Sahar and Hirschler, Lydiane and van der Plas, Merlijn and Elmahdy, Mohamed S and Sokooti, Hessam and Van Osch, Matthias and Staring, Marius},
    booktitle={International Workshop on Machine Learning for Medical Image Reconstruction},
    pages={25--35},
    year={2019},
    organization={Springer}
    }


![Alt Text](figures/AnyConv.com__cnn-1.png)
Figure 1- Proposed network, a multi-stage DenseUnet. Inputs: an interleaved crushed and non-crushed Hadamard-te arterial spin labeling (ASL) of rank 8. Output: dynamic angiography and perdusion scans at 8 time-point.

![Alt Text](figures/AnyConv.com__data_generator-1.png)
Figure 2- Proposed data generator.

![Alt Text](figures/angiography_res.bmp)
Figure 3- Results of reconstructed angiography scans for one subject.

![Alt Text](figures/perfusion_res.bmp)
Figure 4- Results of reconstructed perfusion scans for one subject..

