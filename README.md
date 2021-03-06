# If this repository helps you in anyway, show your love :heart: by putting a :star: on this project 

# Dynamic Angiography and Perfusion Reconstruction from Hadamard-te Arterial Spin Labeling of rank 8
# 1- Introduction
In this work 4D Angiography and Perfusion at eight time-points are reconstructed from an interleaved half-sampled crushed and non-crushed Hadamard-te arterial spin labeling (ASL) of rank 8. The network uses DenseUnet structure and multi-stage loss function. Different loss functions have been applied for training including: perceptual loss (PL), mean squre error (MSE), Structural Similarity Index (SSIM) in a single and multi-stage fasions. Also, a framework for generating dynamic ASL scans based on the Hadamard ASL kinetic model has been proposed. 

The reconstruction process can be formulated as: 
<img src="https://latex.codecogs.com/svg.latex?\;M\left(\{I_i^{NC}\},%20\{I_i^{C}\}\right)_{i=1}^{H}%20=\{{P}(t),%20{A}(t)\}_{t=1}^{H-1}"/>,
in which <img src="https://latex.codecogs.com/svg.latex?\;M"/> is the decoding and subtraction function, <img src="https://latex.codecogs.com/svg.latex?\;I_i^{NC}"/> and <img src="https://latex.codecogs.com/svg.latex?\;I_i^{C}"/> are the acquired scans of the <img src="https://latex.codecogs.com/svg.latex?\;i^{th}"/> row of non-crushed and crushed Hadamard te-pCASL datasets, <img src="https://latex.codecogs.com/svg.latex?\;P"/> and <img src="https://latex.codecogs.com/svg.latex?\;{A}"/> denote perfusion and angiography scans respectively. 

<a href="https://github.com/yousefis/Hadamard_te_asl_signal">Here</a> you can find the Hadamard te-ASL signal generator.
# 2- Citation
    @inproceedings{yousefi2019fast,
    title={Fast Dynamic Perfusion and Angiography Reconstruction using an end-to-end 3D Convolutional Neural Network},
    author={Yousefi, Sahar and Hirschler, Lydiane and van der Plas, Merlijn and Elmahdy, Mohamed S and Sokooti, Hessam and Van Osch, Matthias and Staring, Marius},
    booktitle={International Workshop on Machine Learning for Medical Image Reconstruction},
    pages={25--35},
    year={2019},
    organization={Springer}
    }

# 3- Proposed network


<p>
    <img src="figures/AnyConv.com__cnn-1.png" alt>
    <em>Figure 1- Proposed network, a multi-stage DenseUnet. Inputs: an interleaved half-sampled crushed and non-crushed Hadamard-te arterial spin labeling (ASL) of rank 8. Output: dynamic angiography and perdusion scans at 8 time-point.</em>
</p>



# 4- Proposed data generator
<!---The kinetic model for generating signal of arteries: --->

<!---<img src="https://latex.codecogs.com/svg.latex?\;S_{artery}=\begin{cases} 0 & \text{if $t<\Delta t_b$}\\M_{0a}\cdot aCBV \cdot L_r(b)\times e^{\frac{-\Delta t_b}{T_{1b}}} &\text{if $\Delta t_{b} + \sum_{b^\prime=1}^{b-1}\tau_{b^\prime}\leq t<\Delta t_{b} + \sum_{b^\prime=1}^{b}\tau_{b^\prime}$} \\0 & \text{if $t\geq \Delta t_{b} + \sum_{b^\prime=1}^{N}\tau_{b^{\prime}}$}\\ \end{cases}"/>,--->

<!---and The kinetic model for generating signal of tissue:--->

<!---<img src="https://latex.codecogs.com/svg.latex?\;S_{tissue}=\begin{cases} 0 & \text{if $t<\Delta t_b$}\\\gamma\Gamma_{\beta=0}& \text{if $\Delta t_a \leq t <\Delta t_a + \tau_1$} \\ \gamma\left[\Gamma_{\beta=1}+\Xi_{1:1}\right]& \text{if $\Delta t_a+\tau_1 \leq t <\Delta t_a +\sum_{b=1}^2 \tau_b$} \\\gamma \left[\Gamma_{\beta=B-1}+\Xi_{B-1:1}\right] &\text{if $\Delta t_a+\sum_{b=1}^{B-1}\tau_b\leq t<\Delta t_a + \sum_{b=1}^B \tau_b; B\in\left[3,7\right]$}\\\gamma \Xi_{N:1}& \text{if $t\geq \Delta t_a+\sum_{b=1}^N\tau_b;N=7$}\end{cases}"/>,---> 

<!---in which <img src="https://latex.codecogs.com/svg.latex?\;\gamma=M_{0a}\cdot f \cdot e^{\frac{-\Delta t_a}{T_{1a}}} \cdot T_{1a}"/> and <img src="https://latex.codecogs.com/svg.latex?\;\Gamma_\beta=L_r(\beta+1)\left(1-e^{-\frac{t-\Delta t_a-\sum_{b=1}^{\beta}\tau_b}{T_{1a}}}\right)"/>, and <img src="https://latex.codecogs.com/svg.latex?\;\Xi_{\beta:\beta^\prime}=\sum_{b^\prime=\beta}^{\beta\prime}L_r(b^\prime)\left(e^{-\frac{t-\Delta t_a-\sum_{b=1}^{b^\prime}\tau_b}{T_{1a}}}-e^{-\frac{t-\Delta t_a-\sum_{b=1}^{b^\prime-1}\tau_b}{T_{1a}}}\right)"/>.--->


<p>
    <img src="figures/AnyConv.com__data_generator-1.png" alt>
    <em>Figure 2- Proposed data generator.</em>
</p>

# 5- Results
<p>
    <img src="figures/angiography_res.bmp" alt>
    <em>Figure 3- Results of reconstructed angiography scans for one subject.</em>
</p>

<p>
    <img src="figures/perfusion_res.bmp" alt>
    <em>Figure 4- Results of reconstructed perfusion scans for one subject.</em>
</p>


# Requirments

Tensorflow<2 & python>3.4
