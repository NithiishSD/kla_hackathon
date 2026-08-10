good range for our model:



Metric	Your Current	"Good" (Baseline)	"Winning" (Fab-Ready)	Critical Interpretation
PSNR	29.42 dB	28–31 dB	33–36+ dB	Measure of global intensity fidelity. Above 32dB, speckle noise is effectively invisible.
SSIM	0.7835	0.80–0.85	0.92–0.96	Measure of structural fidelity. Your 0.78 is why the images look "melted." You must hit >0.85 to be competitive.
LPIPS	0.2384	0.15–0.20	< 0.10	Perceptual similarity. Lower is better. Note: LPIPS is biased toward natural textures, so don't obsess over it as much as SSIM.

CD Bias	< ±0.5 pixels	Essential! Ensures you aren't shrinking or growing the hardware.
LER Error	< 5.0 %	Ensures you are preserving the "roughness" without fake smoothing.
Slope Fidelity	> 90 %	Ensures edges are as "sharp" as the original.