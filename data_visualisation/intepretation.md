1. Look at the "Edge Profile" (Blur Analysis)
The Visualization: Look at the green line vs. the blue line.
The Analysis: If the blue line (Noisy) takes 5-10 pixels to reach the peak that the green line (GT) hits in 2 pixels, you have a large blur radius.
Parameter Fix: You need more num_blocks. If the blur is massive, increase from num_blocks=2 to 3 or 4. The model needs a deeper "Receptive Field" to pull those smeared pixels back into a sharp line.
2. Look at the "FFT Power Spectrum" (Periodic Artifacts)
The Visualization: Look for bright white dots (not in the center) or bright horizontal/vertical lines in the FFT.
The Analysis: Those dots represent periodic interference (scan-lines). Spatial CNNs cannot "see" these; they just think it's part of the circuit.
Parameter Fix: If you see spikes, set freq_weight = 0.1. This forces the Fourier Unit to work harder. If the FFT is "clean" (just a glow in the center), you can keep freq_weight = 0.0 or 0.05.
3. Look at the "Intensity Histogram" (Speckle Range)
The Visualization: Look at the blue tail extending to the right.
The Analysis: If the Noisy LR histogram goes far beyond the GT histogram (e.g., GT ends at 1.0 but LR goes to 1.5), that is confirmed Speckle.
Parameter Fix: This justifies your Global Stats Calibration. If the tail is very long, ensure your p99 isn't too high, or you'll "squash" the real signal too much.
4. Look at the "Absolute Error Map" (Complexity)
The Visualization: Is the error map random "snow" or does it look like a "ghost" of the circuit lines?
The Analysis: If the error map looks like a "ghost" (you can see the shapes of the lines in the error), the model is struggling with structural reconstruction.
Parameter Fix: Increase dim from 64 to 80 or 96. A "wider" model has more capacity to understand complex geometric structures.