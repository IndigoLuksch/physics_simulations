# Metal solidification 

Simulates solidification of metals

Inputs: phase diagram (.tbd file), simulation parameters 

Outputs: animation of grain growth and temperature field

## How it works 
Parallelised using Numba for fast processing: 

- Temperature change calculated using thermal diffusion laws
- Kinetic Monte Carlo simulation updates state of each pixel 

Not parallelised: 

- Extraction of data from phase diagram using phycalphad

## Acknowledgements 
Framework for [config.py](config.py), [phase_diagram.py](phase_diagram.py) created using Claude / Gemini 

Code is generally handwritten, but Claude / Gemini was used throughout the coding process (especially for syntax) 

