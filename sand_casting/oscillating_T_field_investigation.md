# Investigating temperature field instabilities  

### Temperature update loop

$q_x = k\frac{dT}{dx}$ 

where:  $q =$ heat flux per unit area, $k =$ thermal conductivity

Temperature is updated using the formula:   
$∆T = \large\frac{q \times dt}{shc \times density \times pixdim^2}$ in addition to a term related to latent heat release/absorption

### Cause of instability

If `dt` is too high or `pix_dim` is too low ($\rightarrow$ high SA:Vol ratio), heat will continue flowing between two pixels even after equilibrium should have been reached (since the simulation has not updated yet) resulting in heat conduction up a temperature gradient and overshooting of equilibrium temperature.

Fitting a straight line of best fit ($R^2 = 0.9991$) determines the ***condition for stability:***

$\large\frac{pixdim^2}{dt} > 3.297 \times 10^{-5}$

Note that the condition depends on simulation parameters such as thermal conductivity of the materials (pure aluminium was used in this example).  

Code used to test stability limit: [oscillating_T_field_investigation.py](oscillating_T_field_investigation.py)

### Implementation in code

The more general governing condition is the ***Courant–Friedrichs–Lewy condition,*** which in this case is: 

$\large\frac{k \times dt}{shc \times density \times pixdim^2} > 0.25$

This condition has been implemented into the simulation code. 


![oscillating_T_field_stability_test.png](oscillating_T_field_stability_investigation.png)

![oscillating_T_field.gif](oscillating_T_field.gif)

