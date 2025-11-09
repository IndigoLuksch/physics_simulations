# Investigating temperature field instabilities  

$q = k∆T$ 

where:  $q =$ heat flux per unit area, $k =$ thermal conductivity

Temperature is updated using the formula:   
$dT = \large\frac{q \times dt}{shc \times density \times pixdim}$

If `dt` is too high or `pix_dim` is too low ($\rightarrow$ high SA:Vol ratio), heat will continue flowing between two pixels even after equilibrium should have been reached (since the simulation has not updated yet) resulting in heat conduction up a temperature gradient. 

Fitting a straight line of best fit ($R^2 = 0.9991$) determines the ***condition for stability:***

$\frac{pixdim^2}{dt} > 3.297 \times 10^{-5}$

Code used to test stability limit: [oscillating_T_field_investigation.py](oscillating_T_field_investigation.py)

![oscillating_T_field_stability_test.png](oscillating_T_field_stability_investigation.png)

![oscillating_T_field.gif](oscillating_T_field.gif)

