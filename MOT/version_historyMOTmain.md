## Joran  2/12
 * Added some plots and fixed an error in an equation.
 * Added a file to test when Hankel2 function can be truncated
 * Commented out some code that wasn't doing anything
 * Wondered what was necessary to get some sensible output
 ## Joran 4/12
 * Continued on what I was doing
 * Added some more figures
 * Made Z function be cached (memoized) to increase speed
 * Changed a few parameters (k=omega/c, ..)
 * Got some reasonable result
 * Got some less reasonable results:
    - For high omega serious instability issues (due to A being near zero?)
    - Analytical in function of phi doesn't seem to make sense to me, would expect more like what the numerical looks like
    - ...

## Boro 18/12 all added to testmot because something weird happens

* added fprime1(calculates time derivative)
* added Incidentfield1(calculation of Ei field)
* added animation of incident field in radial direction
* added T and made that the pulsewidth (worked weird) T1 is the first pulse
* adjusted time for the start pulse
* changed atan2 to arctan2 (my numpy behaved weirdly)

* put it into testmot into __main__
* notes
    - I feel like i still need to adjust for boundary condition using FEM(i don't know how I can do that need to look later today)
    - when i changed the width p

## Boro 19/12 Class switch
* turned it into a class for convenience and clarity
* added solve function that solves basically
* all the plotting of 6.1 is in plot61
* only problem still need to resize the Ei and generelize Ei such that i can calculate scattering in everyway

## BORO 21/12 
* added animations for the current
* added more comments and changed variable names for clarity

## Joran 23/12
* Removed two redundant dimensions in quadrature coordinates and weights
* Removed some boilerplate that came along with it
## BORO 23/12
* added analytical zeros which finds the omega of the analytical solution of the eigenmodes. omega n,m equals m-th order zero and n-th order besselfunction
* added text when it says that the A is too small thus the current is too large

## Joran 24/12
* Compared L with 2 c dt (commented out: lines 82-87), can affirm that L is a constant value (as expected) and L=l, so l is smaller than 2 c dt.

## Joran 27/12
* Remade MOT code in new file: tmp.ipynb
* Happy to announce that it seems to work and seems stable for longer timescales
* I'll be changing the old file to use the new implementation some time soon

## Boro 28/12
* changed parameters for physically better simulations

## Joran 29/12
* Update __main.py__ to have the new implementation from tmp.ipynb
* gave init a curve variable: handy for when we want something other than a circle as a PEC. With this I removed R and N_S: N_S is derived from the shape of curve
* made dt also a variable in init
* Removed E_i from MOT into the Q_6_1_validation function and with it the _init_incident_pulse
* Replaced the Z and F function with _init_F and _init_Z, which are necessary for the new version in which Z and F are arrays.
* In build_geometry I made self.rho the midpoints of all segments,
self.rhop is the linear interpolation along the segments, and self.curve is the edges of the segments (what was previously self.curvepoints).
* Since MOT has no R field anymore updated some functions to take an R parameter.
* the solve function now takes an argument V which is the excitation, or the incident field.
* Moved analytical current to Q_6_1_validation function
* Broke functions animate_Ei1 and animate_Ei1_2D, because I moved E_i out of the MOT.
* Made two functions, Q_6_1_validation and Q_6_2_cylindrical_cavity for those questions repectively.