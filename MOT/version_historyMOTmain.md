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

## Boro 17/12 all added to testmot because something weird happens

* added fprime1(calculates time derivative)
* added Incidentfield1(calculation of Ei field)
* added animation of incident field in radial direction
* added T and made that the pulsewidth (worked weird) T1 is the first pulse
* adjusted time for the start pulse
* changed atan2 to arctan2 (my numpy behaved weirdly)


* notes
    - I feel like i still need to adjust for boundary condition using FEM(i don't know how I can do that need to look later today)
    - when i changed the width p