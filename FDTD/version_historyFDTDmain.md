## BORO// 23/11/25

* ik heb scatterer naar $__main__.py$ gezet. (zodat ik zeker ben in welke file ik moet kijken voor onze updated code)
* ik heb units gezet zodat het duidelijker wordt bij berekeningen
* Y was twee keer assigned dus ga ik die eerste Y van naam veranderen naar kappaY
* 

## Joran 9/12

* Ik heb de wedge/triangle toegevoegd in __main__.py
* Het is voor een harde wand (o_n = 0), Z=? was iets te ingewikkeld
* Ik heb een nieuwe file non_cartesian.py gemaakt, hier is een niet orthogonaal grid gebruikt (mist nog absorberende randen) waarin de triangle vanzelfsprekender is

## Joran 16/12

* Grid refinement rond de wedge
* dx en dy zijn momenteel stap functies (zijn 2 keer zo fijn rond wedge)
* Nog te doen: smooth interpolatie van dx en dy
* instabiel: CN moet < 0.5
* rectangle is broken: dx en dy in BC moeten nog geupdate worden

## Joran 18/12

* Added parabolic coordinate transformation: dx and dy now change linearly, no longer step function
* Made plots of transformed spatial coordinates and steps
* fixed error in comment of F2 and F3, with/without floor
* fixed trechter achtige vorm bij bron: was doordat de bron in de PML stond
* added interpolation of derivatives for stability with parabolic coordinate transfo
* changed placing of source to match possibility of non-uniform grid
* did the same for observation points
* rectangle works again (see previous)
* PROBLEMS: parabolic interpolation is still unstable, tho at longer timescales

## Joran 19/12

* Fixed CN: we were working with an incorrect definition
* Now everything is stable, even non-cartesian is now stable at CN=1
* Changed Z from 0 to 2 as in the assignment to see its effect
* PROBLEM: I looked back at some older versions and it seems that the triangle used to be way better without any refinement; now it has some ugly oscillations at the edge.

## Joran 22/12 (non-cartesian)

* added PML
* problem: upper PML is implemented incorrectly
* fixed: upper PML is now implemented 'correctly', however previous simple approach seems to have been better

## Joran 24/12 (non-cartesian)

* I had the sine and cosine swapped around in the upper PML
