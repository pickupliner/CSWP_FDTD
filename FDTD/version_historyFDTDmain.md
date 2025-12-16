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