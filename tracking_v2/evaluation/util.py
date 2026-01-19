import math

def anderson_darling_pvalue(z: float) -> float:
    if z >= .6:
        p = math.exp(1.2937 - 5.709*z - .0186*(z**2))
    elif z >=.34:
        p = math.exp(.9177 - 4.279*z - 1.38*(z**2))
    elif z >.2:
        p = 1 - math.exp(-8.318 + 42.796*z - 59.938*(z**2))
    else:
        p = 1 - math.exp(-13.436 + 101.14*z - 223.73*(z**2))
    return p
