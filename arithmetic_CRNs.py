import numpy as np

def plus(ap,an,bp,bn,xp,xn) : 
    return np.array([ap+bp-xp,an+bn-xn])

def minus(ap,an,bp,bn,xp,xn) : 
    return np.array([ap+bn-xp,an+bp-xn])
       
def times(ap,an,bp,bn,xp,xn) : 
    return np.array([ap*bp+an*bn-xp,ap*bn+an*bp-xn])

def divide(ap,bp,xp) : 
    if(bp==0) : return np.array([ap-bp*xp])
    return np.array([ap-bp*xp])/bp