from numpy import *
import random
import jedi
t = arange(0, 650, 1)
sigma = 4.5
# X_at = []
# for t in t:
#     X = 1715*e**(0.0305*t)/(1715+e**(0.0305*t)-1)+285
#     X_at.append(X)


def haha():
    X_at = [1715*e**(0.0305*t)/(1715+e**(0.0305*t)-1)+285 for t in t]
    Ts = [15 + (sigma/log(2))*log(X/285) for X in X_at]
    print('hello')


haha()
