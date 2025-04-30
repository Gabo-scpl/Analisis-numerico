# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 20:31:04 2025

@author: el_ga
"""
import sys
import math 
import numpy as np
import matplotlib.pyplot as plt
# -- Examen 2 --
# -- Ejercicio 1 --

    
def err(string):
  print(string)
  input('Press return to exit')
  sys.exit()

def newtonRaphson(f,df,a,b,tol=1.0e-4):
  from numpy import sign
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa) == sign(fb): err('La raiz no esta en el intervalo')
  x = 0.5*(a + b)
  for i in range(30):
    print(i)
    fx = f(x)
    if fx == 0.0: return x 
    if sign(fa) != sign(fx): b = x # Haz el intervalo mas pequeño
    else: a = x
    dfx = df(x)  
    try: dx = -fx/dfx # Trata un paso con la expresion de Delta x
    except ZeroDivisionError: dx = b - a # Si division diverge, intervalo afuera
    x = x + dx # avanza en x
    if (b - x)*(x - a) < 0.0: # Si el resultado esta fuera, usa biseccion
      dx = 0.5*(b - a)
      x = a + dx 
    if abs(dx) < tol*max(abs(b),1.0): return x # Checa la convergencia y sal
  print('Too many iterations in Newton-Raphson')
x=75   
def f(x):
   return x**3 - 75 
    
    
def df(x): 
    return 3*x**2
  
root = newtonRaphson(f,df,-5,5)
print('Root =',root)



#%%
#-- Ejercicio 2 --

def y(x):                    # define la funcion y(x)
  y = x**3 - 3.23*x**2 - 5.54*x + 9.48
  return y

x1 = 1 # peticion de valor x1
x2 = 2 # peticion de valor x2
print('x1=1')
print('x2=2')
y1 = y(x1)                                    # evalua la funcion y(x1)
y2 = y(x2)                                    # evalua la funcion y(x1)
                                  # evalua la funcion y(x1)

if y1*y2 > 0:                                 # prueba si los signos son iguales
  print('No hay raices en el intervalo')
  exit

for i in range(100):
  xh = (x1+x2)/2
  yh = y(xh)                                  # evalua la funcion y(xh)
  y1 = y(x1)                                  # evalua la funcion y(x1)
  if abs(y1) < 1.0e-6:
    break
  elif y1*yh < 0:
    x2 = xh
  else:
    x1 = xh
print('La raiz es: %.5f' % x1)
print('Numero de bisecciones: %d' % (i+1))

#%%
#-- Ejercicio 3 --


#%%
#-- Ejercicio 4 --
#para resolverlo proponemos el cambio de variable:
# Sea t = sqrt(x)
def trapecio_recursiva(f,a,b,Iold,k):
  if k == 1: Inew = (f(a) + f(b))*(b - a)/2.0
  else:
    n = 2**(k -2 ) # numero de nuevos puntos
    h = (b - a)/n # espaciamiento de nuevos puntos
    x = a + h/2.0
    sum = 0.0
    for i in range(n):
      sum = sum + f(x)
      x = x + h
      Inew = (Iold + h*sum)/2.0
  return Inew


def f(x):
    return math.sin(x**2) * 2*x
Iold = 0.0
for k in range(1,21):
  Inew = trapecio_recursiva(f,0.0,math.pi,Iold,k)
  if (k > 1) and (abs(Inew - Iold)) < 1.0e-6: break
  Iold = Inew

print('Integral =',Inew)
print('n Panels =',2**(k-1))