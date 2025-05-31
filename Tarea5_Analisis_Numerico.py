# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:35:39 2025

@author: el_ga
"""
#%%
#-- 7.1 --
# -- ejercicio 3 --
import numpy as np
import matplotlib.pyplot as plt

# Método de Euler
def eulerint(F, x, y, xStop, h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h, xStop - x)
        y = y + h * F(x, y)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# Sistema de ecuaciones: y' = sin(y)
def F(x, y):
    return np.array([np.sin(y[0])])  # y es array con un solo valor

# Imprimir solución
def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n     x         y")
        print("-----------------------")
        
    def imprimeLinea(x, y, n):
        print("{:8.4f}  {:10.6f}".format(x, y[0]))

    n = len(Y[0]) if isinstance(Y[0], (list, np.ndarray)) else 1
    imprimeEncabezado(n)
    for i in range(0, len(X), frec):
        imprimeLinea(X[i], Y[i], n)
    if i != len(X) - 1:
        imprimeLinea(X[-1], Y[-1], n)

# Condición inicial y ejecución
y0 = np.array([1.0])  # y(0) = 1
X, Y = eulerint(F, 0.0, y0, 0.5, 0.1)

# Mostrar resultados
print("La solución es:")
imprimeSol(X, Y, 1)

# Graficar
plt.plot(X, Y[:,0], 'o-', label="Euler y' = sin(y)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solución con Método de Euler")
plt.grid()
plt.legend()
plt.show()

#%%
#-- ejercicio 4 --

# Método de Euler
def eulerint(F, x, y, xStop, h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h, xStop - x)
        y = y + h * F(x, y)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

#Ecuacion diferencial
def F(x, y):
    return y**(1/3) 
#Solucion no trivial
def y_exact(x):
    return (2 * x / 3) ** (3 / 2)

# Imprimir solución
def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n     x         y")
        print("-----------------------")
        
    def imprimeLinea(x, y, n):
        print("{:8.4f}  {:10.6f}".format(x, y[0]))

    n = len(Y[0]) if isinstance(Y[0], (list, np.ndarray)) else 1
    imprimeEncabezado(n)
    for i in range(0, len(X), frec):
        imprimeLinea(X[i], Y[i], n)
    if i != len(X) - 1:
        imprimeLinea(X[-1], Y[-1], n)

# Condición inicial y ejecución
y0 = np.array([0.0])  # y(0) = 0

#Caso a
X_a, Y_a = eulerint(F, 0.0, 0.0, 2.0, 0.01)

#Caso b
X_b, Y_b = eulerint(F, 0.0, 1e-16, 2.0, 0.01)

#Solucion exacta
X_exact = np.linspace(0.0, 2.0, 500)
Y_exact = y_exact(X_exact)


# Mostrar resultados
print("La solución es:")
imprimeSol(X, Y, 1)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(X_a, Y_a, label="Euler: y(0) = 0", linestyle='--')
plt.plot(X_b, Y_b, label="Euler: y(0) = 1e-16")
plt.plot(X_exact, Y_exact, label="Solución exacta: (2x/3)^(3/2)", color='black')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Soluciones de dy/dx = y^{1/3} con el Método de Euler")
plt.legend()
plt.grid(True)
plt.show()

#%%
#-- 8.1 --
#-- ejercicio 3 --

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def Run_Kut4(F,x,y,xStop,h):
  def run_kut4(F,x,y,h):
    K0 = h*F(x,y)
    K1 = h*F(x + h/2.0, y + K0/2.0)
    K2 = h*F(x + h/2.0, y + K1/2.0)
    K3 = h*F(x + h, y + K2)
    return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0
  X = []
  Y = []
  X.append(x)
  Y.append(y)
  while x < xStop:
    h = min(h,xStop - x)
    y = y + run_kut4(F,x,y,h)
    x=x+h
    X.append(x)
    Y.append(y)
  return np.array(X),np.array(Y)

def imprimeSol(X,Y,frec):
 
  def imprimeEncabezado(n):
    print("\n x ",end=" ")
    for i in range (n):
      print(" y[",i,"] ",end=" ")
    print()

  def imprimeLinea(x,y,n):
    print("{:13.4e}".format(x),end=" ")
    for i in range (n):
      print("{:13.4e}".format(y[i]),end=" ")
    print() 
  
  m = len(Y)
  try: n = len(Y[0])
  except TypeError: n = 1
  if frec == 0: frec = m
  imprimeEncabezado(n)
  for i in range(0,m,frec):
   imprimeLinea(X[i],Y[i],n)
  if i != m - 1: imprimeLinea(X[m - 1],Y[m - 1],n)
  
# Ecuacion diferencial para el inciso a
def F_a(x,y):
  return np.array([y[1],-np.exp(-y[0])])
#Ecuacion diferencial para el inciso c
def F_b(x,y):
    return np.array([y[1], np.cos(x * y[0])])

y=np.array([1.0,0.5])

#Metodo del disparo para los incisos a y c
def shoot_a(s):
    y0 = np.array([1.0, s])
    X, Y = Run_Kut4(F_a, 0.0, y0, 1.0, 0.01)
    return Y[-1, 0] - 0.5

def shoot_b(s):
    y0 = np.array([0.0, s])
    X, Y = Run_Kut4(F_b, 0.0, y0, 2.0, 0.01)
    return Y[-1, 0] - 2.0

#Metodo de disaro
res = root_scalar(shoot_a, bracket=[-5, 5], method='brentq')
s_sola = res.root
print("Valor de disparo s para el inciso a:", s_sola)

res = root_scalar(shoot_b, bracket=[-10, 10], method='brentq')
s_solb = res.root
print("Valor del disparo s para el inciso b:", s_solb)

X,Y=Run_Kut4(F_a,0.0,np.array([1.0, s_sola]),1.0,0.01)
X1,Y1=Run_Kut4(F_b,0.0,np.array([0.0, s_solb]),2.0,0.01)

print("La solución es")
imprimeSol(X,Y,4)

print("\nSolución para el inciso c:")
imprimeSol(X1, Y1, 4)

x1=np.arange(0.0,2.0,0.001)
plt.plot(X,Y[:,0],"o",label="Método de Runke-Kutta 4, inciso a")
plt.plot(X1,Y1[:,0],"o",label="Método de Runke-Kutta 4, inciso c")
plt.xlim([0, 2])
plt.grid()
plt.legend()
plt.show()