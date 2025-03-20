# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 21:21:15 2025

@author: el_ga
"""
#Ejercicio 1
import numpy as np
a = np.array([[2.0, -1.0, 3.0],
              [0.0, 3.0, -1.0],
              [7.0, -5.0, 0.0],
            ])

b = np.array([[24.0], [14.0], [6.0]])

def gaussElimin(a,b):
  n = len(b)
  # Fase de eliminacion
  for k in range(0,n-1):
    for i in range(k+1,n):
      if a[i,k] != 0.0:
        lam = a [i,k]/a[k,k]
        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
        b[i] = b[i] - lam*b[k]
  # Fase de sustitucion hacia atras
  for k in range(n-1,-1,-1):
    b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
  return b

print(f"El punto (x, y, z) de interseccion es: \n {gaussElimin(a,b)}")

#%%
# Algoritmo de eliminación de Gauss
a = np.array([[2.0, 1.0],
              [1.0, 2.0],
            ])


b = np.array([[1.0], [0.0]])

def gaussElimin(a,b):
  n = len(b)
  # Fase de eliminacion
  for k in range(0,n-1):
    for i in range(k+1,n):
      if a[i,k] != 0.0:
        lam = a [i,k]/a[k,k]
        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
        b[i] = b[i] - lam*b[k]
  # Fase de sustitucion hacia atras
  for k in range(n-1,-1,-1):
    b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
  return b

print(f"La carga de los quarks up y down son respectivamente: \n {gaussElimin(a,b)}")

#%%
#Ejercicio 3

a =np.array([[ 1, 5, 10, 20],[ 0, 1,-4, 0],[ -1, 2, 0, 0],[ 1, 1, 1, 1]])
b = np.array([[95.0], [0.0], [1.0], [26.0]])

def gaussElimin(a,b):
  n = len(b)
  # Fase de eliminacion
  for k in range(0,n-1):
    for i in range(k+1,n):
      if a[i,k] != 0.0:
        lam = a [i,k]/a[k,k]
        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
        b[i] = b[i] - lam*b[k]
  # Fase de sustitucion hacia atras
  for k in range(n-1,-1,-1):
    b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
  return b
print(f"El numero asociado a los meteoros de 1, 5, 10 y 20 kg es: \n {gaussElimin(a, b)}")