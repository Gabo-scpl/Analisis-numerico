# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 20:58:07 2025

@author: el_ga
"""
#%%
#Ejercicio 1

import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi

# Función que evalúa el polinomio de Newton
def evalPoly(a, xData, x):
    n = len(xData) - 1
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - xData[n - k]) * p
    return p

# Función para calcular coeficientes de Newton
def coeffts(xData, yData):
    m = len(xData)
    a = np.copy(yData).astype(float)  # Copia de yData como float
    for k in range(1, m):
        for i in range(m - 1, k - 1, -1):  # Corrección en el cálculo de diferencias divididas
            a[i] = (a[i] - a[i - 1]) / (xData[i] - xData[i - k])
    return a

# Datos
xData = np.array([0.0, 21.1, 37.8, 54.4, 71.1, 87.7, 100.0])
yData = np.array([0.101, 1.79, 1.13, 0.696, 0.519, 0.338, 0.296])

# Calcular coeficientes de Newton
coeff = coeffts(xData, yData)

# Puntos para graficar la interpolación
x = np.linspace(0, 100, 100)  # Más puntos para una curva suave
yInterp = np.array([evalPoly(coeff, xData, xi) for xi in x])  # Evaluar en cada xi

# Puntos de prueba específicos para ver interpolación
xTest = np.array([10, 30, 60, 90])
yTestInterp = np.array([evalPoly(coeff, xData, xi) for xi in xTest])

# Graficar interpolación y puntos
plt.plot(x, yInterp, "r-", label="Interpolación de Newton")  # Curva de interpolación
plt.plot(xData, yData, "bo", label="Datos originales")  # Datos originales
plt.plot(xTest, yTestInterp, "ro", markersize=8, label="Puntos interpolados")  # Puntos interpolados
plt.xlabel("Temperatura°C")
plt.ylabel("Viscocidad cinetica (10^-3m^2/s)")
plt.legend()
plt.grid()
plt.show()


#%%
#Ejercicio 2

import sympy as sp
def lagrange_1(x_points, y_points, xp):
    """
    Calcula y grafica el polinomio de interpolación de Lagrange.

    Parámetros:
    x_points (list or array): Puntos en el eje x.
    y_points (list or array): Puntos en el eje y.
    xp (float): Punto en el que se desea interpolar.

    Retorna:
    yp (float): Valor interpolado en xp.
    """
    m = len(x_points)
    n = m - 1
    # Definir la variable simbólica
    x = sp.symbols("x")

    # Función para calcular los polinomios básicos de Lagrange
    def lagrange_basis(xp, x_points, i):
        L_i = 1
        for j in range(len(x_points)):
            if j != i:
                L_i *= (xp - x_points[j]) / (x_points[i] - x_points[j])
        return L_i

    # Función para calcular el polinomio de Lagrange
    def lagrange_interpolation(xp, x_points, y_points):
        yp = 0
        for i in range(len(x_points)):
            yp += y_points[i] * lagrange_basis(xp, x_points, i)
        return yp

    # Calcular el valor interpolado
    yp = lagrange_interpolation(xp, x_points, y_points)
    print("For x = %.1f, y = %.1f" % (xp, yp))

    # Crear puntos para la interpolación
    x_interpolado = np.linspace(min(x_points), max(x_points), 100)
    y_interpolado = [
        lagrange_interpolation(x_val, x_points, y_points) for x_val in x_interpolado
    ]

    # Graficar los puntos originales
    plt.scatter(x_points, y_points, label="Puntos Originales", color="red")

    # Graficar el polinomio de interpolación de Lagrange
    plt.plot(
        x_interpolado, y_interpolado, label="Interpolación de Lagrange", linestyle="-"
    )

    # Graficar el valor interpolado
    plt.scatter(xp, yp, color="blue", zorder=5)
    plt.text(xp, yp, f"({xp:.1f}, {yp:.1f})", fontsize=12, verticalalignment="bottom")

    # Añadir etiquetas y leyenda
    plt.xlabel("altura(km)")
    plt.ylabel("Densidad relativa del aire")
    plt.title("Polinomio de Interpolación de Lagrange")
    plt.legend()
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

    # Construir el polinomio de interpolación simbólicamente
    polinomio = 0
    for i in range(len(x_points)):
        term = y_points[i]
        for j in range(len(x_points)):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        polinomio += term

    # Simplificar el polinomio
    polinomio_simplificado = sp.simplify(polinomio)

    # Imprimir el polinomio de interpolación
    print("Polinomio de Interpolación de Lagrange:")
    print(f"y(x) = {polinomio}")
    print("\nPolinomio Simplificado:")
    print(f"y(x) = {polinomio_simplificado}")

    return yp
try:
    x_points = [0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150]
    y_points = [1, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741]
    xp = float(10.5)
    lagrange_1(x_points, y_points, xp)
except ValueError:
    print("Please insert a valid number")

#%%
#Ejercico 3

def evalPoly(a, xData, x):  # Función que evalua polinomios de Lagrange
    n = len(xData) - 1  # Grado del polinomio
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - xData[n - k]) * p
    return p


def coeffts(xData, yData):
    m = len(xData)  # Número de datos
    a = yData.copy()
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (xData[k:m] - xData[k - 1])
    return a

#Definimos nuestros datos iniciales
xData = np.array([0.0, 400.0, 800.0, 1200.0, 1600.0])
yData = np.array([0.0, 0.072, 0.233, 0.712, 3.400])

coeff = coeffts(xData, yData)

#Creamos esto para ajustar y suavizar el polinomio
x = np.linspace(0, 2500, 100)
yInterp = np.array([evalPoly(coeff, xData, xi) for xi in x])

#Creamos un rango de valores para interpolarlos a los demas
xTest = np.arange(0, 2500, 250)
yTestInterp = np.array([evalPoly(coeff, xData, xi) for xi in xTest])

#graficamos la linea, los datos iniciales y datos interpolados
plt.plot(x, yInterp, "r", label="Newton")
plt.plot(xData, yData, "o", label="Datos")
plt.plot(xTest, yTestInterp, "ro", markersize=8, label="Puntos interpolados")
plt.xlabel("Velocidad (rpm)")
plt.ylabel("Amplitud (mm)")
plt.legend()
plt.grid()
plt.show()



