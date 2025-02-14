x=float(input('Inngrese los ºF a convertir: '))
C=(x-32)*5/9
print('Los',x,'ºF equivalen a',C,'ºC')


from cmath import sin, sinh, exp, e,cos, tan
y=float(input('Ingrese un valor para y: '))
s1=sinh(y)
print(s1)

s2=(exp(y)-exp(-y))/2

s3=(e**(y)-e**(-y))/2

print(s1,'=',s2,'=',s3)

import random as  rn

z=rn.randint(0, 20)
s4=sin(z*1j)
s5=1j*sinh(z)

s6=cos(z)+1j*sin(z)
s7=exp(1j*z)

print('Seno de(',z,'i)=',s4,';Seno Hiperbòlico de (',z,'i)=',s5)

from sympy import(symbols, diff, integrate, Rational, lambdify)

t, v0, g= symbols('t v0 g')
y=v0*t-Rational(1,2)*g*t**2
print('primer derivada',diff(y,t))
print('Segunda derivada',diff(y,t,t))

from math import radians

v1=float(input('Ingrese la velocidad inicial: '))
x1=float(input('Ingrese la coordenada horizontal: '))
y0=float(input('Ingrese la altura inicial: '))
w=float(input('Ingrese el angulo en grados: '))
g=9.81
w=radians(w)
h=x1 * tan(w) - (Rational(g/2) / v1 ) * (( x1**2 ) / ( cos(w)**2 )) + y0
print(type(h),h)

print('La trayectoria de la pelota es', h,'m')