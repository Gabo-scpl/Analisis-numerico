#-- 4.1 --
#-- 11 --
import math

#Método de Newton-Raphson 
def newtonRaphson(f, df, x0, tol=1.0e-9, max_iter=30):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0: 
            return None
        dx = -fx / dfx
        x = x + dx
        if abs(dx) < tol:
            return x
    return None  # no convergió

#Función objetivo y derivada
def f(x):
    return x * math.sin(x) + 3 * math.cos(x) - x

def df(x):
    return math.sin(x) + x * math.cos(x) - 3 * math.sin(x) - 1

#Búsqueda de raíces en (-6, 6)
a, b, dx = -6.0, 6.0, 0.5  
roots = []

print('Raíces encontradas en el intervalo (-6, 6):')
x = a
while x < b:
    guess = x
    root = newtonRaphson(f, df, guess)
    if root is not None:
        # redondea para evitar raíces duplicadas
        root_rounded = round(root, 5)
        if not any(abs(root_rounded - r) < 1e-4 for r in roots):
            print(f"x ≈ {root_rounded}")
            roots.append(root_rounded)
    x += dx

input("Presiona ENTER para salir.")

#%%
#-- 19 --

u = 2510 #m/s
M0 = 2.8*10**6 #kg
mdot= 13.3*10**3 #kg/s
gr = 9.81 #m/s^2
v_sound = 335 #m/s


def g(t):
    if M0 - mdot * t <= 0:
        return float('inf')
    return u * math.log(M0 / (M0 - mdot * t)) - gr * t - v_sound

def dg(t):
    D = M0 - mdot * t
    if D <= M0:
        return float('inf')
    return (u * mdot / D) - g

t0 = 10 #s
t_final = newtonRaphson(g, dg, t0)
print(f"La velocidad del sonido (335 m/s) se alcanza en t ≈ {t_final:.4f} segundos.")    

#%%
#-- 5.1 --
#-- 9 --
import numpy as np
def LUdecomp3(c,d,e): 
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b): 
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b   

def curvatures(xData,yData): #Obtiene los ki de la interpolación cúbica
  xData = np.array(xData)
  yData = np.array(yData)

  n = len(xData) - 1
  c = np.zeros(n)
  d = np.ones(n+1)
  e = np.zeros(n)
  k = np.zeros(n+1)
  c[0:n-1] = xData[0:n-1] - xData[1:n]
  d[1:n] = 2.0*(xData[0:n-1] - xData[2:n+1])
  e[1:n] = xData[1:n] - xData[2:n+1]
  k[1:n] =6.0*(yData[0:n-1] - yData[1:n]) /(xData[0:n-1] - xData[1:n]) -6.0*(yData[1:n] - yData[2:n+1]) \
  /(xData[1:n] - xData[2:n+1])

  c, d, e = LUdecomp3(c,d,e)
  LUsolve3(c,d,e,k)
  return k

xData = [0.0, 0.1, 0.2, 0.3, 0.4]
yData = [0.000000, 0.078348, 0.138910, 0.192916, 0.244981]

print(curvatures(xData,yData))

#%%
#-- 10 --
def o(x,n): #La función a derivar con n decimales
  return round(math.sin(x),n)

def d2fc(x,h,o,n): #Primer derivada de f con aproximación central con n decimales
  d2fc=(f(x+h,n)+f(x-h,n))/(2*h)
  return d2fc

def d2ff(x,q,o,n): #Segunda derivada de f con aproximación forward con n decimales
  d2ff=(-o(x+2*q,n)+4*o(x+q,n)-3*o(x,n))/(2*q)
  return d2ff

#Primer dervidada con aproximacion central

h=0.93
print("Con la aproximación central tenemos que")
print("  h        6 dígitos   Error    8 dígitos     Error")
print("------------------------------------------------------")
for i in range(10):
  E1=abs(((o(1,6)-d2fc(1,h,o,6))/o(1,6))*100)
  E2=abs(((o(1,8)-d2fc(1,h,o,8))/o(1,8))*100)
  print("%.6o   %.6o    %.2o     %.8o    %.2o" %(h,d2fc(1,h,o,6),E1,d2fc(1,h,o,8),E2))
  h=h/2

#Primer dervidada con aproximacion forward

q=0.93
print("Con la aproximación forward tenemos que")
print("  h        6 dígitos   Error    8 dígitos     Error")
print("------------------------------------------------------")
for i in range(10):
  E1=abs(((o(1,6)-d2ff(1,h,o,6))/o(1,6))*100)
  E2=abs(((o(1,8)-d2ff(1,h,o,8))/o(1,8))*100)
  print("%.6o   %.6o    %.2o     %.8o    %.2o" %(h,d2ff(1,h,o,6),E1,d2ff(1,h,o,8),E2))
  h=h/2

#%%
#-- 6.1 --
#-- 1 --
def trapecio_recursiva(p,a,b,Iold,k):
  if k == 1: Inew = (p(a) + p(b))*(b - a)/2.0
  else:
    n = 2**(k -2 ) # numero de nuevos puntos
    h = (b - a)/n # espaciamiento de nuevos puntos
    x = a + h/2.0
    sum = 0.0
    for i in range(n):
      sum = sum + p(x)
      x = x + h
      Inew = (Iold + h*sum)/2.0
  return Inew

def p(x):
    return math.log(1 + math.tan(x))
Iold = 0.0
for k in range(1,21):
  Inew = trapecio_recursiva(p,0.0,math.pi/4,Iold,k)
  if (k > 1) and (abs(Inew - Iold)) < 1.0e-6: break
  Iold = Inew

print('Integral =',Inew)
print('n Panels =',2**(k-1))

print('Como vemos, esto nos ha arrojado dos paneles, lo que nos dice que el programa nos ha separado la integral en dos secciones para una mejor precision y facilitar los calculos')