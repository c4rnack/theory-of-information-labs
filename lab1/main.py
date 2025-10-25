from math import pi, e, sqrt, atan
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd

A = 3.5
T = 33.3 * 10**(-3) #33.3 ms
ti = (2/3) * T
tau = 0.1 * ti
w1 = 2 * pi / T

n = 12 #delta Fk / (1/T)

def S(t):
    if t >= -ti/2 and t < -3*ti/8:
        return A*e**((t + 3*ti/8)/tau)
    elif t >= -3*ti/8 and t < -ti/4:
        return A*e**(-(t + 3*ti/8)/tau)
    elif t >= ti/4 and t < 3*ti/8:
        return A*e**((t - 3*ti/8)/tau)
    elif t >= 3*ti/8 and t <= ti/2:
        return A*e**(-(t - 3*ti/8)/tau)
    else:
        return 0

t1, t2 = -T/2, T/2

def compute_coefficients(K=20):
    a=[]
    b=[]
    S_vals = []
    phi = []
    P_vals = []

    a0, _ = quad(lambda t: S(t), t1, t2, epsabs=1e-12, epsrel=1e-12)
    a0 = (2/T) * a0
    a.append(a0)
    b.append(0)
    S_vals.append(abs(a0))
    phi.append(0)
    P_vals.append(a0**2)

    for k in range(1, K+1):
        ak, _ = quad(lambda t: S(t) * np.cos(k * w1 *t), t1, t2, epsabs=1e-12, epsrel=1e-12)
        bk, _ = quad(lambda t: S(t) * np.sin(k* w1 * t), t1, t2, epsabs=1e-12, epsrel=1e-12)

        ak, bk = 2/T * ak, 2/T * bk

        Sk = sqrt(ak**2 + bk**2)
        phi_k = np.arctan2(bk,ak)

        P_k = (ak**2) / 2

        a.append(ak)
        b.append(bk)
        S_vals.append(Sk)
        phi.append(phi_k)
        P_vals.append(P_k)

    return np.array(a), np.array(b), np.array(S_vals), np.array(phi), np.array(P_vals)

a_k, b_k, S_k, phi_k, P_k = compute_coefficients(20)

k_vals = np.arange(0, 21)
table_data = {
    'k': k_vals,
    'a_k': a_k,
    'b_k': b_k,
    'S_k': S_k,
    'ф_k': phi_k,
}

df = pd.DataFrame(table_data)
print("\nТаблиця коефіцієнтів і потужностей гармонік:")
print(df.to_string(index=False, float_format='%.6f'))

P = np.sum(P_k[:n])

S2, _ = quad(lambda t:(1/T) * (S(t)**2), t1, t2, epsabs=1e-12, epsrel=1e-12)
print(f"P={P}")
print(f'S^2(t)={S2}')

absolute_error = S2 - P
print(f"Абсолютна похибка: {absolute_error}")
relative_error = absolute_error / S2
print(f"Відносна похибка: {relative_error * 100}%")

t = np.linspace(-T/2, T/2, 2000)
S_vals = np.array([S(val) for val in t])

plt.figure(figsize=(8,4))
plt.plot(t, S_vals, color='blue')
plt.title('Графік функції S(t)')
plt.xlabel('t')
plt.ylabel('S(t)')
plt.grid(True)
plt.show()

k_vals = np.array([k for k in range(0, 21)])

plt.figure(figsize=(8,4))
plt.plot(k_vals, S_k, marker="o")
plt.title("Амплітудно-частотний спектр S_k, k")
plt.xlabel("k")
plt.ylabel("S_k")
plt.xticks(np.arange(0,21,1))
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(k_vals, phi_k, marker="o")
plt.title("Фазо-частотний спектр ф_k, k")
plt.xlabel("k")
plt.ylabel("ф_k")
plt.xticks(np.arange(0,21,1))
plt.grid(True)
plt.show()

module_of_ak = np.abs(a_k)

plt.figure(figsize=(8,4))
plt.plot(k_vals, module_of_ak, marker="o")
plt.title("Енергетичний спектр сигналу")
plt.xlabel("k")
plt.ylabel("|ak|")
plt.xticks(np.arange(0,21,1))
plt.grid(True)
plt.show()
