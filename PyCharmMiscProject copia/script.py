import numpy as np
import matplotlib.pyplot as plt

# --- Datos de entrada (actualizados con los tiempos precisos) ---
# Tiempos de decaimiento en días (td)
td_dias = np.array([0.0000, 3.8264, 4.9792])

# Actividad en Bq para cada tiempo
actividades_bq = np.array([8.868, 3.154, 1.788])

# --- Calcular el logaritmo natural de la actividad ---
ln_actividades = np.log(actividades_bq)

# --- Realizar la regresión lineal ---
# Usamos np.polyfit para ajustar una línea recta (grado 1) a los datos.
# Devuelve los coeficientes del polinomio [m, b] donde m es la pendiente y b es la ordenada al origen.
# Aquí, la pendiente m será -lambda.
pendiente_lambda, intercepto = np.polyfit(td_dias, ln_actividades, 1)

# La constante de desintegración (lambda) es el negativo de la pendiente
lambda_calculado = -pendiente_lambda

# --- Calcular la vida media (T1/2) ---
T_half_calculado = np.log(2) / lambda_calculado

# --- Generar puntos para la línea de regresión ---
# Creamos un rango de tiempos para graficar la línea de ajuste
td_fit = np.linspace(min(td_dias) - 0.5, max(td_dias) + 0.5, 100)
# Calculamos los valores de ln(Actividad) predichos por el modelo de regresión
ln_actividades_fit = pendiente_lambda * td_fit + intercepto
# Convertimos de nuevo a actividad para la escala lineal si se desea graficar así
actividades_fit = np.exp(ln_actividades_fit)

# --- Impresión de resultados ---
print(f"Constante de desintegración (lambda): {lambda_calculado:.4f} días^-1")
print(f"Vida media (T_1/2) calculada: {T_half_calculado:.3f} días")
print(f"Valor conocido de T_1/2 para Au-198: 2.695 días")
print("\nNota: La diferencia con el valor conocido se debe a la variabilidad de los datos experimentales y al número limitado de puntos.")

# --- Graficación ---
plt.figure(figsize=(10, 6))

# Gráfico de ln(Actividad) vs. Tiempo (para mostrar la linealidad)
plt.subplot(1, 2, 1) # 1 fila, 2 columnas, primer gráfico
plt.plot(td_dias, ln_actividades, 'o', label='Datos Experimentales (ln(Actividad))')
plt.plot(td_fit, ln_actividades_fit, '-', color='red', label=f'Ajuste Lineal\n($\\lambda$ = {lambda_calculado:.4f} días$^{{-1}}$)')
plt.title('Regresión Lineal: ln(Actividad) vs. Tiempo')
plt.xlabel('Tiempo de decaimiento ($t_d$) [días]')
plt.ylabel('ln(Actividad) [ln(Bq)]')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Gráfico de Actividad vs. Tiempo (para mostrar el decaimiento exponencial)
plt.subplot(1, 2, 2) # 1 fila, 2 columnas, segundo gráfico
plt.plot(td_dias, actividades_bq, 'o', label='Datos Experimentales (Actividad)')
plt.plot(td_fit, actividades_fit, '-', color='green', label=f'Ajuste Exponencial\n($T_{{1/2}}$ = {T_half_calculado:.3f} días)')
plt.title('Decaimiento Radiactivo: Actividad vs. Tiempo')
plt.xlabel('Tiempo de decaimiento ($t_d$) [días]')
plt.ylabel('Actividad [Bq]')
plt.yscale('linear') # Puedes cambiar a 'log' si quieres ver una línea recta aquí también
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()