import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import tkinter as tk
from tkinter import messagebox, scrolledtext
from video import crear_video

# Definir la función de aptitud
def funcion_aptitud_log_cos_x(x):
    return np.log(1 + np.abs(x**2)) + 2 * np.cos(x)

funcion_aptitud = funcion_aptitud_log_cos_x

def calcular_longitud_bits(valor_inicio, valor_fin, precision):
    return math.ceil(math.log2((valor_fin - valor_inicio) / precision + 1))

def flotante_a_binario(valor, valor_minimo, valor_maximo, longitud_bits):
    valor_escalado = (valor - valor_minimo) / (valor_maximo - valor_minimo) * (2 ** longitud_bits - 1)
    return format(int(valor_escalado), '0' + str(longitud_bits) + 'b')

def binario_a_flotante(cadena_binaria, valor_minimo, valor_maximo, longitud_bits):
    valor_entero = int(cadena_binaria, 2)
    return valor_minimo + valor_entero * (valor_maximo - valor_minimo) / (2 ** longitud_bits - 1)

def evaluar_aptitud(individuo, maximizar, valor_minimo, valor_maximo, longitud_bits):
    x = binario_a_flotante(individuo, valor_minimo, valor_maximo, longitud_bits)
    f = funcion_aptitud(x)
    return f if maximizar else -f

def crear_poblacion_inicial(cantidad, valor_minimo, valor_maximo, longitud_bits):
    return [flotante_a_binario(random.uniform(valor_minimo, valor_maximo), valor_minimo, valor_maximo, longitud_bits) for _ in range(cantidad)]

# Estrategia A1: Formación de parejas
def seleccionar_pares(poblacion, n):
    pares = []
    for i in range(len(poblacion)):
        m = random.randint(0, n)
        # Selecciono m individuos aleatorios para formar parejas
        indices_cruce = set()
        intentos = 0
        while len(indices_cruce) < m and intentos < 10 * len(poblacion):
            j = random.randint(0, len(poblacion) - 1)
            if j != i:
                indices_cruce.add(j)
            intentos += 1

        for j in indices_cruce:
            pares.append((poblacion[i], poblacion[j]))
    return pares

# Estrategia C2: Múltiples puntos de cruza
def cruzar(par, longitud_bits):
    # Elijo aleatoriamente la cantidad de puntos de cruza
    n_puntos = random.randint(1, longitud_bits - 1)
    puntos_cruce = sorted(random.sample(range(1, longitud_bits), n_puntos))
    
    if len(puntos_cruce) % 2 != 0:
        puntos_cruce.append(longitud_bits)
        
    hijo1, hijo2 = list(par[0]), list(par[1])
    
    # Intercambio de segmentos de bits
    for i in range(0, len(puntos_cruce), 2):
        start, end = puntos_cruce[i], puntos_cruce[i+1]
        hijo1[start:end], hijo2[start:end] = hijo2[start:end], hijo1[start:end]
    
    return ''.join(hijo1), ''.join(hijo2)

# Estrategia M2: Intercambio de posición de bits
def mutar_gen(individuo, prob_mutacion_gen, longitud_bits):
    individuo = list(individuo)
    
    # Intercambio de posición de bits
    for i in range(longitud_bits):
        if random.random() < prob_mutacion_gen:
            pos1 = random.randint(0, longitud_bits - 1)
            pos2 = random.randint(0, longitud_bits - 1)
            individuo[pos1], individuo[pos2] = individuo[pos2], individuo[pos1]
    return ''.join(individuo)

def mutar(individuo, prob_mutacion_gen, prob_mutacion_individuo, longitud_bits):
    if random.random() < prob_mutacion_individuo:
        return mutar_gen(individuo, prob_mutacion_gen, longitud_bits)
    return individuo

# Estrategia P2: Eliminación aleatoria asegurando mantener al mejor individuo
def podar(poblacion, max_poblacion, valor_minimo, valor_maximo, maximizar, longitud_bits):
    poblacion_unica = list(set(poblacion))
    poblacion_unica.sort(key=lambda ind: evaluar_aptitud(ind, maximizar, valor_minimo, valor_maximo, longitud_bits), reverse=maximizar)
    mejor_individuo = poblacion_unica[0]
    
    # Elimino los individuos que sobran, manteniendo al mejor individuo de la generación
    if len(poblacion_unica) > max_poblacion:
        num_a_mantener = max_poblacion - 1
        a_mantener = random.sample(poblacion_unica[1:], num_a_mantener)
        a_mantener.append(mejor_individuo)
        poblacion_unica = a_mantener
    
    if mejor_individuo not in poblacion_unica:
        poblacion_unica.append(mejor_individuo)
    
    estadisticas = {
        "max": evaluar_aptitud(mejor_individuo, maximizar, valor_minimo, valor_maximo, longitud_bits),
        "min": evaluar_aptitud(poblacion_unica[-1], maximizar, valor_minimo, valor_maximo, longitud_bits),
        "promedio": sum(evaluar_aptitud(ind, maximizar, valor_minimo, valor_maximo, longitud_bits) for ind in poblacion_unica) / len(poblacion_unica)
    }
    return poblacion_unica, estadisticas

def ejecutar_algoritmo_genetico():
    if not validar_entradas():
        return

    # Obtengo los valores de las entradas
    valor_inicio = float(entrada_valor_inicio.get())
    valor_fin = float(entrada_valor_fin.get())
    precision = float(entrada_precision.get())
    numero_generaciones = int(entrada_numero_generaciones.get())
    maximizar = var_maximizar.get() == 1
    prob_mutacion_gen = float(entrada_prob_mutacion_gen.get())
    prob_mutacion_individuo = float(entrada_prob_mutacion_individuo.get())
    cantidad_individuos = int(entrada_cantidad_individuos.get())
    max_poblacion = int(entrada_max_poblacion.get())

    # Calculo la longitud de bits necesaria para representar los valores
    longitud_bits = calcular_longitud_bits(valor_inicio, valor_fin, precision)
    valor_minimo = valor_inicio
    valor_maximo = valor_fin

    carpeta_graficas_generacion = "mejores_y_peores_por_generacion"
    carpeta_grafica_evolucion = "grafica_de_evolucion"
    carpeta_video = "video_de_evolucion"

    if not os.path.exists(carpeta_graficas_generacion):
        os.makedirs(carpeta_graficas_generacion)
    if not os.path.exists(carpeta_grafica_evolucion):
        os.makedirs(carpeta_grafica_evolucion)
    if not os.path.exists(carpeta_video):
        os.makedirs(carpeta_video)

    # Creo los valores de "x" y "y" para la función de aptitud
    valores_x = np.linspace(valor_minimo, valor_maximo, 400)
    valores_y = [funcion_aptitud(x) for x in valores_x]
    
    # Creo la población inicial
    poblacion = crear_poblacion_inicial(cantidad_individuos, valor_minimo, valor_maximo, longitud_bits)
    mejores_aptitudes = []
    peores_aptitudes = []
    aptitudes_promedio = []

    texto_resultados.delete('1.0', tk.END)

    for generacion in range(1, numero_generaciones + 1):
        aptitudes = [evaluar_aptitud(ind, maximizar, valor_minimo, valor_maximo, longitud_bits) for ind in poblacion]
        
        mejor_aptitud = max(aptitudes) if maximizar else min(aptitudes)
        peor_aptitud = min(aptitudes) if maximizar else max(aptitudes)
        aptitud_promedio = sum(aptitudes) / len(aptitudes)
        
        # Guardo las mejores, peores y promedio de aptitudes por generación
        mejores_aptitudes.append(mejor_aptitud)
        peores_aptitudes.append(peor_aptitud)
        aptitudes_promedio.append(aptitud_promedio)

        # Encuentro al mejor y peor individuo de la generación
        mejor_individuo = poblacion[aptitudes.index(mejor_aptitud)]
        peor_individuo = poblacion[aptitudes.index(peor_aptitud)]
        
        # Convierto al mejor individuo de su valor binario a flotante para mostrarlo en la tabla
        mejor_valor_x = binario_a_flotante(mejor_individuo, valor_minimo, valor_maximo, longitud_bits)
        
        tabla = PrettyTable()
        tabla.field_names = ["Generación", "Cadena de Bits", "Índice", "Valor de x", "Valor de Aptitud"]
        tabla.add_row([generacion, mejor_individuo, aptitudes.index(mejor_aptitud), round(mejor_valor_x, 3), round(mejor_aptitud, 3)])
        texto_resultados.insert(tk.END, tabla.get_string() + "\n")

        graficar_funcion_con_individuos(valores_x, valores_y, poblacion, mejor_individuo, peor_individuo, generacion, carpeta_graficas_generacion, valor_minimo, valor_maximo, maximizar, longitud_bits)
        
        if generacion < numero_generaciones:
            pares = seleccionar_pares(poblacion, len(poblacion))
            nueva_poblacion = []

            for par in pares:
                if random.random() < random.random():
                    descendencia = cruzar(par, longitud_bits)
                    nueva_poblacion.extend(descendencia)
                else:
                    nueva_poblacion.extend(par)

            nueva_poblacion = [mutar(ind, prob_mutacion_gen, prob_mutacion_individuo, longitud_bits) for ind in nueva_poblacion]
            poblacion = [ind for ind in nueva_poblacion if valor_minimo <= binario_a_flotante(ind, valor_minimo, valor_maximo, longitud_bits) <= valor_maximo]
            poblacion, estadisticas = podar(poblacion, max_poblacion, valor_minimo, valor_maximo, maximizar, longitud_bits)
            poblacion.append(mejor_individuo)

    poblacion, estadisticas = podar(poblacion, max_poblacion, valor_minimo, valor_maximo, maximizar, longitud_bits)

    graficar_evolucion(mejores_aptitudes, peores_aptitudes, aptitudes_promedio, carpeta_grafica_evolucion, maximizar)
    crear_video(carpeta_graficas_generacion, numero_generaciones)


# Funciones para graficar

def graficar_funcion_con_individuos(valores_x, valores_y, individuos, mejor, peor, generacion, carpeta, valor_minimo, valor_maximo, maximizar, longitud_bits):
    plt.figure(figsize=(10, 5))
    plt.plot(valores_x, valores_y, color='blue', linewidth=2)

    x_individuos = [binario_a_flotante(ind, valor_minimo, valor_maximo, longitud_bits) for ind in individuos]
    y_individuos = [funcion_aptitud(x) for x in x_individuos]

    plt.scatter(x_individuos, y_individuos, color='lightblue', label='Individuos', s=90, edgecolor='lightblue')

    mejor_x = binario_a_flotante(mejor, valor_minimo, valor_maximo, longitud_bits)
    mejor_y = funcion_aptitud(mejor_x)
    peor_x = binario_a_flotante(peor, valor_minimo, valor_maximo, longitud_bits)
    peor_y = funcion_aptitud(peor_x)

    if maximizar:
        plt.scatter([mejor_x], [mejor_y], color='green', label='Mejor Individuo', s=100, edgecolor='black')
        plt.scatter([peor_x], [peor_y], color='red', label='Peor Individuo', s=100, edgecolor='black')
    else:
        plt.scatter([mejor_x], [mejor_y], color='red', label='Peor Individuo', s=100, edgecolor='black')
        plt.scatter([peor_x], [peor_y], color='green', label='Mejor Individuo', s=100, edgecolor='black')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Función y Individuos - Generación {generacion}')
    plt.legend()
    plt.grid(True)

    plt.xlim(valor_minimo, valor_maximo)
    plt.ylim(min(valores_y), max(valores_y))

    nombre_grafica = f"Generacion_{generacion}.png"
    plt.savefig(os.path.join(carpeta, nombre_grafica))
    plt.close()

def graficar_evolucion(mejores_aptitudes, peores_aptitudes, aptitudes_promedio, carpeta, maximizar):
    plt.figure(figsize=(10, 5))
    
    plt.plot(mejores_aptitudes, label='Mejor Aptitud', color='green')
    plt.plot(peores_aptitudes, label='Peor Aptitud', color='red')
    plt.plot(aptitudes_promedio, label='Aptitud Promedio', color='blue')

    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    if maximizar:
        plt.title('Evolución de la Maximización de Aptitudes')
    else:
        plt.title('Evolución de la Minimización de Aptitudes')

    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(carpeta, 'Evolucion_Aptitud.png'))
    plt.close()
    

# Interfaz Gráfica
def validar_entradas():
    try:
        valor_inicio = float(entrada_valor_inicio.get())
        valor_fin = float(entrada_valor_fin.get())
        precision = float(entrada_precision.get())
        numero_generaciones = int(entrada_numero_generaciones.get())
        prob_mutacion_gen = float(entrada_prob_mutacion_gen.get())
        prob_mutacion_individuo = float(entrada_prob_mutacion_individuo.get())
        cantidad_individuos = int(entrada_cantidad_individuos.get())
        max_poblacion = int(entrada_max_poblacion.get())
        
        if valor_fin < valor_inicio:
            messagebox.showerror("Error de Validación", "El valor final no puede ser menor que el valor inicial.")
            return False
        if not (0 < precision <= 1):
            messagebox.showerror("Error de Validación", "Delta X debe estar entre 0 y 1.")
            return False
        if not (0 <= prob_mutacion_gen <= 1):
            messagebox.showerror("Error de Validación", "La probabilidad de mutación del gen debe estar entre 0 y 1.")
            return False
        if not (0 <= prob_mutacion_individuo <= 1):
            messagebox.showerror("Error de Validación", "La probabilidad de mutación del individuo debe estar entre 0 y 1.")
            return False
        if numero_generaciones <= 0 or cantidad_individuos <= 0 or max_poblacion <= 0:
            messagebox.showerror("Error de Validación", "El número de generaciones, individuos y población máxima deben ser números enteros positivos.")
            return False

        return True
    except ValueError:
        messagebox.showerror("Error de Validación", "Por favor, ingrese valores válidos en todos los campos.")
        return False

root = tk.Tk()
root.title("Algoritmo Genético")

tk.Label(root, text="Límite Inferior:").grid(row=0, column=0, sticky=tk.W)
entrada_valor_inicio = tk.Entry(root)
entrada_valor_inicio.grid(row=0, column=1)

tk.Label(root, text="Límite Superior:").grid(row=1, column=0, sticky=tk.W)
entrada_valor_fin = tk.Entry(root)
entrada_valor_fin.grid(row=1, column=1)

tk.Label(root, text="Resolución:").grid(row=2, column=0, sticky=tk.W)
entrada_precision = tk.Entry(root)
entrada_precision.grid(row=2, column=1)

tk.Label(root, text="Número de Generaciones:").grid(row=3, column=0, sticky=tk.W)
entrada_numero_generaciones = tk.Entry(root)
entrada_numero_generaciones.grid(row=3, column=1)

tk.Label(root, text="Probabilidad de Mutación del Gen:").grid(row=4, column=0, sticky=tk.W)
entrada_prob_mutacion_gen = tk.Entry(root)
entrada_prob_mutacion_gen.grid(row=4, column=1)

tk.Label(root, text="Probabilidad de Mutación del Individuo:").grid(row=5, column=0, sticky=tk.W)
entrada_prob_mutacion_individuo = tk.Entry(root)
entrada_prob_mutacion_individuo.grid(row=5, column=1)

tk.Label(root, text="Población Inicial:").grid(row=6, column=0, sticky=tk.W)
entrada_cantidad_individuos = tk.Entry(root)
entrada_cantidad_individuos.grid(row=6, column=1)

tk.Label(root, text="Población Máxima:").grid(row=7, column=0, sticky=tk.W)
entrada_max_poblacion = tk.Entry(root)
entrada_max_poblacion.grid(row=7, column=1)

tk.Label(root, text="Maximizar Función:").grid(row=8, column=0, sticky=tk.W)
var_maximizar = tk.IntVar()
tk.Checkbutton(root, variable=var_maximizar).grid(row=8, column=1, sticky=tk.W)

tk.Button(root, text="Ejecutar", command=ejecutar_algoritmo_genetico).grid(row=9, column=0, columnspan=2)

texto_resultados = scrolledtext.ScrolledText(root, width=80, height=20)
texto_resultados.grid(row=10, column=0, columnspan=2)

root.mainloop()
