import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from prettytable import PrettyTable
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext

def funcion_aptitud_log_cos_x(x):
    return x * np.cos(x)

funcion_aptitud = funcion_aptitud_log_cos_x

def calcular_longitud_bits(valor_inicial, valor_final, precision):
    return math.ceil(math.log2((valor_final - valor_inicial) / precision + 1))

def flotante_a_binario(valor, valor_min, valor_max, longitud_bits):
    valor_escalado = (valor - valor_min) / (valor_max - valor_min) * (2**longitud_bits - 1)
    return format(int(valor_escalado), '0' + str(longitud_bits) + 'b')

def binario_a_flotante(cadena_binaria, valor_min, valor_max, longitud_bits):
    valor_entero = int(cadena_binaria, 2)
    return valor_min + valor_entero * (valor_max - valor_min) / (2**longitud_bits - 1)

def evaluar_aptitud(individuo, maximizar, valor_min, valor_max, longitud_bits):
    x = binario_a_flotante(individuo, valor_min, valor_max, longitud_bits)
    f = funcion_aptitud(x)
    return f if maximizar else -f

def crear_poblacion_inicial(cantidad, valor_min, valor_max, longitud_bits):
    return [flotante_a_binario(random.uniform(valor_min, valor_max), valor_min, valor_max, longitud_bits) for _ in range(cantidad)]

def seleccionar_pares(poblacion):
    pares = []
    n = len(poblacion)
    for i in range(n):
        for j in range(i + 1, n):
            pares.append((poblacion[i], poblacion[j]))
    return pares

def cruzar(par, longitud_bits):
    punto_cruce = random.randint(1, longitud_bits - 1)
    hijo1 = par[0][:punto_cruce] + par[1][punto_cruce:]
    hijo2 = par[1][:punto_cruce] + par[0][punto_cruce:]
    return hijo1, hijo2

def mutar(individuo, prob_mutacion_gen, prob_mutacion_individuo, longitud_bits):
    if random.random() < prob_mutacion_individuo:
        individuo = list(individuo)
        for i in range(longitud_bits):
            if random.random() < prob_mutacion_gen:
                individuo[i] = '1' if individuo[i] == '0' else '0'
        return ''.join(individuo)
    return individuo

def podar(poblacion, max_poblacion, valor_min, valor_max, maximizar, longitud_bits):
    poblacion_unica = list(set(poblacion))
    poblacion_unica.sort(key=lambda ind: evaluar_aptitud(ind, maximizar, valor_min, valor_max, longitud_bits), reverse=maximizar)
    if len(poblacion_unica) > max_poblacion:
        mejor_individuo = poblacion_unica[0]
        a_mantener = random.sample(poblacion_unica[1:], max_poblacion - 1)
        a_mantener.append(mejor_individuo)
        poblacion_unica = a_mantener
    estadisticas = {
        "max": evaluar_aptitud(poblacion_unica[0], maximizar, valor_min, valor_max, longitud_bits),
        "min": evaluar_aptitud(poblacion_unica[-1], maximizar, valor_min, valor_max, longitud_bits),
        "media": sum(evaluar_aptitud(ind, maximizar, valor_min, valor_max, longitud_bits) for ind in poblacion_unica) / len(poblacion_unica)
    }
    return poblacion_unica, estadisticas

def graficar_funcion_con_individuos(x_valores, y_valores, individuos, mejor, peor, generacion, carpeta, valor_min, valor_max, maximizar, longitud_bits):
    plt.figure(figsize=(10, 5))
    plt.plot(x_valores, y_valores, label=f'f(x) = {funcion_aptitud.__name__}')

    x_individuos = [binario_a_flotante(ind, valor_min, valor_max, longitud_bits) for ind in individuos]
    y_individuos = [funcion_aptitud(x) for x in x_individuos]

    plt.scatter(x_individuos, y_individuos, color='blue', label='Individuos', alpha=0.6)

    mejor_x = binario_a_flotante(mejor, valor_min, valor_max, longitud_bits)
    mejor_y = funcion_aptitud(mejor_x)
    peor_x = binario_a_flotante(peor, valor_min, valor_max, longitud_bits)
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

    plt.xlim(valor_min, valor_max)
    plt.ylim(min(y_valores), max(y_valores))

    nombre_grafica = f"Generacion_{generacion}.png"
    plt.savefig(os.path.join(carpeta, nombre_grafica))
    plt.close()

def graficar_evolucion(mejores_aptitudes, peores_aptitudes, medias_aptitudes, carpeta, maximizar):
    plt.figure(figsize=(10, 5))
    
    plt.plot(mejores_aptitudes, label='Mejor Aptitud', color='green')
    plt.plot(peores_aptitudes, label='Peor Aptitud', color='red')
    plt.plot(medias_aptitudes, label='Aptitud Media', color='blue')

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

def crear_video(carpeta, cantidad_generaciones):
    carpeta_imagenes = carpeta
    nombre_video = 'VideoAlgoritmoGenetico.avi'

    imagenes = [f"Generacion_{i}.png" for i in range(0, cantidad_generaciones + 1)]
    cuadro = cv2.imread(os.path.join(carpeta_imagenes, imagenes[0]))
    altura, ancho, capas = cuadro.shape

    video = cv2.VideoWriter(
        nombre_video, cv2.VideoWriter_fourcc(*'DIVX'), 1, (ancho, altura))

    for imagen in imagenes:
        video.write(cv2.imread(os.path.join(carpeta_imagenes, imagen)))

    cv2.destroyAllWindows()
    video.release()

def validar_entradas():
    try:
        limite_inferior = float(entry_limite_inferior.get())
        limite_superior = float(entry_limite_superior.get())
        precision = float(entry_resolucion.get())
        cantidad_generaciones = int(entry_cantidad_generaciones.get())
        prob_mutacion_gen = float(entry_prob_mutacion_gen.get())
        prob_mutacion_individuo = float(entry_prob_mutacion_individuo.get())
        cantidad_individuos = int(entry_cantidad_individuos.get())
        max_poblacion = int(entry_max_poblacion.get())
        
        if limite_superior < limite_inferior:
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
        if cantidad_generaciones <= 0 or cantidad_individuos <= 0 or max_poblacion <= 0:
            messagebox.showerror("Error de Validación", "El número de generaciones, individuos y población máxima deben ser números enteros positivos.")
            return False

        return True
    except ValueError:
        messagebox.showerror("Error de Validación", "Por favor, ingrese valores válidos en todos los campos.")
        return False

def ejecutar_algoritmo_genetico():
    if not validar_entradas():
        return

    limite_inferior = float(entry_limite_inferior.get())
    limite_superior = float(entry_limite_superior.get())
    resolucion = float(entry_resolucion.get())
    cantidad_generaciones = int(entry_cantidad_generaciones.get())
    maximizar = var_maximizar.get() == 1
    prob_mutacion_gen = float(entry_prob_mutacion_gen.get())
    prob_mutacion_individuo = float(entry_prob_mutacion_individuo.get())
    cantidad_individuos = int(entry_cantidad_individuos.get())
    max_poblacion = int(entry_max_poblacion.get())

    longitud_bits = calcular_longitud_bits(limite_inferior, limite_superior, resolucion)
    limite_inferior_x = limite_inferior
    limite_superior_x = limite_superior

    carpeta_graficas_generacion = "graficas_generacion"
    if not os.path.exists(carpeta_graficas_generacion):
        os.makedirs(carpeta_graficas_generacion)

    x_valores = np.linspace(limite_inferior_x, limite_superior_x, 400)
    y_valores = [funcion_aptitud(x) for x in x_valores]

    poblacion = crear_poblacion_inicial(cantidad_individuos, limite_inferior_x, limite_superior_x, longitud_bits)
    mejores_aptitudes = []
    peores_aptitudes = []
    medias_aptitudes = []

    resultados_texto.delete('1.0', tk.END)

    for generacion in range(cantidad_generaciones + 1):
        aptitudes = [evaluar_aptitud(ind, maximizar, limite_inferior_x, limite_superior_x, longitud_bits) for ind in poblacion]
        mejor_aptitud = max(aptitudes) if maximizar else min(aptitudes)
        peor_aptitud = min(aptitudes) if maximizar else max(aptitudes)
        media_aptitud = sum(aptitudes) / len(aptitudes)

        mejores_aptitudes.append(mejor_aptitud)
        peores_aptitudes.append(peor_aptitud)
        medias_aptitudes.append(media_aptitud)

        mejor_individuo = poblacion[aptitudes.index(mejor_aptitud)]
        mejor_x_valor = binario_a_flotante(mejor_individuo, limite_inferior_x, limite_superior_x, longitud_bits)
        peor_individuo = poblacion[aptitudes.index(peor_aptitud)]

        tabla = PrettyTable()
        tabla.field_names = ["Generación", "Cadena de Bits", "Índice", "Valor de x", "Valor de Aptitud"]
        tabla.add_row([generacion, mejor_individuo, aptitudes.index(mejor_aptitud), round(mejor_x_valor, 3), round(mejor_aptitud, 3)])
        resultados_texto.insert(tk.END, tabla.get_string() + "\n")

        graficar_funcion_con_individuos(x_valores, y_valores, poblacion, mejor_individuo, peor_individuo, generacion, carpeta_graficas_generacion, limite_inferior_x, limite_superior_x, maximizar, longitud_bits)
        
        if generacion < cantidad_generaciones:
            pares = seleccionar_pares(poblacion)
            nueva_poblacion = []

            for par in pares:
                if random.random() < random.random():
                    descendencia = cruzar(par, longitud_bits)
                    nueva_poblacion.extend(descendencia)
                else:
                    nueva_poblacion.extend(par)

            nueva_poblacion = [mutar(ind, prob_mutacion_gen, prob_mutacion_individuo, longitud_bits) for ind in nueva_poblacion]
            poblacion = [ind for ind in nueva_poblacion if limite_inferior_x <= binario_a_flotante(ind, limite_inferior_x, limite_superior_x, longitud_bits) <= limite_superior_x]
            poblacion, estadisticas = podar(poblacion, max_poblacion, limite_inferior_x, limite_superior_x, maximizar, longitud_bits)
            poblacion.append(mejor_individuo)

    poblacion, estadisticas = podar(poblacion, max_poblacion, limite_inferior_x, limite_superior_x, maximizar, longitud_bits)

    graficar_evolucion(mejores_aptitudes, peores_aptitudes, medias_aptitudes, carpeta_graficas_generacion, maximizar)
    crear_video(carpeta_graficas_generacion, cantidad_generaciones)

root = tk.Tk()
root.title("Algoritmo Genético")

tk.Label(root, text="Número de Individuos:").grid(row=0, column=0, sticky=tk.W)
entry_cantidad_individuos = tk.Entry(root)
entry_cantidad_individuos.grid(row=0, column=1)

tk.Label(root, text="Población Máxima:").grid(row=1, column=0, sticky=tk.W)
entry_max_poblacion = tk.Entry(root)
entry_max_poblacion.grid(row=1, column=1)

tk.Label(root, text="Limite Inferior de X:").grid(row=2, column=0, sticky=tk.W)
entry_limite_inferior = tk.Entry(root)
entry_limite_inferior.grid(row=2, column=1)

tk.Label(root, text="Limite Superior de X:").grid(row=3, column=0, sticky=tk.W)
entry_limite_superior = tk.Entry(root)
entry_limite_superior.grid(row=3, column=1)

tk.Label(root, text="Delta X:").grid(row=4, column=0, sticky=tk.W)
entry_resolucion = tk.Entry(root)
entry_resolucion.grid(row=4, column=1)

tk.Label(root, text="Probabilidad de Mutación del Individuo:").grid(row=5, column=0, sticky=tk.W)
entry_prob_mutacion_individuo = tk.Entry(root)
entry_prob_mutacion_individuo.grid(row=5, column=1)

tk.Label(root, text="Probabilidad de Mutación del Gen:").grid(row=6, column=0, sticky=tk.W)
entry_prob_mutacion_gen = tk.Entry(root)
entry_prob_mutacion_gen.grid(row=6, column=1)

tk.Label(root, text="Maximizar Función:").grid(row=7, column=0, sticky=tk.W)
var_maximizar = tk.IntVar()
tk.Checkbutton(root, variable=var_maximizar).grid(row=7, column=1, sticky=tk.W)

tk.Label(root, text="Número de Generaciones:").grid(row=8, column=0, sticky=tk.W)
entry_cantidad_generaciones = tk.Entry(root)
entry_cantidad_generaciones.grid(row=8, column=1)

tk.Button(root, text="Ejecutar", command=ejecutar_algoritmo_genetico).grid(row=9, column=0, columnspan=2)

resultados_texto = scrolledtext.ScrolledText(root, width=80, height=20)
resultados_texto.grid(row=10, column=0, columnspan=2)

root.mainloop()
