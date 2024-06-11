import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd

if not os.path.exists('graficas_generacion'):
    os.makedirs('graficas_generacion')
if not os.path.exists('graficas_estadisticas'):
    os.makedirs('graficas_estadisticas')

def funcion_objetivo(x):
    return x * np.cos(x)

def binario_a_decimal(cadena_binaria, x_min, x_max):
    decimal = int(cadena_binaria, 2)
    valor_max = 2**len(cadena_binaria) - 1
    return x_min + (decimal / valor_max) * (x_max - x_min)

def decimal_a_binario(valor, x_min, x_max, precision):
    valor_max = 2**precision - 1
    normalizado = int((valor - x_min) / (x_max - x_min) * valor_max)
    return f'{normalizado:0{precision}b}'

def calcular_longitud_cromosoma(x_min, x_max, precision):
    rango = x_max - x_min
    valores_discretos = int(rango * precision)
    longitud_cromosoma = valores_discretos.bit_length()
    return longitud_cromosoma

def inicializar_poblacion(tamano, x_min, x_max, precision):
    poblacion = []
    longitud_cromosoma = calcular_longitud_cromosoma(x_min, x_max, precision)
    for _ in range(tamano):
        valor = random.uniform(x_min, x_max)
        cadena_binaria = decimal_a_binario(valor, x_min, x_max, longitud_cromosoma)
        poblacion.append(cadena_binaria)
    return poblacion

def evaluar_poblacion(poblacion, x_min, x_max):
    return np.array([funcion_objetivo(binario_a_decimal(ind, x_min, x_max)) for ind in poblacion])

# Seleccionar parejas de individuos (Estrategia A1)
def formar_parejas(poblacion):
    parejas = []
    n = len(poblacion)
    for individuo in poblacion:
        m = random.randint(1, n)
        companeros = random.sample(list(poblacion), m)
        if individuo in companeros:
            companeros.remove(individuo)
        parejas.append((individuo, companeros))
    return parejas

# Cruza de información (Estrategia C2)
def cruzar(padres):
    padre1, padre2 = padres
    num_puntos = random.randint(1, len(padre1) - 1)
    puntos_cruza = sorted(random.sample(range(1, len(padre1)), num_puntos))
    
    descendiente = list(padre1)
    for i in range(len(puntos_cruza)):
        if i % 2 == 0:
            descendiente[puntos_cruza[i]:] = padre2[puntos_cruza[i]:]
        else:
            descendiente[:puntos_cruza[i]] = padre2[:puntos_cruza[i]]
    
    return ''.join(descendiente)

# Mutar los individuos descendientes (Estrategia M2)
def mutar(individuo, prob_mutacion_individuo, prob_mutacion_gen):
    if random.random() < prob_mutacion_individuo:
        individuo = list(individuo)
        for i in range(len(individuo)):
            if random.random() < prob_mutacion_gen:
                j = random.randint(0, len(individuo) - 1)
                individuo[i], individuo[j] = individuo[j], individuo[i]
        individuo = ''.join(individuo)
    return individuo

# Poda (Estrategia P2)
def podar_poblacion(poblacion, aptitud, tamano, maximizar):
    poblacion_unica, indices_unicos = np.unique(poblacion, return_index=True)
    aptitud_unica = aptitud[indices_unicos]
    
    if len(poblacion_unica) > tamano:
        if maximizar:
            indices_ordenados = np.argsort(-aptitud_unica)
        else:
            indices_ordenados = np.argsort(aptitud_unica)
        poblacion_podada = poblacion_unica[indices_ordenados][:tamano]
        aptitud_podada = aptitud_unica[indices_ordenados][:tamano]
    else:
        poblacion_podada = poblacion_unica
        aptitud_podada = aptitud_unica
    
    return poblacion_podada, aptitud_podada

def algoritmo_genetico(tamano_poblacion, tamano_maximo_poblacion, generaciones, prob_mutacion_individuo, prob_mutacion_gen, x_min, x_max, precision, maximizar):
    poblacion = inicializar_poblacion(tamano_poblacion, x_min, x_max, precision)
    aptitud = evaluar_poblacion(poblacion, x_min, x_max)
    
    mejores_aptitudes = []
    aptitudes_promedio = []
    peores_aptitudes = []
    info_mejores_individuos = []
    
    for generacion in range(generaciones):
        nueva_poblacion = []
        parejas = formar_parejas(poblacion)
        
        for individuo, companeros in parejas:
            for companero in companeros:
                descendiente = cruzar([individuo, companero])
                descendiente = mutar(descendiente, prob_mutacion_individuo, prob_mutacion_gen)
                nueva_poblacion.append(descendiente)
        
        nueva_poblacion = np.array(nueva_poblacion)
        nueva_aptitud = evaluar_poblacion(nueva_poblacion, x_min, x_max)
        
        poblacion_combinada = np.concatenate((poblacion, nueva_poblacion))
        aptitud_combinada = np.concatenate((aptitud, nueva_aptitud))
        
        # Guardar estadísticos de la población
        if maximizar:
            indice_mejor = np.argmax(aptitud_combinada)
            mejor_aptitud = np.max(aptitud_combinada)
            peor_aptitud = np.min(aptitud_combinada)
        else:
            indice_mejor = np.argmin(aptitud_combinada)
            mejor_aptitud = np.min(aptitud_combinada)
            peor_aptitud = np.max(aptitud_combinada)
        
        aptitud_promedio = np.mean(aptitud_combinada)
        
        mejor_individuo = poblacion_combinada[indice_mejor]
        mejor_valor = binario_a_decimal(mejor_individuo, x_min, x_max)
        
        print(f'Generación {generacion}: Mejor Individuo = {mejor_valor} Aptitud = {mejor_aptitud}')
        
        # Guardar información del mejor individuo
        info_mejores_individuos.append({
            'Generación': generacion,
            'Individuo': mejor_individuo,
            'Valor del índice': indice_mejor,
            'Valor de x': mejor_valor,
            'Aptitud': mejor_aptitud
        })
        
        mejores_aptitudes.append(mejor_aptitud)
        aptitudes_promedio.append(aptitud_promedio)
        peores_aptitudes.append(peor_aptitud)
        
        # Crear y guardar la gráfica de la generación
        valores = [binario_a_decimal(ind, x_min, x_max) for ind in poblacion_combinada]
        peor_valor = binario_a_decimal(poblacion_combinada[np.argmin(aptitud_combinada) if maximizar else np.argmax(aptitud_combinada)], x_min, x_max)

        valores_objetivo = [funcion_objetivo(val) for val in valores]

        plt.figure()
        plt.plot(np.linspace(x_min, x_max, 400), [funcion_objetivo(x) for x in np.linspace(x_min, x_max, 400)], label='f(x)')
        plt.scatter(valores, valores_objetivo, color='lightblue', label='Individuos', alpha=0.6)
        plt.scatter([mejor_valor], [funcion_objetivo(mejor_valor)], color='green', label='Mejor Individuo', zorder=5)
        plt.scatter([peor_valor], [funcion_objetivo(peor_valor)], color='red', label='Peor Individuo', zorder=5)
        plt.title(f'Generación {generacion}')
        plt.xlabel('X')
        plt.ylabel('f(x)')
        plt.legend()
        plt.savefig(f'graficas_generacion/generacion_{generacion}.png')
        plt.close()

        # Poda de la población
        poblacion, aptitud = podar_poblacion(poblacion_combinada, aptitud_combinada, min(tamano_poblacion, tamano_maximo_poblacion), maximizar)
    
    # Crear y guardar la gráfica de los estadísticos
    plt.figure()
    plt.plot(mejores_aptitudes, label='Mejor Aptitud')
    plt.plot(aptitudes_promedio, label='Promedio de Aptitud')
    plt.plot(peores_aptitudes, label='Peor Aptitud')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.legend()
    plt.title('Estadísticos por Generación')
    plt.savefig('graficas_estadisticas/estadisticas.png')
    plt.close()
    
    # Guardar la información del mejor individuo en un archivo CSV
    df_mejores_individuos = pd.DataFrame(info_mejores_individuos)
    df_mejores_individuos.to_csv('mejores_individuos.csv', index=False)

# Parámetros del algoritmo
tamano_poblacion = 100
tamano_maximo_poblacion = 200
generaciones = 50
prob_mutacion_individuo = 0.2
prob_mutacion_gen = 0.1
x_min = 0
x_max = 10
precision = 100  # Precisión decimal
maximizar = True  # Cambia a True si deseas maximizar la función objetivo

algoritmo_genetico(tamano_poblacion, tamano_maximo_poblacion, generaciones, prob_mutacion_individuo, prob_mutacion_gen, x_min, x_max, precision, maximizar)
