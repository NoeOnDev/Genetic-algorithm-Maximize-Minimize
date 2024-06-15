import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from prettytable import PrettyTable
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext

def fitness_function_log_cos_x(x):
    return x*np.cos(x)

fitness_function = fitness_function_log_cos_x

def calculate_bit_length(start_value, end_value, precision):
    return math.ceil(math.log2((end_value - start_value) / precision + 1))

def float_to_binary(value, min_value, max_value, bit_length):
    scaled_value = (value - min_value) / (max_value - min_value) * (2**bit_length - 1)
    return format(int(scaled_value), '0' + str(bit_length) + 'b')

def binary_to_float(binary_str, min_value, max_value, bit_length):
    int_value = int(binary_str, 2)
    return min_value + int_value * (max_value - min_value) / (2**bit_length - 1)

def evaluate_fitness(individual, maximize, min_value, max_value, bit_length):
    x = binary_to_float(individual, min_value, max_value, bit_length)
    f = fitness_function(x)
    return f if maximize else -f

def create_initial_population(count, min_value, max_value, bit_length):
    return [float_to_binary(random.uniform(min_value, max_value), min_value, max_value, bit_length) for _ in range(count)]

def select_pairs(population):
    pairs = []
    n = len(population)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((population[i], population[j]))
    return pairs

def crossover(pair, bit_length):
    crossover_point = random.randint(1, bit_length - 1)
    child1 = pair[0][:crossover_point] + pair[1][crossover_point:]
    child2 = pair[1][:crossover_point] + pair[0][crossover_point:]
    return child1, child2

def mutate(individual, mutation_prob_gene, mutation_prob_individual, bit_length):
    if random.random() < mutation_prob_individual:
        individual = list(individual)
        for i in range(bit_length):
            if random.random() < mutation_prob_gene:
                individual[i] = '1' if individual[i] == '0' else '0'
        return ''.join(individual)
    return individual

def prune(population, max_population, min_value, max_value, maximize, bit_length):
    unique_population = list(set(population))
    unique_population.sort(key=lambda ind: evaluate_fitness(ind, maximize, min_value, max_value, bit_length), reverse=maximize)
    if len(unique_population) > max_population:
        best_individual = unique_population[0]
        to_keep = random.sample(unique_population[1:], max_population - 1)
        to_keep.append(best_individual)
        unique_population = to_keep
    statistics = {
        "max": evaluate_fitness(unique_population[0], maximize, min_value, max_value, bit_length),
        "min": evaluate_fitness(unique_population[-1], maximize, min_value, max_value, bit_length),
        "average": sum(evaluate_fitness(ind, maximize, min_value, max_value, bit_length) for ind in unique_population) / len(unique_population)
    }
    return unique_population, statistics

def plot_function_with_individuals(x_values, y_values, individuals, best, worst, generation, folder, min_value, max_value, maximize, bit_length):
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label=f'f(x) = {fitness_function.__name__}')

    x_individuals = [binary_to_float(ind, min_value, max_value, bit_length) for ind in individuals]
    y_individuals = [fitness_function(x) for x in x_individuals]

    plt.scatter(x_individuals, y_individuals, color='blue', label='Individuos', alpha=0.6)

    best_x = binary_to_float(best, min_value, max_value, bit_length)
    best_y = fitness_function(best_x)
    worst_x = binary_to_float(worst, min_value, max_value, bit_length)
    worst_y = fitness_function(worst_x)

    if maximize:
        plt.scatter([best_x], [best_y], color='green', label='Mejor Individuo', s=100, edgecolor='black')
        plt.scatter([worst_x], [worst_y], color='red', label='Peor Individuo', s=100, edgecolor='black')
    else:
        plt.scatter([best_x], [best_y], color='red', label='Peor Individuo', s=100, edgecolor='black')
        plt.scatter([worst_x], [worst_y], color='green', label='Mejor Individuo', s=100, edgecolor='black')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Función y Individuos - Generación {generation}')
    plt.legend()
    plt.grid(True)

    plt.xlim(min_value, max_value)
    plt.ylim(min(y_values), max(y_values))

    plot_name = f"Generation_{generation}.png"
    plt.savefig(os.path.join(folder, plot_name))
    plt.close()

def plot_evolution(best_fitnesses, worst_fitnesses, average_fitnesses, folder, maximize):
    plt.figure(figsize=(10, 5))
    
    plt.plot(best_fitnesses, label='Mejor Aptitud', color='green')
    plt.plot(worst_fitnesses, label='Peor Aptitud', color='red')
    plt.plot(average_fitnesses, label='Aptitud Media', color='blue')

    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    if maximize:
        plt.title('Evolución de la Maximización de Aptitudes')
    else:
        plt.title('Evolución de la Minimización de Aptitudes')

    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(folder, 'Evolution_Fitness.png'))
    plt.close()

def create_video(folder, generations_count):
    image_folder = folder
    video_name = 'GeneticAlgorithmVideo.avi'

    images = [f"Generation_{i}.png" for i in range(0, generations_count + 1)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def validate_entries():
    try:
        start_value = float(entry_start_value.get())
        end_value = float(entry_end_value.get())
        precision = float(entry_precision.get())
        generations_count = int(entry_generations_count.get())
        mutation_prob_gene = float(entry_mutation_prob_gene.get())
        mutation_prob_individual = float(entry_mutation_prob_individual.get())
        individuals_count = int(entry_individuals_count.get())
        max_population = int(entry_max_population.get())
        
        if end_value < start_value:
            messagebox.showerror("Error de Validación", "El valor final no puede ser menor que el valor inicial.")
            return False
        if not (0 < precision <= 1):
            messagebox.showerror("Error de Validación", "Delta X debe estar entre 0 y 1.")
            return False
        if not (0 <= mutation_prob_gene <= 1):
            messagebox.showerror("Error de Validación", "La probabilidad de mutación del gen debe estar entre 0 y 1.")
            return False
        if not (0 <= mutation_prob_individual <= 1):
            messagebox.showerror("Error de Validación", "La probabilidad de mutación del individuo debe estar entre 0 y 1.")
            return False
        if generations_count <= 0 or individuals_count <= 0 or max_population <= 0:
            messagebox.showerror("Error de Validación", "El número de generaciones, individuos y población máxima deben ser números enteros positivos.")
            return False

        return True
    except ValueError:
        messagebox.showerror("Error de Validación", "Por favor, ingrese valores válidos en todos los campos.")
        return False

def run_genetic_algorithm():
    if not validate_entries():
        return

    start_value = float(entry_start_value.get())
    end_value = float(entry_end_value.get())
    precision = float(entry_precision.get())
    generations_count = int(entry_generations_count.get())
    maximize = var_maximize.get() == 1
    mutation_prob_gene = float(entry_mutation_prob_gene.get())
    mutation_prob_individual = float(entry_mutation_prob_individual.get())
    individuals_count = int(entry_individuals_count.get())
    max_population = int(entry_max_population.get())

    bit_length = calculate_bit_length(start_value, end_value, precision)
    min_value = start_value
    max_value = end_value

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    x_values = np.linspace(min_value, max_value, 400)
    y_values = [fitness_function(x) for x in x_values]

    population = create_initial_population(individuals_count, min_value, max_value, bit_length)
    best_fitnesses = []
    worst_fitnesses = []
    average_fitnesses = []

    results_text.delete('1.0', tk.END)

    for generation in range(generations_count + 1):
        fitnesses = [evaluate_fitness(ind, maximize, min_value, max_value, bit_length) for ind in population]
        best_fitness = max(fitnesses) if maximize else min(fitnesses)
        worst_fitness = min(fitnesses) if maximize else max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)

        best_fitnesses.append(best_fitness)
        worst_fitnesses.append(worst_fitness)
        average_fitnesses.append(average_fitness)

        best_individual = population[fitnesses.index(best_fitness)]
        best_x_value = binary_to_float(best_individual, min_value, max_value, bit_length)
        worst_individual = population[fitnesses.index(worst_fitness)]

        # Crear una tabla con los resultados de la generación actual
        table = PrettyTable()
        table.field_names = ["Generación", "Cadena de Bits", "Índice", "Valor de x", "Valor de Aptitud"]
        table.add_row([generation, best_individual, fitnesses.index(best_fitness), round(best_x_value, 3), round(best_fitness, 3)])
        results_text.insert(tk.END, table.get_string() + "\n")

        plot_function_with_individuals(x_values, y_values, population, best_individual, worst_individual, generation, plots_folder, min_value, max_value, maximize, bit_length)
        
        if generation < generations_count:
            pairs = select_pairs(population)
            new_population = []

            for pair in pairs:
                if random.random() < random.random():
                    offspring = crossover(pair, bit_length)
                    new_population.extend(offspring)
                else:
                    new_population.extend(pair)

            new_population = [mutate(ind, mutation_prob_gene, mutation_prob_individual, bit_length) for ind in new_population]
            population = [ind for ind in new_population if min_value <= binary_to_float(ind, min_value, max_value, bit_length) <= max_value]
            population, stats = prune(population, max_population, min_value, max_value, maximize, bit_length)
            population.append(best_individual)

    population, stats = prune(population, max_population, min_value, max_value, maximize, bit_length)

    plot_evolution(best_fitnesses, worst_fitnesses, average_fitnesses, plots_folder, maximize)
    create_video(plots_folder, generations_count)

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Algoritmo Genético")

tk.Label(root, text="Valor Inicial:").grid(row=0, column=0, sticky=tk.W)
entry_start_value = tk.Entry(root)
entry_start_value.grid(row=0, column=1)

tk.Label(root, text="Valor Final:").grid(row=1, column=0, sticky=tk.W)
entry_end_value = tk.Entry(root)
entry_end_value.grid(row=1, column=1)

tk.Label(root, text="Delta X:").grid(row=2, column=0, sticky=tk.W)
entry_precision = tk.Entry(root)
entry_precision.grid(row=2, column=1)

tk.Label(root, text="Número de Generaciones:").grid(row=3, column=0, sticky=tk.W)
entry_generations_count = tk.Entry(root)
entry_generations_count.grid(row=3, column=1)

tk.Label(root, text="Probabilidad de Mutación del Gen:").grid(row=4, column=0, sticky=tk.W)
entry_mutation_prob_gene = tk.Entry(root)
entry_mutation_prob_gene.grid(row=4, column=1)

tk.Label(root, text="Probabilidad de Mutación del Individuo:").grid(row=5, column=0, sticky=tk.W)
entry_mutation_prob_individual = tk.Entry(root)
entry_mutation_prob_individual.grid(row=5, column=1)

tk.Label(root, text="Número de Individuos:").grid(row=6, column=0, sticky=tk.W)
entry_individuals_count = tk.Entry(root)
entry_individuals_count.grid(row=6, column=1)

tk.Label(root, text="Población Máxima:").grid(row=7, column=0, sticky=tk.W)
entry_max_population = tk.Entry(root)
entry_max_population.grid(row=7, column=1)

tk.Label(root, text="Maximizar Función:").grid(row=8, column=0, sticky=tk.W)
var_maximize = tk.IntVar()
tk.Checkbutton(root, variable=var_maximize).grid(row=8, column=1, sticky=tk.W)

tk.Button(root, text="Ejecutar", command=run_genetic_algorithm).grid(row=9, column=0, columnspan=2)

results_text = scrolledtext.ScrolledText(root, width=80, height=20)
results_text.grid(row=10, column=0, columnspan=2)

root.mainloop()