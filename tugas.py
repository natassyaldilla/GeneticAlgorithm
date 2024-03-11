import random
import numpy as np  # Add this line to import NumPy
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import io

# Definisikan nilai awal
strLength = 5
ukuranPopulasi = 10
generasiMaks = 50
kCrossOv = 0.8
mutation_prob = 0.1

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()

# Data CSV yang diberikan dalam bentuk string
data = """
Jalan;1;2;3;4;5
1;0;3;4;2;7
2;3;0;4;6;3
3;4;4;0;5;8
4;2;6;5;0;6
5;7;3;8;6;0
"""

# Membaca data CSV ke dalam DataFrame
df = pd.read_csv(io.StringIO(data), delimiter=';')

# Menampilkan DataFrame
print(df)

# Encode the 'Jalan' column
encoded_cities = label_encoder.fit_transform(df['Jalan'].astype(str))

def fitness(chromo):
    return np.sum(chromo == 1)  # Count the occurrences of 1 (assuming labels are now integers)

def crossover(parent1, parent2):
    min_length = min(len(parent1), len(parent2))
    cross_point = random.randint(1, min_length - 1)

    child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]), axis=None)
    child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]), axis=None)

    return child1, child2

def mutasi(chromosome):
    mutation_point = random.randint(0, strLength - 1)
    mutated_chromosome = list(chromosome)
    if mutated_chromosome[mutation_point] == 1:
        mutated_chromosome[mutation_point] = 0
    else:
        mutated_chromosome[mutation_point] = 1
    return mutated_chromosome

# Inisialisasi populasi menggunakan LabelEncoder
populasi = []
for i in range(ukuranPopulasi):
    chromosome = label_encoder.transform(random.sample(df['Jalan'].astype(str).tolist(), strLength))
    populasi.append(chromosome)

for generation in range(generasiMaks):
    fitness_scores = [fitness(chromosome) for chromosome in populasi]

    chromosomeTerb = populasi[fitness_scores.index(max(fitness_scores))]
    best_fitness = max(fitness_scores)

    print(f"Generasi {generation}: Chromosome Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}")

    new_populasi = []
    while len(new_populasi) < ukuranPopulasi:
        parent1 = random.choices(populasi, weights=fitness_scores)[0]
        parent2 = random.choices(populasi, weights=fitness_scores)[0]
        if random.random() < kCrossOv:
            child1, child2 = crossover(parent1, parent2)
            new_populasi.append(child1)
            new_populasi.append(child2)
        else:
            new_populasi.append(parent1)
            new_populasi.append(parent2)

    for i in range(ukuranPopulasi):
        if random.random() < mutation_prob:
            new_populasi[i] = mutasi(new_populasi[i])

    populasi = new_populasi

chromosomeTerb = populasi[fitness_scores.index(max(fitness_scores))]
best_fitness = max(fitness_scores)
print(f"\nHasil Akhir: Chromosome Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}")
