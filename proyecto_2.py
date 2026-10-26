# =====================================================
# PRACTICA 2 - ALGORITMOS AVANZADOS
#
# Ejercicio 1 - Parallel Speedup
# Ejercicio 2 - Greedy Knapsack
# Ejercicio 3 - TSP + 2-opt
# Ejercicio 4 - Simulated Annealing
# Ejercicio 5 - MST Approximation
# =====================================================

import multiprocessing as mp
import random
import time
import matplotlib.pyplot as plt
import math


# =====================================================
# EJERCICIO 1
# Parallel Speedup
# =====================================================

def estimate_pi(seed, samples=100_000):
    rng = random.Random(seed)

    inside = 0

    for _ in range(samples):
        x = rng.random()
        y = rng.random()

        if x**2 + y**2 <= 1:
            inside += 1

    return 4 * inside / samples


def ejercicio1():

    print("\n==========================")
    print("EJERCICIO 1")
    print("==========================")

    procesos = [1, 2, 4, 8]

    tiempos = []

    for p in procesos:

        seeds = list(range(p))

        inicio = time.perf_counter()

        with mp.Pool(processes=p) as pool:
            resultados = pool.map(estimate_pi, seeds)

        tiempo = time.perf_counter() - inicio

        tiempos.append(tiempo)

        promedio_pi = sum(resultados) / len(resultados)

        print(
            f"{p} procesos | "
            f"Tiempo = {tiempo:.3f} s | "
            f"Pi ≈ {promedio_pi:.5f}"
        )

    t1 = tiempos[0]

    speedup = [t1 / t for t in tiempos]

    ideal = procesos

    plt.figure(figsize=(8, 5))

    plt.plot(
        procesos,
        speedup,
        marker="o",
        linewidth=2,
        label="Speedup Real"
    )

    plt.plot(
        procesos,
        ideal,
        linestyle="--",
        marker="s",
        label="Speedup Ideal"
    )

    plt.title("Ejercicio 1 - Parallel Speedup")
    plt.xlabel("Numero de Procesos")
    plt.ylabel("Speedup")

    plt.grid(True)
    plt.legend()

    plt.savefig("speedup.png")

    plt.close()

    print("\nGrafica generada:")
    print("speedup.png")



# =====================================================
# EJERCICIO 2
# Greedy Knapsack
# =====================================================

def greedy_ratio(values, weights, capacity):

    items = list(range(len(values)))

    items.sort(
        key=lambda i: values[i] / weights[i],
        reverse=True
    )

    chosen = []
    total_value = 0
    total_weight = 0

    for i in items:

        if total_weight + weights[i] <= capacity:

            chosen.append(i)
            total_weight += weights[i]
            total_value += values[i]

    return chosen, total_value, total_weight


def greedy_lightest(values, weights, capacity):

    items = list(range(len(values)))

    items.sort(
        key=lambda i: weights[i]
    )

    chosen = []
    total_value = 0
    total_weight = 0

    for i in items:

        if total_weight + weights[i] <= capacity:

            chosen.append(i)

            total_weight += weights[i]
            total_value += values[i]

    return chosen, total_value, total_weight

def ejercicio2():

    print("\n==========================")
    print("EJERCICIO 2")
    print("==========================")

    # Instancia diseñada para que
    # Menor Peso sea peor que Valor/Peso

    items = ["A", "B", "C"]

    values = [120, 50, 50]
    weights = [20, 10, 10]

    capacity = 20

    ratio = greedy_ratio(
        values,
        weights,
        capacity
    )

    light = greedy_lightest(
        values,
        weights,
        capacity
    )

    print("\nMetodo Valor/Peso")

    print(
        "Objetos:",
        [items[i] for i in ratio[0]]
    )

    print(
        "Valor total:",
        ratio[1]
    )

    print(
        "Peso usado:",
        ratio[2]
    )

    print("\nMetodo Menor Peso")

    print(
        "Objetos:",
        [items[i] for i in light[0]]
    )

    print(
        "Valor total:",
        light[1]
    )

    print(
        "Peso usado:",
        light[2]
    )

    plt.figure(figsize=(8,5))

    metodos = [
        "Valor/Peso",
        "Menor Peso"
    ]

    valores = [
        ratio[1],
        light[1]
    ]

    plt.bar(
        metodos,
        valores,
        color=["green", "orange"]
    )

    plt.ylabel("Valor Total")

    plt.title(
        "Ejercicio 2 - Comparacion Greedy Knapsack"
    )

    plt.savefig("knapsack.png")

    plt.close()

    print("\nGrafica generada:")
    print("knapsack.png")

# =====================================================
# FUNCIONES TSP
# =====================================================

cities = {
    "A": (0, 0),
    "B": (0, 3),
    "C": (4, 0),
    "D": (4, 3),
    "E": (2, 5)
}


def tour_length(route):

    return sum(
        math.dist(
            cities[route[i]],
            cities[route[(i + 1) % len(route)]]
        )
        for i in range(len(route))
    )


def two_opt_move(route, i, k):

    return (
        route[:i]
        + route[i:k + 1][::-1]
        + route[k + 1:]
    )

def nearest_neighbor():

    unvisited = list(cities.keys())

    current = random.choice(unvisited)

    route = [current]

    unvisited.remove(current)

    while unvisited:

        next_city = min(
            unvisited,
            key=lambda c:
            math.dist(
                cities[current],
                cities[c]
            )
        )

        route.append(next_city)

        unvisited.remove(next_city)

        current = next_city

    return route

def two_opt(route):

    best = route[:]

    improvements = 0

    improved = True

    while improved:

        improved = False

        best_length = tour_length(best)

        for i in range(1, len(best) - 1):

            for k in range(i + 1, len(best)):

                candidate = two_opt_move(
                    best,
                    i,
                    k
                )

                candidate_length = tour_length(
                    candidate
                )

                if candidate_length < best_length:

                    best = candidate

                    best_length = candidate_length

                    improvements += 1

                    improved = True

    return best, best_length, improvements


def ejercicio3():

    print("\n==========================")
    print("EJERCICIO 3")
    print("==========================")

    random_route = list(cities.keys())

    random.shuffle(random_route)

    nn_route = nearest_neighbor()

    random_result = two_opt(random_route)

    nn_result = two_opt(nn_route)

    print("\nInicio Aleatorio")

    print(
        "Mejoras:",
        random_result[2]
    )

    print(
        "Longitud Final:",
        round(random_result[1], 2)
    )

    print("\nVecino Mas Cercano")

    print(
        "Mejoras:",
        nn_result[2]
    )

    print(
        "Longitud Final:",
        round(nn_result[1], 2)
    )

    plt.figure(figsize=(8, 5))

    metodos = [
        "Aleatorio",
        "Vecino Cercano"
    ]

    mejoras = [
        random_result[2],
        nn_result[2]
    ]

    plt.bar(
        metodos,
        mejoras,
        color=[
            "royalblue",
            "darkorange"
        ]
    )

    plt.ylabel("Numero de Mejoras")

    plt.title(
        "Ejercicio 3 - Comparacion 2-opt"
    )

    plt.savefig("two_opt.png")

    plt.close()

    print("\nGrafica generada:")
    print("two_opt.png")

# =====================================================
# SIMULATED ANNEALING
# =====================================================

def simulated_annealing(schedule="linear"):

    route = list(cities.keys())
    random.shuffle(route)

    current = route[:]
    current_cost = tour_length(current)

    best = current[:]
    best_cost = current_cost

    costs = []

    T = 100

    start = time.perf_counter()

    for iteration in range(100):

        i = random.randint(1, len(route) - 2)
        k = random.randint(i + 1, len(route) - 1)

        candidate = two_opt_move(current, i, k)

        candidate_cost = tour_length(candidate)

        delta = current_cost - candidate_cost

        if delta > 0:

            current = candidate
            current_cost = candidate_cost

        else:

            probability = math.exp(delta / max(T, 0.001))

            if random.random() < probability:
                current = candidate
                current_cost = candidate_cost

        if current_cost < best_cost:

            best = current[:]
            best_cost = current_cost

        costs.append(best_cost)

        if schedule == "linear":
            T = max(0.1, T - 1)

        elif schedule == "exponential":
            T = max(0.1, T * 0.95)

    execution_time = time.perf_counter() - start

    return best, best_cost, costs, execution_time

# =====================================================
# EJERCICIO 4
# Simulated Annealing
# =====================================================

def ejercicio4():

    print("\n==========================")
    print("EJERCICIO 4")
    print("==========================")

    linear = simulated_annealing("linear")

    exponential = simulated_annealing(
        "exponential"
    )

    print("\nEnfriamiento Lineal")

    print(
        "Costo Final:",
        round(linear[1], 2)
    )

    print(
        "Tiempo:",
        round(linear[3], 4),
        "s"
    )

    print("\nEnfriamiento Exponencial")

    print(
        "Costo Final:",
        round(exponential[1], 2)
    )

    print(
        "Tiempo:",
        round(exponential[3], 4),
        "s"
    )

    # -----------------------------------
    # Grafica 1
    # Convergencia
    # -----------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        linear[2],
        label="Lineal"
    )

    plt.plot(
        exponential[2],
        label="Exponencial"
    )

    plt.xlabel("Iteraciones")
    plt.ylabel("Costo")

    plt.title(
        "Ejercicio 4 - Curvas de Convergencia"
    )

    plt.legend()

    plt.savefig(
        "annealing_convergencia.png"
    )

    plt.close()

    # -----------------------------------
    # Grafica 2
    # Tiempo vs Calidad
    # -----------------------------------

    plt.figure(figsize=(8,5))

    metodos = [
        "Lineal",
        "Exponencial"
    ]

    calidad = [
        linear[1],
        exponential[1]
    ]

    plt.bar(
        metodos,
        calidad,
        color=["purple", "red"]
    )

    plt.ylabel(
        "Costo Final"
    )

    plt.title(
        "Ejercicio 4 - Calidad de Solucion"
    )

    plt.savefig(
        "annealing_calidad.png"
    )

    plt.close()

    print(
        "\nGraficas generadas:"
    )

    print(
        "annealing_convergencia.png"
    )

    print(
        "annealing_calidad.png"
    )

# =====================================================
# FUNCIONES MST
# =====================================================

def generate_random_cities(n=9):

    cities = {}

    for i in range(n):

        cities[i] = (
            random.uniform(0, 100),
            random.uniform(0, 100)
        )

    return cities


def route_length(route, cities):

    total = 0

    for i in range(len(route)):

        a = cities[route[i]]
        b = cities[route[(i + 1) % len(route)]]

        total += math.dist(a, b)

    return total


def mst_approximation(cities):

    unvisited = list(cities.keys())

    current = unvisited[0]

    route = [current]

    unvisited.remove(current)

    while unvisited:

        next_city = min(
            unvisited,
            key=lambda c:
            math.dist(
                cities[current],
                cities[c]
            )
        )

        route.append(next_city)

        unvisited.remove(next_city)

        current = next_city

    return route


def route_two_opt(route, cities):

    best = route[:]

    improved = True

    while improved:

        improved = False

        best_length = route_length(
            best,
            cities
        )

        for i in range(1, len(best) - 1):

            for k in range(i + 1, len(best)):

                candidate = (
                    best[:i]
                    + best[i:k+1][::-1]
                    + best[k+1:]
                )

                candidate_length = route_length(
                    candidate,
                    cities
                )

                if candidate_length < best_length:

                    best = candidate

                    best_length = candidate_length

                    improved = True

    return best, best_length


# =====================================================
# EJERCICIO 5
# MST Approximation
# =====================================================

def ejercicio5():

    print("\n==========================")
    print("EJERCICIO 5")
    print("==========================")

    ratios = []
    mst_costs = []
    mst2_costs = []
    bl_costs = []
    sa_costs = []

    for _ in range(100):
        # 1. Generar 100 instancias aleatorias de 9 ciudades
        cities_e5 = generate_random_cities(9)

        # 2. MST y MST + 2-opt
        mst_route = mst_approximation(cities_e5)
        mst_cost = route_length(mst_route, cities_e5)
        _, mst2_cost = route_two_opt(mst_route, cities_e5)

        # 3. Búsqueda Local (Vecino Cercano + 2-opt sobre estas 9 ciudades)
        unvisited = list(cities_e5.keys())
        current = random.choice(unvisited)
        nn_route = [current]
        unvisited.remove(current)
        while unvisited:
            next_city = min(unvisited, key=lambda c: math.dist(cities_e5[current], cities_e5[c]))
            nn_route.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        # Funciones auxiliares locales para evaluar con el diccionario de 9 ciudades de e5
        def tour_len_ext(r, c_dict):
            return sum(math.dist(c_dict[r[i]], c_dict[r[(i + 1) % len(r)]]) for i in range(len(r)))

        def two_opt_ext(r, c_dict):
            best = r[:]
            improved = True
            while improved:
                improved = False
                b_len = tour_len_ext(best, c_dict)
                for i in range(1, len(best) - 1):
                    for k in range(i + 1, len(best)):
                        cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                        c_len = tour_len_ext(cand, c_dict)
                        if c_len < b_len:
                            best = cand
                            b_len = c_len
                            improved = True
            return best, b_len

        _, bl_cost = two_opt_ext(nn_route, cities_e5)

        # 4. Recocido Simulado adaptado a estas 9 ciudades
        sa_route = list(cities_e5.keys())
        random.shuffle(sa_route)
        curr = sa_route[:]
        curr_cost = tour_len_ext(curr, cities_e5)
        best_sa = curr[:]
        best_sa_cost = curr_cost
        T = 100

        for _ in range(100):
            i = random.randint(1, len(sa_route) - 2)
            k = random.randint(i + 1, len(sa_route) - 1)
            cand = curr[:i] + curr[i:k + 1][::-1] + curr[k + 1:]
            cand_cost = tour_len_ext(cand, cities_e5)
            delta = curr_cost - cand_cost

            if delta > 0 or random.random() < math.exp(delta / max(T, 0.001)):
                curr = cand
                curr_cost = cand_cost

            if curr_cost < best_sa_cost:
                best_sa = curr[:]
                best_sa_cost = curr_cost
            T = max(0.1, T - 0.95)

        sa_costs.append(best_sa_cost)
        bl_costs.append(bl_cost)

        # Estadísticas de MST para el histograma
        ratio = mst_cost / mst2_cost
        ratios.append(ratio)
        mst_costs.append(mst_cost)
        mst2_costs.append(mst2_cost)

    promedio_mst = sum(mst_costs) / len(mst_costs)
    promedio_mst2 = sum(mst2_costs) / len(mst2_costs)
    promedio_bl = sum(bl_costs) / len(bl_costs)
    promedio_sa = sum(sa_costs) / len(sa_costs)

    print(f"\nCosto promedio MST: {promedio_mst:.2f}")
    print(f"Costo promedio MST + 2-opt: {promedio_mst2:.2f}")
    print(f"Costo promedio Búsqueda Local: {promedio_bl:.2f}")
    print(f"Costo promedio Recocido: {promedio_sa:.2f}")
    print(f"Ratio promedio: {sum(ratios)/len(ratios):.3f}")

    # -----------------------------------
    # Histograma (MST Approximation)
    # -----------------------------------
    plt.figure(figsize=(8,5))
    plt.hist(ratios, bins=10, color="skyblue", edgecolor="black")
    plt.axvline(x=2, color="red", linestyle="--", label="Limite teorico = 2")
    plt.xlabel("Razon de Aproximacion")
    plt.ylabel("Frecuencia")
    plt.title("Ejercicio 5 - MST Approximation")
    plt.legend()
    plt.savefig("mst_histograma.png")
    plt.close()

    # -----------------------------------
    # Comparación de costos (Gráfica de barras)
    # -----------------------------------
    plt.figure(figsize=(8,5))

    metodos = [
        "MST",
        "MST + 2-opt",
        "Busqueda Local",
        "Recocido"
    ]

    costos = [
        promedio_mst,
        promedio_mst2,
        promedio_bl,
        promedio_sa
    ]

    plt.bar(
        metodos,
        costos,
        color=[
            "blue",
            "green",
            "orange",
            "purple"
        ]
    )

    plt.ylabel("Costo")
    plt.title("Ejercicio 5 - Comparacion")
    plt.savefig("mst_comparacion.png")
    plt.close()

    print("\nGraficas generadas:")
    print("mst_histograma.png")
    print("mst_comparacion.png")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    ejercicio1()
    ejercicio2()
    ejercicio3()
    ejercicio4()
    ejercicio5()

    print("\nPrograma ejecutado correctamente.")