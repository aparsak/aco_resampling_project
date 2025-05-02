import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def initialize_pheromones(n):
    return np.ones((n, n))

def aco_outlier_detection(data, num_ants=10, iterations=10, alpha=1, beta=2, evaporation=0.5):
    n = len(data)
    pheromones = initialize_pheromones(n)
    visibility = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                visibility[i][j] = 1 / (euclidean_distance(data[i], data[j]) + 1e-10)

    scores = np.zeros(n)

    for it in range(iterations):
        for ant in range(num_ants):
            visited = []
            current = np.random.randint(n)
            visited.append(current)
            for _ in range(10):  # limited walk
                probs = []
                for j in range(n):
                    if j not in visited:
                        tau = pheromones[current][j] ** alpha
                        eta = visibility[current][j] ** beta
                        probs.append((j, tau * eta))
                if not probs:
                    break
                nodes, weights = zip(*probs)
                total = sum(weights)
                probs_norm = [w / total for w in weights]
                next_node = np.random.choice(nodes, p=probs_norm)
                visited.append(next_node)
                current = next_node

            for i in range(len(visited) - 1):
                pheromones[visited[i]][visited[i + 1]] += 1

        pheromones *= (1 - evaporation)

    for i in range(n):
        scores[i] = np.sum(pheromones[i])

    threshold = np.percentile(scores, 5)
    outliers = np.where(scores < threshold)[0]
    return outliers
