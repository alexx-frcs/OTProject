import numpy as np
import itertools
import random

class Dataset():
    def __init__(self,N,s, dim):
        self.N = N
        self.s = s
        self.dimension = dim

    def generate_random(self):
        # Pre-allocate arrays for performance
        measures = np.empty((self.binomial(self.N, self.s), self.s), dtype=int)
        answers = np.empty(self.binomial(self.N, self.s), dtype=int)
        query = np.arange(self.N, self.N + self.s)

        # Generate all combinations of measures
        for i, subset in enumerate(itertools.combinations(range(self.N), self.s)):
            measures[i] = subset

        # Generate points and normalize them
        points = np.random.normal(0, 1, (self.N + self.s, self.dimension))
        points[:self.N] /= np.linalg.norm(points[:self.N], axis=1, keepdims=True)

        # Randomly choose a measure and update points and answers
        i = random.randint(0, len(measures) - 1)
        closest = measures[i]
        answers[0] = i
        answers[1:] = [x for x in range(len(measures)) if x != i]

        # Perturb points for the query
        eps = 1e-5
        for idx, point_idx in enumerate(closest):
            perturbation = np.random.normal(0, eps / 2, self.dimension)
            points[self.N + idx] = points[point_idx] + perturbation

        # Normalize all points again to ensure they're on the unit sphere
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        return points, measures, query[np.newaxis, :], answers[np.newaxis, :]

    @staticmethod
    def binomial(n, k):
        """Compute the binomial coefficient"""
        return int(np.prod([(n - i) / (i + 1) for i in range(k)]))

    def generate_random_slow2(self):
        # Generate all combinations of measures
        measures = list(itertools.combinations(range(self.N), self.s))

        # Generate points and normalize them
        points = np.random.normal(0, 1, (self.N, self.dimension))
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        # Select a random measure and generate the query
        i = random.randint(0, len(measures) - 1)
        closest = measures[i]

        query = np.arange(self.N, self.N + self.s)
        perturbation = np.random.normal(0, 0.01/ 2, (self.s, self.dimension))
        new_points = points[list(closest)] + perturbation
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)

        # Combine original and new points
        all_points = np.vstack([points, new_points])

        # Answers array (the index of the chosen measure and the remaining ones)
        answers = [i] + [0]*(len(measures)-1)  

        return all_points, np.array(measures), query[np.newaxis, :], np.array(answers)[np.newaxis, :]

    def generate_random_slow(self):
        """
        Returns:
            points: np.array (self.N + self.s, self.dimension) uniformly sampled on the unit sphere
            measures: np.array(binomial(self.N, self.s), self.s) array of lists containing the indexes of the support of the measures
            query: np.array(self.s, ) [self.N, self.N+1, ..., self.N+self.s] the query contains by construction all the last points. 
            answer: np.array(binomial(self.N, self.s), ) indexes of measures sorted in order of proximity to query
        """

        measures = []
        query = []
        answers = []
        for subset in itertools.combinations(range(0, self.N ), self.s):
            measure = subset 
            measures.append(list(measure))

        eps = 1e-5

        points = np.random.normal(0, 1, (self.N, self.dimension))

        norms = np.linalg.norm(points, axis = 1)

        points_sphere = points / norms[:, np.newaxis]


        i = random.randint(0, self.N)
        answers.append(i)
        remaining = [x for x in range(len(measures)) if x!=i]
        answers +=remaining
        closest = measures[i]
        iteration = self.N

        for index in closest:

            perturbation = np.random.normal(0, eps / 2, self.dimension)
            new_point = points_sphere.copy()[index] + perturbation
            new_point_stacked = new_point[np.newaxis, :]
            query.append(iteration)
            iteration+=1
            points_sphere = np.vstack([points_sphere, new_point_stacked])
            
        

        norms = np.linalg.norm(points_sphere, axis = 1)

        points_sphere_bis = points_sphere / norms[:, np.newaxis]
        return np.array(points_sphere_bis), np.array(measures), np.array(query)[np.newaxis, :], np.array(answers)[np.newaxis, :]
    
