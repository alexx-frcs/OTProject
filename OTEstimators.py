import numpy as np
import random 
import generate_data


EPS = 1e-8
EPS2 = 1e-5
EPS3 = 1e-3




class OTEstimators():
    def __init__(self):
        self.stage=0
        self.parents = []
        self.leaf = []
        self.marked = []
        self.num_queries = 0
        self.node_id = []
        self.id_node = []
        self.subtree = []
        self.excess = []
        self.delta_node = []
        self.dictionary = None  
        self.unleaf = []
        self.dataset_embedding = []
        self.raw_dataset = []
        self.means = None  
        self.query_mean = None  
        self.distances = []
    
    def load_vocabulary(self, points):
        if self.stage != 0:
            raise ValueError("load_vocabulary() should be called once in the beginning")
        
        self.stage = 1

        if points.ndim != 2:
            raise ValueError("load_vocabulary() expects a two-dimensional NumPy array")

        n, d = points.shape
        self.dictionary = points

        cmin = np.min(self.dictionary)
        cmax = np.max(self.dictionary)
        delta = cmax - cmin
        cmin -= delta

        bounding_box = [(cmin + random.uniform(0, delta), cmax + random.uniform(0, delta)) for _ in range(d)]

        all_indices = list(range(n))
        self.leaf = [None] * n
        self.build_quadtree(all_indices, bounding_box, 0, -1)

        self.num_queries = 0
        self.marked = [-1] * len(self.parents)
        self.node_id = [None] * len(self.parents)


    
    def check_measure(self, measure):
        sum_masses = 0.0
        n = self.dictionary.shape[0]
        boolean = True 
        
        for atom in measure:
            index, mass = atom
            if index < 0 or index >= n:
                raise ValueError("Invalid index in the measure")

            if mass < -EPS:
                raise ValueError("Negative mass")
            sum_masses += mass
        return boolean

        if abs(sum_masses - 1.0) > EPS3:
            raise ValueError("The masses don't sum to 1")
        
    def load_dataset(self, dataset):
        if self.stage != 1:
            raise ValueError("load_dataset() should be called once after calling load_vocabulary()")
        
        self.stage = 2

        if not dataset:
            raise ValueError("The dataset can't be empty")


        self.dataset_embedding = [self.compute_embedding(measure) for measure in dataset if self.check_measure(measure)]

        self.raw_dataset = dataset
        for measure in self.raw_dataset:
            measure.sort()

        self.means = np.zeros((len(dataset), self.dictionary.shape[1]))
        for i, measure in enumerate(dataset):
            self.means[i] = self.dictionary[measure[0][0]] * measure[0][1]
            for j in range(1, len(measure)):
                self.means[i] += self.dictionary[measure[j][0]] * measure[j][1]

        self.query_mean = np.zeros(self.dictionary.shape[1])
        self.distances = [None] * len(dataset)


    def means_rank(self, query, input_ids, output_ids, output_scores, to_sort):
        self.check_stage()
        self.check_measure(query)
        self.check_input_output_arrays(input_ids, output_ids, output_scores)

        self.query_mean = self.dictionary[query[0][0]] * query[0][1]
        for j in range(1, len(query)):
            self.query_mean += self.dictionary[query[j][0]] * query[j][1]

        self.distances = [(np.linalg.norm(self.query_mean - self.means[input_id])**2, input_id)
                          for input_id in input_ids]

        self.select_topk_aux(len(input_ids), output_ids, output_scores, to_sort)

    def overlap_rank(self, query, input_ids, output_ids, output_scores, to_sort):
        self.check_stage()
        self.check_measure(query)
        
        query_copy = sorted(query)
        scores = []

        for input_id in input_ids:
            point = self.raw_dataset[input_id]
            score = 0
            qp, dp = 0, 0

            while qp < len(query_copy) and dp < len(point):
                if query_copy[qp][0] < point[dp][0]:
                    qp += 1
                elif query_copy[qp][0] > point[dp][0]:
                    dp += 1
                else:
                    score += 1
                    qp += 1
                    dp += 1

            scores.append((-score, input_id))

        self.distances = scores
        self.select_topk_aux(len(input_ids), output_ids, output_scores, to_sort)

    def quadtree_rank(self, query, input_ids, output_ids, output_scores, to_sort):
        self.check_stage()
        self.check_measure(query)
        self.check_input_output_arrays(input_ids, output_ids, output_scores)

        query_embedding = self.compute_embedding(query)
        scores = []

        for input_id in input_ids:
            point_embedding = self.dataset_embedding[input_id]
            score = 0.0
            qp, dp = 0, 0

            while qp < len(query_embedding) or dp < len(point_embedding):
                if qp == len(query_embedding):
                    score += point_embedding[dp][1]
                    dp += 1
                elif dp == len(point_embedding):
                    score += query_embedding[qp][1]
                    qp += 1
                elif query_embedding[qp][0] < point_embedding[dp][0]:
                    score += query_embedding[qp][1]
                    qp += 1
                elif point_embedding[dp][0] < query_embedding[qp][0]:
                    score += point_embedding[dp][1]
                    dp += 1
                else:
                    score += abs(query_embedding[qp][1] - point_embedding[dp][1])
                    qp += 1
                    dp += 1

            scores.append((score, input_id))

        self.distances = scores
        self.select_topk_aux(len(input_ids), output_ids, output_scores, to_sort)

    def flowtree_rank(self, query, input_ids, output_ids, output_scores, to_sort):
        self.check_stage()
        self.check_measure(query)
        self.check_input_output_arrays(input_ids, output_ids, output_scores)

        self.distances = [(self.flowtree_query(query, self.raw_dataset[input_id]), input_id)
                          for input_id in input_ids]

        self.select_topk_aux(len(input_ids), output_ids, output_scores, to_sort)



    def build_quadtree(self, subset, bounding_box, depth, parent):
        node_id = len(self.parents)
        self.parents.append(parent)

        if len(subset) == 1:
            self.leaf[subset[0]] = node_id
            return

        d = self.dictionary.shape[1]
        mid = [(bbox[0] + bbox[1]) / 2.0 for bbox in bounding_box]

        parts = {}
        for ind in subset:
            code = [0] * ((d + 7) // 8)
            for i in range(d):
                if self.dictionary[ind, i] > mid[i]:
                    code[i // 8] |= 1 << (i % 8)
            code_tuple = tuple(code)  # Convert list to tuple for dictionary key
            if code_tuple not in parts:
                parts[code_tuple] = []
            parts[code_tuple].append(ind)

        for part, indices in parts.items():
            new_bounding_box = []
            for i in range(d):
                bit = (part[i // 8] >> (i % 8)) & 1
                if bit:
                    new_bounding_box.append((mid[i], bounding_box[i][1]))
                else:
                    new_bounding_box.append((bounding_box[i][0], mid[i]))
            self.build_quadtree(indices, new_bounding_box, depth + 1, node_id)
    
    def flowtree_query(self, a, b):
        num_nodes = 0
        self.id_node.clear()
        all_nodes = a + b
        for x in all_nodes:
            id = self.leaf[x[0]]
            while id != -1:
                if self.marked[id] != self.num_queries:
                    self.id_node.append(id)
                    self.node_id[id] = num_nodes
                    num_nodes += 1
                self.marked[id] = self.num_queries
                id = self.parents[id]

        if len(self.subtree) < num_nodes:
            self.subtree = [[] for _ in range(num_nodes)]

        for i in range(num_nodes):
            self.subtree[i].clear()

        for i in range(num_nodes):
            u = self.parents[self.id_node[i]]
            if u != -1:
                self.subtree[self.node_id[u]].append(i)

        if len(self.excess) < num_nodes:
            self.excess = [[] for _ in range(num_nodes)]

        self.delta_node = [0.0] * num_nodes
        self.unleaf = [None] * num_nodes

        for x in a:
            self.delta_node[self.node_id[self.leaf[x[0]]]] += x[1]
            self.unleaf[self.node_id[self.leaf[x[0]]]] = x[0]

        for x in b:
            self.delta_node[self.node_id[self.leaf[x[0]]]] -= x[1]
            self.unleaf[self.node_id[self.leaf[x[0]]]] = x[0]

        res = self.run_query(0, self.node_id[0])
        if self.excess[self.node_id[0]]:
            unassigned = sum(x[0] for x in self.excess[self.node_id[0]])
            if unassigned > EPS2:
                raise ValueError("Too much unassigned flow")

        self.num_queries += 1
        return res
    
    def run_query(self, depth, nd):
        res = 0.0
        for x in self.subtree[nd]:
            res += self.run_query(depth + 1, x)

        self.excess[nd].clear()

        if not self.subtree[nd]:
            if abs(self.delta_node[nd]) > EPS:
                self.excess[nd].append((self.delta_node[nd], self.unleaf[nd]))
        else:
            for x in self.subtree[nd]:
                if not self.excess[x]:
                    continue
                same = False
                #print('type excess', type(self.excess[x][0][0]))
                if not self.excess[nd] or self.sign(self.excess[x][0][0]) == self.sign(self.excess[nd][0][0]):
                    same = True
                if same:
                    self.excess[nd].extend(self.excess[x])
                else:
                    while self.excess[x] and self.excess[nd]:
                        u = self.excess[nd][-1]
                        v = self.excess[x][-1]

                        dist = np.linalg.norm(self.dictionary[u[1]] - self.dictionary[v[1]])
                        if abs(u[0] + v[0]) < EPS:
                            self.excess[nd].pop()
                            self.excess[x].pop()
                            res += dist * abs(u[0])
                        elif abs(u[0]) < abs(v[0]):
                            self.excess[nd].pop()
                            self.excess[x][-1] = (v[0] + u[0], v[1])
                            res += dist * abs(u[0])
                        else:
                            self.excess[x].pop()
                            self.excess[nd][-1] = (u[0] + v[0], u[1])
                            res += dist * abs(v[0])

                    if self.excess[x]:
                        self.excess[nd], self.excess[x] = self.excess[x], self.excess[nd]

        return res


    def check_stage(self):
        if self.stage != 2:
            raise ValueError("Need to call load_vocabulary() and load_dataset() first")

    def check_dimension(self, x):
        if x.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional")

    def get_length(self, x):
        return x.shape[0]

    def check_input_output_arrays(self, input_ids, output_ids=None, output_scores=None, input_scores=None):
        self.check_dimension(input_ids)
        
        if output_ids is not None and output_scores is not None:
            self.check_dimension(output_ids)
            self.check_dimension(output_scores)
            if self.get_length(output_ids) != self.get_length(output_scores):
                raise ValueError("Output_ids and output_scores must be of the same length")
            if self.get_length(output_ids) > self.get_length(input_ids):
                raise ValueError("Output_ids and output_scores must be no longer than input_ids")
            
            for val in input_ids:
                if val < 0 or val >= len(self.raw_dataset):
                    raise ValueError("Input_ids contain an invalid index")

        if input_scores is not None:
            self.check_dimension(input_scores)
            if self.get_length(input_ids) != self.get_length(input_scores):
                raise ValueError("Input_ids and input_scores must be of the same length")
            
    def select_topk_aux(self, k1, output_ids, output_scores, to_sort):
        k2 = min(len(output_ids), k1)
        topk_distances = sorted(self.distances[:k1], key=lambda x: x[0], reverse=not to_sort)[:k2]

        for i in range(k2):
            output_scores[i] = topk_distances[i][0]
            output_ids[i] = topk_distances[i][1]
    
    def select_topk(self, input_ids, input_scores, output_ids, output_scores, to_sort):
        self.check_input_output_arrays(input_ids, input_scores, output_ids, output_scores)

        self.distances = list(zip(input_scores, input_ids))

        self.select_topk_aux(len(input_ids), output_ids, output_scores, to_sort)

    def compute_embedding(self, a):
        result = []
        for x in a:
            id = self.leaf[x[0]]
            level = 0
            while id != -1:
                level += 1
                id = self.parents[id]

            id = self.leaf[x[0]]
            while id != -1:
                level -= 1
                result.append((id, x[1] / (1 << level)))
                id = self.parents[id]

        result.sort()

        ans = []
        for x in result:
            if not ans or ans[-1][0] != x[0]:
                ans.append(x)
            else:
                ans[-1] = (ans[-1][0], ans[-1][1] + x[1])

        return ans
            
    def sign(self, x):
        if abs(x) < EPS:
            raise ValueError("Computing sign of ~0")
        return 1 if x > 0 else -1


    