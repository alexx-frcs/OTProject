import time
import numpy as np
import generate_data as data
import algorithms as test
import sys
import matplotlib.pyplot as plt


s = int(sys.argv[1])
print('s', s)
dimension = int(sys.argv[2])
print('dimension', dimension)
method1 = sys.argv[3]
print(method1)
method2 = sys.argv[4]
print(method2)

results = {}
results[method1] = []
results[method2] = []

Ns=[10, 20, 30, 40, 50]

methods = [method1, method2]
for method in methods:
    for N in Ns:
        accuracies =[]
        for epoch in range(100):
            print(epoch)
            test.load_data(N, s, dimension)

            ncs = [1]

            # Parameter for Sinkhorn or LC-WMD
            method_param = 1


            target_accuracy = 0.9
            sort_flag = True

            ### Query set

            query_idx = np.arange(1)
            qn = len(query_idx)
            queries = test.queries[query_idx]
            queries_modified = [test.queries_modified[i] for i in query_idx]
            answers = test.answers[:,query_idx]

            def call_method_weighted(fnc, q_index, input_idx, nc):
                result = np.zeros(nc, dtype=np.int32)
                score_result = np.zeros(nc, dtype=np.float32)
                fnc(queries_modified[q_index], input_idx, result, score_result, sort_flag)
                return result

            def call_method_uniform(fnc, q_index, input_idx, nc, method_param=-1):
                result = np.zeros(nc, dtype=np.int32)
                score_result = np.zeros(nc, dtype=np.float32)
                if method_param == -1:
                    fnc(queries[q_index], input_idx, result, score_result, sort_flag)
                else:
                    fnc(queries[q_index], input_idx, result, score_result, sort_flag, method_param)
                return result

            def call_exact(q_index, input_idx):
                return [test.exact_emd(queries[q_index], input_idx)]

            fdic = {}
            fdic["mean"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.means_rank, q_index, input_idx, nc)
            fdic["overlap"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.overlap_rank, q_index, input_idx, nc)
            fdic["quadtree"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.quadtree_rank, q_index,input_idx,  nc)
            fdic["flowtree"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.flowtree_rank, q_index, input_idx, nc)
            fdic["rwmd"] = lambda q_index, input_idx, nc:call_method_uniform(test.rwmd, q_index, input_idx, nc)
            fdic["lcwmd"] = lambda q_index, input_idx, nc:call_method_uniform(test.lc_wmd, q_index, input_idx, nc, method_param)
            fdic["sinkhorn"] = lambda q_index, input_idx, nc:call_method_uniform(test.sinkhorn, q_index, input_idx, nc, method_param)
            fdic["exact"] = lambda q_index, input_idx, nc:call_exact(q_index, input_idx)
            ### Main

            accuracy = 0
            accs = np.zeros(len(methods))
            total_time = 0
            clength = 0
            input_idx = None
            orig_input_idx = np.zeros(len(test.dataset), dtype=np.int32)
            for i in range(len(test.dataset)):
                orig_input_idx[i] = i

            start = time.time()
            for q in range(qn):
                input_idx = orig_input_idx
                for m in range(len(methods)):
                    input_idx = fdic[methods[m]](q, input_idx, ncs[m])
                    if m==0:
                        clength += np.mean([len(test.dataset[j]) for j in input_idx])
                    if answers[0,q] in input_idx:
                        accs[m] += 1
                if answers[0,q] in input_idx:
                    accuracy += 1
                total_time = time.time() - start
            # At the end of each iteration of the main loop, input_idx is a list of indices (in the dataset) of the final candidates
            # for the current query, of the length specified by the final number in the input list of candidate numbers (third parameter).
            # As an example let us
            #  print the top-10 nearest neighbors found by the pipeline for each of the first three queries:

                accuracies.append(accuracy)
        results[method].append(sum(accuracies)/100)

plt.plot(results[method1], label= method1)

# Tracer la courbe pour 'method2'
plt.plot(results[method2], label= method2)

# Ajouter des étiquettes et un titre
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.title('Accuracies for Quadtree and Flowtree')

# Ajouter une légende
plt.legend()

# Sauvegarder le graphique
plt.savefig('comparaison_methods_s{}_d{}.png'.format(s, dimension))


