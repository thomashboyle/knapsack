# This is a sample Python script.

import algorithms

def print_table(table):
    for i, column in enumerate(table):
        print(column)
def print_decision_table(table):
    for i, column in enumerate(table):
        print([bin(decisions) for decisions in column])

items1 = [(9, 7),  (7, 9), (8, 6), (6, 2), (6, 5), (5, 3), (3, 4)]
itemstxtbk = [(6, 2),  (5, 3), (8, 6), (9, 7), (6, 5), (7, 9), (3, 4)]
itemstxtbk2 = [(3, 4),  (5, 3), (8, 6), (9, 7), (6, 5), (6, 2), (7, 9)]

decisionsi1c9 = [0, 0, 0, 1, 0, 1, 1]

print(algorithms.greedy_heuristic(items1, 9) == decisionsi1c9)
print(algorithms.relaxed_greedy_heuristic(items1, 9))
print(algorithms.bellman_recursion(itemstxtbk[0:6], 7))
print(algorithms.bellman_recursion_decision(items1, 9))
print_table(algorithms.DP1(itemstxtbk, 9))
dp, decisions = algorithms.DPdecisions(itemstxtbk, 9)
print_decision_table(decisions)
print(algorithms.DP2(itemstxtbk, 9))
print(algorithms.DP3(itemstxtbk, 9))
print(algorithms.branch_and_bound(itemstxtbk, 9))


