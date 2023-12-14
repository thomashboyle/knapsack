
# input is list of tuples [(profit0,weight0),(profit1,weight1),...] and capacity c
def greedy_heuristic(itemList, capacity):
    efficiencyOrderedInput = sorted(enumerate(itemList), key=lambda x: x[1][0]/x[1][1], reverse=True)
    decisionList = [0]*len(itemList)
    totalWeight, totalProfit = 0, 0

    for (originalIndex, (profit, weight)) in efficiencyOrderedInput:
        if totalWeight + weight <= capacity:
            decisionList[originalIndex] = 1
            totalWeight += weight
            totalProfit += profit
    return decisionList

def relaxed_greedy_heuristic(itemList, capacity):
    efficiencyOrderedInput = sorted(enumerate(itemList), key=lambda x: x[1][0]/x[1][1], reverse=True)
    decisionList = [0]*len(itemList)
    totalWeight, totalProfit = 0, 0

    for (originalIndex, (profit, weight)) in efficiencyOrderedInput:
        if totalWeight + weight <= capacity:
            decisionList[originalIndex] = 1
            totalWeight += weight
            totalProfit += profit
        else:
            decisionList[originalIndex] = (capacity - totalWeight) / weight
            totalWeight = capacity
            totalProfit += profit * decisionList[originalIndex]
    return decisionList, totalProfit

def bellman_recursion(itemSet, capacity):
    if capacity == 0 or len(itemSet) == 0:
        return 0
    elif capacity < itemSet[-1][1]:
        return bellman_recursion(itemSet[0:-1], capacity)
    else:
        return max(bellman_recursion(itemSet[0:-1],
                                     capacity),
                   bellman_recursion(itemSet[0:-1],
                                     capacity - itemSet[-1][1]) +
                                        itemSet[-1][0])

def bellman_recursion_decision(itemSet, capacity):
    if capacity == 0 or len(itemSet) == 0:
        return [0]*len(itemSet)
    elif capacity < itemSet[-1][1]: # item cannot be packed
        return bellman_recursion_decision(itemSet[0:-1], capacity)+[0]
    else: # item can be packed
        dontTakeDecisions = bellman_recursion_decision(itemSet[0:-1], capacity)
        takeDecisions = bellman_recursion_decision(itemSet[0:-1], capacity - itemSet[-1][1])
        if profit_of_set(itemSet[0:-1], dontTakeDecisions, False) > profit_of_set(itemSet[0:-1], takeDecisions, False) + itemSet[-1][0]:
            return dontTakeDecisions + [0]
        else:
            return takeDecisions + [1]


# returns profit with given items and decisions
def profit_of_set(itemSet, decisionSet, useWeight):
    # print("weight" if useWeight else "profit", "of", itemSet, decisionSet, "=", end=" ")
    if type(decisionSet) is int:
        # print(bin(decisionSet))
        total = 0
        i = len(itemSet) - 1
        while decisionSet != 0:
            total += (decisionSet & 1)*itemSet[i][useWeight]
            i -= 1
            decisionSet >>= 1
        # print(total)
        return total
    else:
        # print(sum([item[useWeight] * int(decision) for item, decision in zip(itemSet, decisionSet)]))
        return sum([item[useWeight] * int(decision) for item, decision in zip(itemSet, decisionSet)])


# figure 2.2
def DP1(itemSet, capacity):
    dpTable = []
    #initialize table to 0
    for i in range(len(itemSet) + 1):
        dpTable += [[0]*(capacity+1)]
    # loop through items
    for (i, (profit, weight)) in enumerate(itemSet):
        # fill out subproblem profit if item cant be packed
        for subProblemCapacity in range(1,weight):
            dpTable[i+1][subProblemCapacity] = dpTable[i][subProblemCapacity]
        # decide whether to add it if it can fit
        for subProblemCapacity in range(weight, capacity+1):
            dpTable[i+1][subProblemCapacity] = max(dpTable[i][subProblemCapacity - weight] + profit,
                                                   dpTable[i][subProblemCapacity])
    return dpTable


# DP1 that keeps track of decisions
def DPdecisions(itemSet, capacity):
    dpTable = []
    decisionTable = []
    for i in range(len(itemSet) + 1):
        dpTable += [[0]*(capacity+1)]
        decisionTable += [[0]*(capacity+1)]
    for (i, (profit, weight)) in enumerate(itemSet):
        for subProblemCapacity in range(1,weight): # item cannot be packed
            dpTable[i+1][subProblemCapacity] = dpTable[i][subProblemCapacity]
            decisionTable[i+1][subProblemCapacity] = decisionTable[i][subProblemCapacity] << 1
        for subProblemCapacity in range(weight, capacity+1): # item can be packed
            if dpTable[i][subProblemCapacity - weight] + profit > dpTable[i][subProblemCapacity]:
                dpTable[i + 1][subProblemCapacity] = dpTable[i][subProblemCapacity - weight] + profit
                decisionTable[i + 1][subProblemCapacity] = (decisionTable[i][subProblemCapacity - weight] << 1) + 1
            else:
                dpTable[i + 1][subProblemCapacity] = dpTable[i][subProblemCapacity]
                decisionTable[i + 1][subProblemCapacity] = decisionTable[i][subProblemCapacity] << 1

    return dpTable, decisionTable

# figure 2.4
def DP2(itemSet, capacity):
    # only keep track of one row at a time
    dpTable = [0]*(capacity+1)

    # overwrite row for each item
    for profit, weight in itemSet:
        # fill out row from high to low capacity
        for subProblemCapacity in range(capacity, weight-1, -1):  # item can be packed
            dpTable[subProblemCapacity] = max(dpTable[subProblemCapacity - weight] + profit,
                                                   dpTable[subProblemCapacity])
    return dpTable


def DP3(itemSet, capacity):
    solution = []
    solutionProfit = 0
    numItems = len(itemSet)
    iterationCapacity = capacity
    # run DP2 for each item in solution set (so dont have to store decisions)
    while iterationCapacity >= min(itemSet, key=lambda x: x[1])[1]:
        profitTable = [0] * (capacity + 1)
        itemTable = [0] * (capacity + 1)

        for i, (profit, weight) in enumerate(itemSet[:numItems]):
            for subProblemCapacity in range(iterationCapacity, weight - 1, -1):  # item can be packed
                if profitTable[subProblemCapacity - weight] + profit > profitTable[subProblemCapacity]:
                    profitTable[subProblemCapacity] = profitTable[subProblemCapacity - weight] + profit
                    itemTable[subProblemCapacity] = i
        itemToAdd = itemTable[iterationCapacity]
        solution += [itemToAdd]
        solutionProfit = max(profitTable[iterationCapacity], solutionProfit)
        numItems = itemToAdd - 1
        iterationCapacity -= itemSet[itemToAdd][1]
    return solutionProfit, solution


def branch_and_bound(itemSet, capacity, branchAt=0, branchDecisions=0, bestProfit=0, bestDecisions=0):
    # get weight and profit of current branch
    current_weight = profit_of_set(itemSet[0:branchAt], branchDecisions, True)
    current_profit = profit_of_set(itemSet[0:branchAt], branchDecisions, False)
    if current_weight > capacity:
        return bestProfit, bestDecisions  # if over capacity, stop processing
    # if best solution seen so far
    if current_profit > bestProfit:
        bestProfit = current_profit
        bestDecisions = branchDecisions << (len(itemSet) - branchAt)
    if branchAt > len(itemSet):
        return bestProfit, bestDecisions  # if done with items, stop processing

    # use relaxed greedy heuristic to get upper bound, only check branches if could be more.
    if (relaxed_greedy_heuristic(itemSet[branchAt:], capacity - current_weight)[1]) // 1 + current_profit > bestProfit:
        bestProfit, bestDecisions = branch_and_bound(itemSet, capacity, branchAt + 1, (branchDecisions << 1) + 1, bestProfit, bestDecisions)
        bestProfit, bestDecisions = branch_and_bound(itemSet, capacity, branchAt + 1, branchDecisions << 1, bestProfit, bestDecisions)
    return bestProfit, bestDecisions


# extended greed is at least half of optimal profit
def extended_greedy(itemSet, capacity):
    # run greedy heuristic
    greedyDecisions = greedy_heuristic(itemSet, capacity)
    # find item with highest profit
    maxItem = max(enumerate(itemSet), key=lambda x: x[1][0])
    # choose higher of two profits
    if profit_of_set(itemSet, greedyDecisions, False) > maxItem[1][0]:
        return greedyDecisions
    else:
        return [0]*(maxItem[0]) + [1] + [0]*(len(itemSet) - maxItem[0] - 1)


# extended greedy improved is at least 3/4 of optimal profit
def extended_greedy_improved(itemSet, capacity):
    maxItem = max(enumerate(itemSet), key=lambda x: x[1][0])
    bestSolution = [0] * (maxItem[0]) + [1] + [0] * (len(itemSet) - maxItem[0] - 1)
    bestProfit = maxItem[1][0]
    # choose all pairs
    for i, (profiti, weighti) in enumerate(itemSet):
        for j, (profitj, weightj) in enumerate(itemSet[i+1:]):
            print("Pair chosen:", i, j, (profiti, weighti), (profitj, weightj))
            # if pair fits in bag
            if weighti + weightj <= capacity:
                # find subset of items left that can fit in remaining capacity and have less profit than both
                subsetCapacity = capacity - weighti - weightj
                subset = [x for x in itemSet if x[0] < min(profiti, profitj) and x[1] <= subsetCapacity]
                subsetIndices  = [i for i, x in enumerate(itemSet) if x[0] < min(profiti, profitj) and x[1] <= subsetCapacity]

                # if none left
                if len(subset) == 0:
                    # check if better profit
                    if profiti + profitj > bestProfit:
                        bestSolution = [0]*len(itemSet)
                        bestSolution[i] = 1
                        bestSolution[j + i + 1] = 1
                        bestProfit = profiti + profitj
                    continue

                # run extended greedy algorithm on remaining items
                subsetSolution = extended_greedy(subset, subsetCapacity)
                subsetProfit = profit_of_set(subset, subsetSolution, False)

                if profiti + profitj + subsetProfit > bestProfit:
                    bestProfit = profiti + profitj + subsetProfit
                    bestSolution = [1 if i in subsetIndices and subsetSolution[subsetIndices.index(i)]
                                    else 0 for i, _ in enumerate(itemSet)]
                    bestSolution[i] = 1
                    bestSolution[i + j + 1] = 1
    return bestSolution


import networkx as nx
import matplotlib.pyplot as plt

class state:
    def __init__(self, name, districts, state):
        self.name = name
        self.districts = sorted(enumerate(districts), key=lambda x: x[1])
        self.stateVotes = state
        self.solvedState = [0]
        self.stateFlips = 0

    def solveState(self):
        votes = 0
        stateFlipped = False
        for _, district in self.districts:
            votes += district
            if not stateFlipped and votes >= self.stateVotes:
                self.stateFlips = len(self.solvedState)
                self.solvedState += [self.stateVotes, self.stateVotes]
                stateFlipped = True
            self.solvedState += [votes]
        if not stateFlipped:
            self.solvedState += [self.stateVotes, self.stateVotes]


def makeGraph(states, votesNeeded):
    G = nx.DiGraph()

    states = sorted(enumerate(states), key=lambda x: len(x[1]), reverse=True)
    stateLens = list(map(lambda x: len(x[1]) - 1, states))
    votesLeft = [sum(stateLens[i+1:]) for i, _ in enumerate(states)]
    # print(states)
    maxVotes = 0
    for i, people in enumerate(states[0][1]): # handle source node
        if i + votesLeft[0] < votesNeeded: continue
        if i > votesNeeded: break
        # print("start-0", "1-" + str(i), i, people)
        G.add_edge("start-0", "1-" + str(i), weight=people)
    maxVotes += len(states[0][1])
    
    for i, (_, state) in enumerate(states[1:-1]):    # loop through states to create layers
        # print(state)
        for j, people in enumerate(state):      # loop through edges to be added
            if people is None: continue
            for ecVotes in range(max(0, votesNeeded - votesLeft[i+1] - j), min(maxVotes, votesNeeded-j+1)): # loop through nodes in layer 
                # print(str(i+1) + "-" + str(ecVotes), str(i+2) + "-" + str(ecVotes+j), j, people)
                G.add_edge(str(i+1) + "-" + str(ecVotes), str(i+2) + "-" + str(ecVotes+j), weight=people)
        maxVotes += len(state) - 1
        # G.add_edges_from([[(state.name + str(ecVotes), states[i+1].name + str(ecVotes+j), people) for j, people in enumerate(state)] for ecVotes in range(max)])
    
    for i, people in enumerate(states[-1][1]): # handle sink node
        if i > votesNeeded: break # stop making nodes if greater than votes needed
        # print(str(len(states) - 1) + "-" + str(votesNeeded-i), "end-" + str(votesNeeded), i, people)
        G.add_edge(str(len(states) - 1) + "-" + str(votesNeeded-i), "end-" + str(votesNeeded), weight=people)

    return G, [(i, j) for i, (j, _) in enumerate(states)]


def makeGraph2(states, votesNeeded):
    G = nx.DiGraph()
    return G


def pathTraceback(G, path, statesMap, country, showDetails=False):
    weights = nx.get_edge_attributes(G, "weight")
    prev = 0
    for i, j in statesMap:
        ecVotes = int(path[i+1].split('-')[1])
        ecVotesFromState = ecVotes - prev
        prev = ecVotes
        peopleTowardsState = weights[path[i], path[i+1]]
        
        print("Move", peopleTowardsState, "to state", country[j].name, "to gain", ecVotesFromState, "electoral college votes")

        if showDetails: # TODO fix
            votes = 0
            stateFlipped = False
            for x in range(ecVotesFromState):
                district = country[j].districts[x]
                if not stateFlipped and votes + district[1] > country[j].stateVotes:
                    print("\tput", country[j].stateVotes - votes, "people in district", district[0], "to win the state")
                    if x > ecVotesFromState: break
                    else:
                        stateFlipped = True
                        print("\t\tput remaining", votes + district[1] - country[j].stateVotes, "people in district", district[0])
                else:
                    print("\tput", district[1], "people in district", district[0])
                
                votes += district[1]


def ECsolver(country, votesToFlip):
    for s in country:
        s.solveState()
    L, statesMap = makeGraph(map(lambda x: x.solvedState, country), votesToFlip)
    # nx.draw(L, with_labels=True)
    # plt.show()
    print(nx.dijkstra_path_length(L, "start-0", "end-" + str(votesToFlip)),
          nx.dijkstra_path(L, "start-0", "end-" + str(votesToFlip)))
    path = nx.dijkstra_path(L, "start-0", "end-" + str(votesToFlip))

    weights=nx.get_edge_attributes(L, "weight")

    pathTraceback(L, path, statesMap, country)



l = [state("South Ruhspekt", [1, 4, 4, 7], 4), \
     state("North Ruhspekt", [2, 3, 4, 6, 8], 8), \
     state("New Bikesico", [9, 11, 16, 54], 5)]

# l = [state("Boogville", [1, 3, 19], 3),
#      state("Oogburg", [7], 3),
#      state("Fellatown", [4, 5], 16)]

ECsolver(l, 9)

