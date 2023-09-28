
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
    print(itemSet, decisionSet)
    if type(decisionSet) is int:
        print(bin(decisionSet))
        total = 0
        i = len(itemSet) - 1
        while decisionSet != 0:
            total += (decisionSet & 1)*itemSet[i][useWeight]
            i -= 1
            decisionSet >>= 1
        return total
    else:
        return sum([item[useWeight] * int(decision) for item, decision in zip(itemSet, decisionSet)])


# figure 2.2
def DP1(itemSet, capacity):
    dpTable = []
    for i in range(len(itemSet) + 1):
        dpTable += [[0]*(capacity+1)]
    for (i, (profit, weight)) in enumerate(itemSet):
        for subProblemCapacity in range(1,weight): # item cannot be packed
            dpTable[i+1][subProblemCapacity] = dpTable[i][subProblemCapacity]
        for subProblemCapacity in range(weight, capacity+1): # item can be packed
            dpTable[i+1][subProblemCapacity] = max(dpTable[i][subProblemCapacity - weight] + profit,
                                                   dpTable[i][subProblemCapacity])
    return dpTable


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
    dpTable = [0]*(capacity+1)

    for profit, weight in itemSet:
        for subProblemCapacity in range(capacity, weight-1, -1):  # item can be packed
            dpTable[subProblemCapacity] = max(dpTable[subProblemCapacity - weight] + profit,
                                                   dpTable[subProblemCapacity])
    return dpTable

def DP3(itemSet, capacity):
    solution = []
    solutionProfit = 0
    numItems = len(itemSet)
    iterationCapacity = capacity
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

