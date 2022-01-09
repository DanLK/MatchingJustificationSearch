# ----------------------------------------------------------------------------- #
# Analysis of One-to-One Matching Mechanisms via SAT Solving
# Ulle Endriss, ILLC, University of Amsterdam, November 2019
# ----------------------------------------------------------------------------- #

# Python code to generate formulas in CNF that encode axiomatic properties of
# matching mechanisms, as used to derive some of the results in this paper:
# 
# Ulle Endriss. "Analysis of One-to-One Matching Mechanisms via SAT Solving:
# Impossibilities for Universal Axioms". In Proceedings of the 34th AAAI
# Conference on Artificial Intelligence (AAAI-2020). Association for the
# Advancement of Artificial Intelligence, 2020.
#
# Refer to the file results.txt for examples of usage.
# Tested using Python 3.7.0.

# ----------------------------------------------------------------------------- #

# Need to match two groups of n agents each (n of type 0 and n of type 1).
# Agents in each groups are referred to by indices ranging from 0 to n-1.
# Preference orders over n agents are represented as numbers from 0 to n!-1.
# Profiles (vectors of preferences for 2n agents) are represented as numbers
# from 0 to (n!)^(2n)-1. To understand certain details in the code, think of
# profiles as numbers with 2n digits in the base-n! number system, with each
# digit representing the preference of one agent.

# ----------------------------------------------------------------------------- #

from math import factorial
from itertools import permutations

# --- MARKET DIMENSION -------------------------------------------------------- #

# default value for the dimension (number of agents per group)
n = 3

# change the value of n (the dimension), with argument 2 or 3 being recommended
def setDimension(k):
    global n
    n = k

# retrieve the current value of n (the dimension)
def getDimension():
    return n 

# --- BASIC BUILDING BLOCKS --------------------------------------------------- #

# return range of indices to refer to agents in each of the two groups
def allIndices():
    return range(n)

# return range of numbers representing possible preference relations 
def allPreferences():
    return range(factorial(n))

# return range of numbers representing possible profiles
def allProfiles():
    return range(factorial(n) ** (2*n))

# return list of indices that satisfy a given condition
def indices(condition):
    return [x for x in allIndices() if condition(x)]

# return list of preferences that satisfy a given condition
def preferences(condition):
    return [x for x in allPreferences() if condition(x)]

# return list of profiles that satisfy a given condition
def profiles(condition):
    return [x for x in allProfiles() if condition(x)]

# --- REASONING ABOUT PREFERENCES --------------------------------------------- #

# extract preference (as ID) of ith agent of type t from profile p
def prefid(t, i, p):
    base = factorial(n)
    exp = t*n + i
    return ( p % (base ** (exp+1)) ) // (base ** exp)
    
# extract preference (as list) of ith agent of type t from profile p
def preflist(t, i, p):
    preflists = list(permutations(allIndices()))
    return preflists[prefid(t,i,p)]

# check whether ith agent of type t prefers j1 to j2 in profile p
def prefers(t, i, j1, j2, p):
    mylist = preflist(t, i, p)
    return mylist.index(j1) < mylist.index(j2)

# check whether ith agent of type t has j at the top of her preference 
def top(t, i, j, p):
    mylist = preflist(t, i, p)
    return mylist[0] == j

# --- OPERATIONS ON PROFILES -------------------------------------------------- #

# return list of all variants of profile p with ith agent of type t deviating 
def iVariants(t, i, p):
    currpref = prefid(t, i, p)
    factor = factorial(n) ** (t*n + i)
    rest = p - currpref * factor
    variants = []
    for newpref in preferences(lambda newpref : newpref != currpref):
        variants.append(rest + newpref * factor)
    return variants    

# return result of swapping groups ("genders") in profile p
def swapGroups(p):
    base = factorial(n) ** n
    lastdigit = p % base
    return lastdigit * base + (p - lastdigit) // base

# return result of swapping agents indexed by i1 and i2 of type 0 in profile p1
def swapLeftAgents(p1, i1, i2):
    base = factorial(n)
    p2 = p1 + (prefid(0,i2,p1) - prefid(0,i1,p1)) * (base ** i1)
    p2 = p2 + (prefid(0,i1,p1) - prefid(0,i2,p1)) * (base ** i2)
    preflists = list(permutations(allIndices()))
    for j in allIndices():
        mylist = list(preflist(1,j,p1))
        idx1 = mylist.index(i1)
        idx2 = mylist.index(i2)
        mylist[idx1], mylist[idx2] = mylist[idx2], mylist[idx1]
        newprefid = preflists.index(tuple(mylist))
        p2 = p2 + (newprefid - prefid(1,j,p1)) * (base ** (n+j))
    return p2

# return result of swapping agents indexed by j1 and j2 of type 1 in profile p1
def swapRightAgents(p1, j1, j2):
    base = factorial(n)
    p2 = p1 + (prefid(1,j2,p1) - prefid(1,j1,p1)) * (base ** (n+j1))
    p2 = p2 + (prefid(1,j1,p1) - prefid(1,j2,p1)) * (base ** (n+j2))
    preflists = list(permutations(allIndices()))
    for i in allIndices():
        mylist = list(preflist(0,i,p1))
        idx1 = mylist.index(j1)
        idx2 = mylist.index(j2)
        mylist[idx1], mylist[idx2] = mylist[idx2], mylist[idx1]
        newprefid = preflists.index(tuple(mylist))
        p2 = p2 + (newprefid - prefid(0,i,p1)) * (base ** i)
    return p2

# --- FORMULAS TO CHARACTERISE MECHANISMS ------------------------------------- #

# return positive literal to say i and j are matched in profile p
def posLiteral(p, i, j):
    return p * (n * n) + (i * n) + j  + 1

# return negative literal to say i and j are not matched in profile p
def negLiteral(p, i, j):
    return (-1) * posLiteral(p, i, j)

# return CNF to say that any outcome returned by a mechanism must be such that
# every i has at least one successor and every j has at most one predecessor
def cnfMechanism():
    cnf = []
    for p in allProfiles():
        for i in allIndices():
            cnf.append([posLiteral(p,i,j) for j in allIndices()])
        for j in allIndices():
            for i1 in allIndices():
                for i2 in indices(lambda i2 : i1 < i2):
                    cnf.append([negLiteral(p,i1,j), negLiteral(p,i2,j)])
    return cnf
      
# save the given CNF in the given text file (using the DIMACS format)
def saveCNF(cnf, filename):
    nvars = (factorial(n) ** (2 * n)) * (n ** 2)
    nclauses = len(cnf)
    file = open(filename, 'w')
    file.write('p cnf ' + str(nvars) + ' ' + str(nclauses) + '\n')
    for clause in cnf:
        file.write(' '.join([str(literal) for literal in clause]) + ' 0\n')
    file.close()

# print human-interpretable representation of the given variable (a number)
def interpretVariable(x):
    j = (x - 1) % n
    i = ( (x - j - 1) % (n * n) ) // n
    p = ( (x - n * i - j - 1) ) // (n * n)
    print('-> in profile number ' + str(p) + ' match ' + str(i) + '/' + str(j))
    s = '( '
    for i in allIndices():
        s = s + '>'.join([str(x) for x in preflist(0,i,p)]) + ' '
    s = s + '| '
    for j in allIndices():
        s = s + '>'.join([str(x) for x in preflist(1,j,p)]) + ' '
    s = s + ')'
    print('-> where profile ' + str(p) + ' = ' + s)
    
# pretty-print mechanism (given as list of literals) on screen (useful for n=2)
def printMechanism(m):
    for p in allProfiles():
        s = '( '
        for i in allIndices():
            s = s + '>'.join([str(x) for x in preflist(0,i,p)]) + ' '
        s = s + '| '
        for j in allIndices():
            s = s + '>'.join([str(x) for x in preflist(1,j,p)]) + ' '
        s = s + ') --> { '
        for i in allIndices():
            for j in indices(lambda j : posLiteral(p,i,j) in m):
                s = s + str(i) + str(j) + ' '
        s = s + '}'
        print(s)   
    
# --- AXIOMS ------------------------------------------------------------------ #

# return CNF encoding the axiom of top-stability
def cnfTopStable():
    cnf = []
    for p in allProfiles():
        for i in allIndices():
            for j in indices(lambda j : top(0,i,j,p) and top(1,j,i,p)):
                cnf.append([posLiteral(p,i,j)])
    return cnf

# return CNF encoding the axiom of stability
def cnfStable():
    cnf = []
    for p in allProfiles():
        for i1 in allIndices():
            for j1 in allIndices():
                for i2 in indices(lambda i2 : prefers(1,j1,i1,i2,p)):
                    for j2 in indices(lambda j2 : prefers(0,i1,j1,j2,p)):
                        cnf.append([negLiteral(p,i1,j2), negLiteral(p,i2,j1)])
    return cnf                        

# return CNF encoding the axiom of strategyproofness for agents of type 0
def cnfLeftStrategyProof():
    cnf = []
    for i in allIndices():
        for p1 in allProfiles():
            for p2 in iVariants(0, i, p1):
                for j1 in allIndices():
                    for j2 in indices(lambda j2 : prefers(0,i,j2,j1,p1)):
                        cnf.append([negLiteral(p1,i,j1), negLiteral(p2,i,j2)])
    return cnf                        

# return CNF encoding the axiom of strategyproofness for agents of type 1
def cnfRightStrategyProof():
    cnf = []
    for j in allIndices():
        for p1 in allProfiles():
            for p2 in iVariants(1, j, p1):
                for i1 in allIndices():
                    for i2 in indices(lambda i2 : prefers(1,j,i2,i1,p1)):
                        cnf.append([negLiteral(p1,i1,j), negLiteral(p2,i2,j)])
    return cnf                        

# return CNF encoding the axiom of two-way strategyproofness
def cnfTwoWayStrategyProof():
    return cnfLeftStrategyProof() + cnfRightStrategyProof()

# retun CNF encoding the axiom of gender-indifference
def cnfGenderIndifferent():
    cnf = []
    for p1 in allProfiles():
        p2 = swapGroups(p1)
        for i in allIndices():
            for j in allIndices():
                cnf.append([negLiteral(p1,i,j), posLiteral(p2,j,i)])
    return cnf        

# return CNF encoding the axiom of peer-indifference for agents of type 0
def cnfLeftPeerIndifferent():
    cnf = []
    for p1 in allProfiles():
        for i1 in allIndices():
            for i2 in indices(lambda i2 : i1 != i2):
                p2 = swapLeftAgents(p1, i1, i2)
                for j in allIndices():
                    cnf.append([negLiteral(p1,i1,j), posLiteral(p2,i2,j)])
                    for i in indices(lambda i : i != i1 and i != i2):
                        cnf.append([negLiteral(p1,i,j), posLiteral(p2,i,j)])
    return cnf

# return CNF encoding the axiom of peer-indifference for agents of type 1
def cnfRightPeerIndifferent():
    cnf = []
    for p1 in allProfiles():
        for j1 in allIndices():
            for j2 in indices(lambda j2 : j1 != j2):
                p2 = swapRightAgents(p1, j1, j2)
                for i in allIndices():
                    cnf.append([negLiteral(p1,i,j1), posLiteral(p2,i,j2)])
                    for j in indices(lambda j : j != j1 and j != j2):
                        cnf.append([negLiteral(p1,i,j), posLiteral(p2,i,j)])
    return cnf

# return CNF encoding the axiom of peer-indifference (for both types of agents)
def cnfPeerIndifferent():
    return cnfLeftPeerIndifferent() + cnfRightPeerIndifferent()

# ----------------------------------------------------------------------------- #