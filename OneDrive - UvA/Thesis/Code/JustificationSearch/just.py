# Code for the search for justifications for partial matching outcomes

import matchsat
from math import factorial
from itertools import chain
from pylgl import solve
import time
import numpy as np
from matplotlib import pyplot as plt
import random
from matching.games import StableMarriage

# interpretVariable from matchsat.py modified
# print human-interpretable representation of the given variable (a number)
def interpretVariable(x):
    if x > 0:
        j = (x - 1) % matchsat.n
        i = ( (x - j - 1) % (matchsat.n * matchsat.n) ) // matchsat.n
        p = (( (x - matchsat.n * i - j - 1) ) // (matchsat.n * matchsat.n)) % (factorial(matchsat.n) ** (2*matchsat.n))
        print('-> in profile number ' + str(p) + ' match ' + str(i) + '/' + str(j))
        s = '( '
        for i in matchsat.allIndices():
            s = s + '>'.join([str(x) for x in matchsat.preflist(0,i,p)]) + ' '
        s = s + '| '
        for j in matchsat.allIndices():
            s = s + '>'.join([str(x) for x in matchsat.preflist(1,j,p)]) + ' '
        s = s + ')'
        print('-> where profile ' + str(p) + ' = ' + s)
    else:
        y = -x
        j = (y - 1) % matchsat.n
        i = ( (y - j - 1) % (matchsat.n * matchsat.n) ) // matchsat.n
        p = (( (y - matchsat.n * i - j - 1) ) // (matchsat.n * matchsat.n)) % (factorial(matchsat.n) ** (2*matchsat.n))
        print('-> in profile number ' + str(p) + ' dont match ' + str(i) + '/' + str(j))
        s = '( '
        for i in matchsat.allIndices():
            s = s + '>'.join([str(x) for x in matchsat.preflist(0,i,p)]) + ' '
        s = s + '| '
        for j in matchsat.allIndices():
            s = s + '>'.join([str(x) for x in matchsat.preflist(1,j,p)]) + ' '
        s = s + ')'
        print('-> where profile ' + str(p) + ' = ' + s)

# ------------------------------------------------------------------------------------------
# Renaming of the axiom (the code remains the same)
# Return CNF encoding the axiom of gender-indifference
def cnfGroupIndifferent():
    cnf = []
    for p1 in matchsat.allProfiles():
        p2 = matchsat.swapGroups(p1)
        for i in matchsat.allIndices():
            for j in matchsat.allIndices():
                cnf.append([matchsat.negLiteral(p1,i,j), matchsat.posLiteral(p2,j,i)])
    return cnf        



# --------------------------------------------------------------------------------------------- #
#Efficiency Axioms

#Return CNF encoding the axiom of at least one agent of type 0 being matched with her favorite

def cnfLeftTopRewarding():
    cnf = []
    for p in matchsat.allProfiles():
        cnf.append([matchsat.posLiteral(p,i,matchsat.preflist(0,i,p)[0]) for i in matchsat.allIndices()])
    return cnf

#Return CNF encoding the axiom of at least one agent of type 1 being matched with her favorite
def cnfRightTopRewarding():
    cnf = []
    for p in matchsat.allProfiles():
        cnf.append([matchsat.posLiteral(p,j,matchsat.preflist(1,j,p)[0]) for j in matchsat.allIndices()])
    return cnf

#Return CNF encoding the axiom of at least one agent on each side being matched with her favorite
def cnfTopRewarding():
    cnf = cnfLeftTopRewarding() + cnfRightTopRewarding()
    return cnf

# Return CNF encoding the axiom of left-swap-stability
def cnfLeftSwapStable():
    cnf = []
    for p in matchsat.allProfiles():
        for i1 in matchsat.allIndices():
            for i2 in matchsat.indices(lambda i2 : i2 < i1):
                for j1 in matchsat.allIndices():
                    for j2 in matchsat.indices(lambda j2 : matchsat.prefers(0,i1,j2,j1,p) and matchsat.prefers(0,i2,j1,j2,p)):
                        cnf.append([matchsat.negLiteral(p,i1,j1),matchsat.negLiteral(p,i2,j2)])
    return cnf

#Return CNF encoding the axiom of right-swap-stability
def cnfRightSwapStable():
    cnf = []
    for p in matchsat.allProfiles():
        for j1 in matchsat.allIndices():
            for j2 in matchsat.indices(lambda j2 : j2 < j1):
                for i1 in matchsat.allIndices():
                    for i2 in matchsat.indices(lambda i2 : matchsat.prefers(0,j1,i2,i1,p) and matchsat.prefers(1,j2,i1,i2,p)):
                        cnf.append([matchsat.negLiteral(p,i1,j1),matchsat.negLiteral(p,i2,j2)])
    return cnf

#Return CNF encoding the axiom of swap-stability

def cnfSwapStable():
    cnf = cnfLeftSwapStable() + cnfRightSwapStable()
    return cnf

#Return CNF encoding the axiom no-bottoms

def cnfNoBottoms():
    cnf = []
    for p in matchsat.allProfiles():
        for i in matchsat.allIndices():
            for j in matchsat.allIndices():
                if matchsat.preflist(0,i,p)[-1] == j and matchsat.preflist(1,j,p)[-1] == i:
                    cnf.append([matchsat.negLiteral(p,i,j)])
    return cnf



#--------------------------------------------------------------------------------------------------------------
# RETURN THE NUMBER OF A CERTAIN PROFILE
# Given the agents' preferences return the encoding of the profile as an integer
#For n = 2
def getProfileTwo(a,b,c,d):
    for p in matchsat.allProfiles():
        if matchsat.preflist(0,0,p) == a and matchsat.preflist(0,1,p) == b and matchsat.preflist(1,0,p) == c and matchsat.preflist(1,1,p) == d:
            return p

#For n = 3
def getProfileThree(a,b,c,d,e,f):
    for p in matchsat.allProfiles():
        if matchsat.preflist(0,0,p) == a and matchsat.preflist(0,1,p) == b and matchsat.preflist(0,2,p) == c and matchsat.preflist(1,0,p) == d and matchsat.preflist(1,1,p) == e and matchsat.preflist(1,2,p) == f:
            return p

#----------------------------------------------------------------------
#----------------- JUSTIFICATION VERIFICATION ------------------

#Returns True if a set of axioms(formulas) are a justification for the feature
#axioms is a list of axioms in DIMACS (without names)
def isJustification(axioms,nfeature): 
    '''Parameters:
    axioms: List of axioms (lists) in DIMACS format
    nfeature: The DIMACS encoding of the negation of the feature that is one clause in the format [clause]
    Returns:
    True: If the set of axioms and the goal are incompatible
    False: If there is a model for the axioms and goals'''
    cnf = list(chain(*axioms)) + matchsat.cnfMechanism()
    if solve(cnf) == 'UNSAT':
        return False
    else:
        cnf += nfeature 
        return solve(cnf) == 'UNSAT'
    



##------------------------------- INSTANCE CHECKER --------
## Given a list of axioms (with names) and a CNF, return the name(s) of the axiom of which the CNF is an instance
## What does it do if it's not an instance of any, then return some error message (or the empty list)

def instanceCheck(axioms, instance):
    r = []
    for axiom in axioms:
        if instance in axiom[1]:
            r.append(axiom[0])
    return r

# --------------------------- Justification SEARCH --------------    

#Function that obtains a list with lists that represent all the possible subsets of a list(set)
def subsets(s):
    """
    :type s: list[]
    """
    sets = []
    for i in range(2**len(s)):
        subset = []
        for j in range(len(s)):
            if i & (1 << j) > 0:
                subset.append(s[j])
        sets.append(subset)
    return sets

## Given a corpus of axioms (a list of axioms with names) and a the encoding of the negation of the feature
## Returns the first subset of axioms that is a justification for the feature and saves the CNF with those axioms in a file under the specified name

def basisSearch(axioms, nfeature, filename):
    search_space = subsets(axioms)
    search_space.remove([])
    for set_of_axioms in search_space:
        result = []
        cnf = matchsat.cnfMechanism()
        axioms = []
        for axiom in set_of_axioms:
            axioms += [axiom[1]]
            print(axiom[0])
        print('-----------')
        if isJustification(axioms,nfeature):
            for axiom in set_of_axioms:
                cnf += axiom[1] 
                result.append(axiom[0])
            cnf += nfeature 
            matchsat.saveCNF(cnf, filename)
            return result
    return 'There is no justification based on these axioms'

# Function that searches for a normative basis but without saving the formula in a file 

def basisSearch2(axioms, nfeature):
    search_space = subsets(axioms)
    search_space.remove([])
    for set_of_axioms in search_space:
        result = []
        axioms = []
        for axiom in set_of_axioms:
            axioms += [axiom[1]]
            print(axiom[0])
        print('-----------')
        if isJustification(axioms,nfeature):
            for axiom in set_of_axioms:
                result.append(axiom[0])
            return result
    return 'There is no justification based on these axioms'

# Function that finds all the possible normative bases among a set of axioms.

def allBasesSearch(axioms, nfeature):
    search_space = subsets(axioms)
    search_space.remove([])
    bases = []
    i = 1
    for set_of_axioms in search_space:
        result = []
        axioms = []
        print('-----------')
        print('Subset number: ',i)
        for axiom in set_of_axioms:
            axioms += [axiom[1]]
            print(axiom[0])
        i += 1
        if isJustification(axioms,nfeature):
            for axiom in set_of_axioms:
                result.append(axiom[0])
            bases.append(result)
            print(result, ' forms a basis')
            print('-----------')
    return bases

# ----------------------------------------------------------------------------------------------------------
# General statistics

# Plot the sizes of the axioms given in a list
def plotSizes(xs,ys):
    plt.plot(xs, ys,
             color="b",      # "r"=red, "b"=blue, "k"=black, "y"=yellow, ... (even shorter: c='r')
             linestyle="-",  # "--"=dashed, "-."=dash-dot, ":"=dotted, ... (even shorter: ls='-')
             marker="o",     # "o"=dot, "s"=square, "+"=cross, ...
             linewidth=0,    # (even shorter: lw=4)
             markersize=5,   # (even shorter: ms=10)
             label="Number of clauses")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of clauses (in millions)')
    plt.title('Number of clauses for axiom encodings when n=3')
    plt.show()

# Function to measure the time it takes to compute another function 
def timeToCompute(function):
    start = time.time()
    function()
    return time.time() - start

# -------------------------------------------------------------------------------------------
# DEFERRED ACCEPTANCE ALGORITHM
# -------------------------------------------------------------------------------------------

#Function that returns True iff the cnf is a well-defined matching outcome under profile "profile"
# This function works for any dimension
def isMatching(cnf, profile):
    left_conditions = []
    right_condition = True
    for i in matchsat.allIndices():
        for j in matchsat.allIndices():
            if [matchsat.posLiteral(profile,i,j)] in cnf:
                left_conditions.append(True)
    for j in matchsat.allIndices():
        for i1 in matchsat.allIndices():
            for i2 in matchsat.indices(lambda i2 : i1 < i2):
                if ([matchsat.posLiteral(profile, i1,j)] in cnf) and ([matchsat.posLiteral(profile,i2,j)] in cnf):
                    right_condition = False
    left_cond = len(left_conditions) == matchsat.getDimension() and all(left_conditions) 
    return left_cond and right_condition

def runLeftDA(p):
    left_agents_prefs = {0: list(matchsat.preflist(0,0,p)), 1: list(matchsat.preflist(0,1,p)),2:list(matchsat.preflist(0,2,p))}
    right_agents_prefs = {0: list(matchsat.preflist(1,0,p)), 1: list(matchsat.preflist(1,1,p)),2:list(matchsat.preflist(1,2,p))}
    game = StableMarriage.create_from_dictionaries(left_agents_prefs,right_agents_prefs)
    matching_dict = game.solve()
    matching_list = []
    # Translate back the matching into the correct format
    left_agents = list(matching_dict.keys())
    right_agents = [matching_dict[l] for l in left_agents]
    for i in range(len(left_agents)):
        matching_list.append([matchsat.posLiteral(p,int(str(left_agents[i])),int(str(right_agents[i])))])
    return matching_list 

# ------------------------------------------------------------------------
# E X P E R I M E N T
# ------------------------------------------------------------------------

# Randomly draw a profile
def drawProfile():
    return random.choice(matchsat.allProfiles())

def experiment(iterations,axioms):
    just_found = []
    info = []
    times = []
    for i in range(iterations):
        print('Iteration #', i+1, ':')
        p = drawProfile()
        matching = runLeftDA(p) # Observe that the basic features correspond to the elements of the matching
        print('The matching is: ', matching )
        profile_info = []
        profile_just_found = []
        profile_times = []
        for feature in matching:
            print('Checking feature ', feature )
            start = time.time()
            y = basisSearch2(axioms,[[-feature[0]]])
            profile_times.append(time.time() - start)
            print(y)
            if y != 'There is no justification based on these axioms':
                profile_just_found.append(1)
                profile_info.append([feature,y])
            else:
                profile_just_found.append(0)
                profile_info.append([feature,['N/A']])
        info.append([p, profile_info])
        just_found.append([p, profile_just_found])
        times.append([p,profile_times])
        print('--------------------------------')
    return info, just_found, times

# ---------------------------------------------------------------------
# Plot experiments results
# ---------------------------------------------------------------------

def plotNumFeaturesJustified(info,justifications):
    profs = [str(y[0]) for y in info]
    number_found = [sum(y[1]) for y in justifications]

    fig,ax = plt.subplots()
    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.set_xlabel('Profiles')
    ax.set_ylabel('Number of features')
    ax.set_title('Number of features that could be justified for each profile')
    plt.bar(profs, number_found)
    plt.xticks(rotation=70, ha='right')
    plt.yticks(np.arange(0,4,1))

    plt.show()


def plotTimes(justs, times):
    justs_minus_profiles = [y[1] for y in justs]
    jmp = list(chain(*justs_minus_profiles))
    features = list(chain(*[y[0] for y in jmp]))
    times_minus_profiles = [y[1] for y in times]
    comp_times = list(chain(*times_minus_profiles))
    plt.plot(features, comp_times,
             color="b",      # "r"=red, "b"=blue, "k"=black, "y"=yellow, ... (even shorter: c='r')
             linestyle="-",  # "--"=dashed, "-."=dash-dot, ":"=dotted, ... (even shorter: ls='-')
             marker="o",     # "o"=dot, "s"=square, "+"=cross, ...
             linewidth=0,    # (even shorter: lw=4)
             markersize=2,   # (even shorter: ms=10)
             )
    plt.title('Computation times for basis searches')
    plt.xlabel('Integers representing the encodings of the features' )
    plt.ylabel('Time (in seconds)')
    plt.xticks(rotation=70, ha='right')

    plt.show()


if __name__=="__main__":
    matchsat.setDimension(3)
# --------------------------------------------------------------------
# Uncomment to run experiments
# --------------------------------------------------------------------
    # start = time.time()
    # print('--------------------------------')
    # axioms = [['LSP',matchsat.cnfLeftStrategyProof()], ['RF', cnfRightFavorite()], ['NOBOT', cnfNoBottoms()], ['TOPSTA', matchsat.cnfTopStable()],['STA', matchsat.cnfStable()],['LSS', cnfLeftSwapStable()], ['PI',matchsat.cnfPeerIndifferent()]]
    # print('------------', time.time()-start, ' seconds to compute the axioms -------------------')
    # new_start = time.time()
    # print('----------- Starting experiments ------------------')
    # x,y,z = experiment(100,axioms)
    # print('------------ End of experiments -------------------')
    # print('---------- ', time.time()-new_start, ' seconds to run the experiments')
    # print('Info = ', y)
    # print('Just found = ',x)
    # print('Times = ', z)
    # plotNumFeaturesJustified(x,y)


# --------------------------------------------------------------------
# Uncomment to compute a list with all the axioms
# --------------------------------------------------------------------
    #all_axioms = [['STA', matchsat.cnfStable()],['TOPSTA',matchsat.cnfTopStable()],['LSP',matchsat.cnfLeftStrategyProof()],['RSP',matchsat.cnfRightStrategyProof()], ['SP', matchsat.cnfTwoWayStrategyProof()],['GI', cnfGroupIndifferent()],['PI',matchsat.cnfPeerIndifferent()],['LF',cnfLeftFavorite()],['RF',cnfRightFavorite()],['FAV',cnfFavorite()],['LSWSTA', cnfLeftSwapStable()], ['RSWSTA',cnfRightSwapStable()], ['SWSTA',cnfSwapStable()],['NOBOT',cnfNoBottoms()]]
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Uncomment this section to obtain all the possible normative bases for the situation in example 3.3
# --------------------------------------------------------------------
    # start = time.time()
    # axioms = [['STA', matchsat.cnfStable()],['TOPSTA',matchsat.cnfTopStable()],['LSP',matchsat.cnfLeftStrategyProof()],['RSP',matchsat.cnfRightStrategyProof()],['GI', cnfGroupIndifferent()],['PI',matchsat.cnfPeerIndifferent()],['NOBOT',cnfNoBottoms()]]
    # feature = [[matchsat.posLiteral(24378,0,2)]]
    # print("--- %s seconds to compute the axioms ---" % (time.time() - start))
    # print('---------------------------------------------------------------------------')
    # new_start = time.time()
    # x = allBasisSearch(axioms,feature)
    # print(x)
    # print("--- %s seconds to compute the bases ---" % (time.time() - new_start))
    # print('---------------------------------------------------------------------------')
# ------------------------------------------------------------


# ------------------------------------------------------------
# Uncomment this section to obtain the number of clauses of the axiom encodings (with graphs)
# ------------------------------------------------------------
    # cnf = []
    # len_formulas = []
    # names = []
    # for axiom in all_axioms:
    #     cnf += axiom[1]
    #     len_formulas += [len(axiom[1])]
    #     names += [axiom[0]]
    # print("The formula has ", len(cnf), " clauses")
    # print("names length is ", len(names))
    # print("len_formulas length is ", len(len_formulas))
    # print('names = ', names)
    # print('len_formulas = ', len_formulas)
    # plotSizes(names,len_formulas)
# ------------------------------------------------------------


# ------------------------------------------------------------
# Uncomment this section to see the time it takes to compute the axiom encodings
# ------------------------------------------------------------
    # function_names = [matchsat.cnfMechanism,matchsat.cnfStable,matchsat.cnfTopStable,matchsat.cnfLeftStrategyProof,matchsat.cnfRightStrategyProof,matchsat.cnfTwoWayStrategyProof,cnfGroupIndifferent,matchsat.cnfPeerIndifferent,cnfLeftFavorite,cnfRightFavorite,cnfFavorite,cnfLeftSwapStable,cnfRightSwapStable,cnfSwapStable,cnfNoBottoms]
    # times = []
    # times_and_names = []
    # for function in function_names:
    #     x = timeToCompute(function)
    #     times_and_names.append([str(function),x])
    #     times += [x]
    # print(times)
    # sum = 0
    # for time in times:
    #     sum += time
    # average = sum/len(times)
    # print('---The average computation time of the axioms is ', average, ' seconds ----------')
    # print('---The total computation time of the axioms is ', sum, ' seconds ----------')
# ------------------------------------------------------------

# --------------------------------------------------------------------
# Results of the 100 runs reported in the thesis 
# Time to compute all of them: 34392.769872665405 = 9.5 hours
# --------------------------------------------------------------------

    info2 = [[45147, [1, 1, 1]], [35529, [0, 1, 0]], [19405, [1, 1, 0]], [21400, [1, 1, 1]], [9078, [1, 1, 1]], [18513, [1, 1, 1]], [12547, [1, 1, 1]], [40330, [1, 1, 1]], [16217, [1, 1, 1]], [14633, [1, 1, 1]], [2498, [1, 1, 1]], [34453, [1, 1, 1]], [7369, [1, 1, 1]], [12701, [1, 1, 1]], [36873, [1, 1, 1]], [17681, [1, 1, 1]], [15311, [1, 1, 1]], [31956, [1, 1, 1]], [44102, [1, 1, 1]], [45811, [1, 1, 1]], [32890, [1, 1, 1]], [25133, [1, 1, 1]], [39000, [1, 1, 1]], [18123, [1, 1, 1]], [14894, [1, 1, 1]], [30315, [1, 1, 1]], [42637, [1, 1, 1]], [46533, [1, 1, 1]], [11747, [1, 1, 1]], [21022, [1, 1, 1]], [45557, [1, 1, 1]], [1559, [1, 1, 1]], [24379, [0, 1, 0]], [12996, [1, 1, 1]], [4525, [1, 1, 1]], [41581, [1, 1, 1]], [34167, [1, 1, 1]], [40509, [1, 1, 1]], [20722, [1, 1, 1]], [26830, [1, 1, 1]], [24968, [1, 1, 1]], [5841, [1, 1, 1]], [17245, [1, 1, 1]], [46025, [1, 1, 1]], [13672, [1, 1, 1]], [37152, [1, 1, 1]], [17242, [1, 1, 1]], [41849, [1, 1, 1]], [35795, [1, 1, 1]], [26305, [1, 1, 1]], [33472, [1, 1, 1]], [5474, [1, 1, 1]], [15027, [1, 1, 1]], [35653, [1, 1, 1]], [25150, [1, 1, 1]], [8438, [1, 1, 1]], [45757, [1, 1, 1]], [5666, [0, 1, 0]], [27489, [1, 1, 1]], [29198, [1, 1, 1]], [3369, [1, 1, 1]], [19149, [1, 1, 1]], [41305, [1, 1, 1]], [32971, [1, 1, 1]], [1180, [1, 1, 1]], [41614, [1, 1, 1]], [13028, [1, 1, 1]], [8778, [1, 1, 1]], [6964, [1, 1, 1]], [31520, [1, 1, 1]], [44677, [1, 1, 1]], [16849, [1, 1, 1]], [46508, [1, 1, 1]], [20379, [1, 1, 1]], [20910, [1, 1, 1]], [41382, [1, 1, 1]], [2189, [1, 1, 1]], [10501, [1, 1, 1]], [37426, [1, 1, 1]], [13416, [1, 1, 1]], [10485, [1, 1, 1]], [15139, [1, 1, 1]], [19561, [1, 1, 1]], [9339, [1, 1, 1]], [20053, [1, 1, 1]], [41124, [1, 1, 1]], [2940, [1, 1, 1]], [19103, [1, 1, 1]], [3222, [1, 1, 1]], [7189, [1, 1, 1]], [1358, [1, 1, 1]], [11205, [1, 1, 1]], [9514, [1, 1, 1]], [37597, [1, 1, 1]], [21723, [1, 1, 1]], [43325, [1, 1, 1]], [26905, [1, 1, 1]], [35860, [1, 1, 1]], [5571, [1, 1, 1]], [1807, [1, 1, 1]]]
    justs2 = [[45147, [[[406325], ['STA']], [[406329], ['STA']], [[406330], ['TOPSTA']]]], [35529, [[[319764], ['N/A']], [[319766], ['STA']], [[319768], ['N/A']]]], [19405, [[[174648], ['STA']], [[174649], ['LSP', 'TOPSTA', 'LSS']], [[174653], ['N/A']]]], [21400, [[[192603], ['LSP', 'TOPSTA']], [[192605], ['LSS']], [[192607], ['LSS']]]], [9078, [[[81703], ['TOPSTA']], [[81708], ['STA']], [[81710], ['STA']]]], [18513, [[[166619], ['STA']], [[166621], ['LSS']], [[166626], ['LSS']]]], [12547, [[[112926], ['TOPSTA']], [[112928], ['TOPSTA']], [[112930], ['TOPSTA']]]], [40330, [[[362971], ['LSP', 'TOPSTA']], [[362975], ['LSP', 'TOPSTA']], [[362979], ['TOPSTA']]]], [16217, [[[145956], ['STA']], [[145958], ['STA']], [[145960], ['STA']]]], [14633, [[[131700], ['TOPSTA']], [[131702], ['STA']], [[131704], ['STA']]]], [2498, [[[22484], ['TOPSTA']], [[22486], ['TOPSTA', 'LSS']], [[22491], ['TOPSTA', 'LSS']]]], [34453, [[[310080], ['TOPSTA', 'LSS']], [[310081], ['TOPSTA']], [[310085], ['TOPSTA', 'LSS']]]], [7369, [[[66324], ['LSP', 'TOPSTA']], [[66326], ['LSP', 'TOPSTA']], [[66328], ['TOPSTA']]]], [12701, [[[114312], ['TOPSTA']], [[114314], ['LSP', 'TOPSTA']], [[114316], ['LSP', 'TOPSTA']]]], [36873, [[[331859], ['TOPSTA']], [[331861], ['TOPSTA']], [[331866], ['TOPSTA']]]], [17681, [[[159132], ['STA']], [[159133], ['TOPSTA']], [[159137], ['STA']]]], [15311, [[[137802], ['TOPSTA']], [[137803], ['LSP', 'TOPSTA']], [[137807], ['LSP', 'TOPSTA']]]], [31956, [[[287606], ['LSP', 'TOPSTA']], [[287608], ['LSP', 'TOPSTA']], [[287613], ['TOPSTA']]]], [44102, [[[396920], ['STA']], [[396922], ['STA']], [[396927], ['STA']]]], [45811, [[[412300], ['STA']], [[412305], ['STA']], [[412307], ['STA']]]], [32890, [[[296013], ['LSS']], [[296015], ['LSS']], [[296017], ['LSS']]]], [25133, [[[226200], ['LSS']], [[226201], ['TOPSTA']], [[226205], ['LSS']]]], [39000, [[[351001], ['TOPSTA']], [[351005], ['STA']], [[351009], ['STA']]]], [18123, [[[163109], ['TOPSTA']], [[163111], ['TOPSTA', 'LSS']], [[163116], ['TOPSTA', 'LSS']]]], [14894, [[[134048], ['LSP', 'TOPSTA', 'LSS']], [[134050], ['STA']], [[134055], ['LSP', 'TOPSTA', 'LSS']]]], [30315, [[[272838], ['TOPSTA']], [[272839], ['TOPSTA']], [[272843], ['TOPSTA']]]], [42637, [[[383736], ['LSP', 'TOPSTA']], [[383738], ['TOPSTA']], [[383740], ['LSP', 'TOPSTA']]]], [46533, [[[418798], ['STA']], [[418803], ['STA']], [[418805], ['TOPSTA']]]], [11747, [[[105726], ['TOPSTA']], [[105727], ['STA']], [[105731], ['STA']]]], [21022, [[[189199], ['LSP', 'TOPSTA']], [[189204], ['TOPSTA']], [[189206], ['LSP', 'TOPSTA']]]], [45557, [[[410014], ['STA']], [[410018], ['STA']], [[410022], ['TOPSTA']]]], [1559, [[[14034], ['TOPSTA']], [[14036], ['STA']], [[14038], ['STA']]]], [24379, [[[219412], ['N/A']], [[219417], ['STA']], [[219419], ['N/A']]]], [12996, [[[116965], ['TOPSTA']], [[116969], ['STA']], [[116973], ['STA']]]], [4525, [[[40726], ['LSP', 'STA']], [[40731], ['LSP', 'STA']], [[40733], ['LSP', 'TOPSTA']]]], [41581, [[[374230], ['TOPSTA']], [[374234], ['LSP', 'TOPSTA']], [[374238], ['LSP', 'TOPSTA']]]], [34167, [[[307505], ['LSP', 'TOPSTA']], [[307507], ['TOPSTA']], [[307512], ['LSP', 'TOPSTA']]]], [40509, [[[364583], ['TOPSTA']], [[364585], ['LSP', 'TOPSTA']], [[364590], ['LSP', 'TOPSTA']]]], [20722, [[[186501], ['STA']], [[186503], ['TOPSTA']], [[186505], ['STA']]]], [26830, [[[241472], ['LSP', 'TOPSTA']], [[241476], ['LSP', 'TOPSTA']], [[241477], ['TOPSTA']]]], [24968, [[[224714], ['TOPSTA']], [[224718], ['LSP', 'TOPSTA']], [[224719], ['LSP', 'TOPSTA']]]], [5841, [[[52572], ['LSP', 'TOPSTA']], [[52573], ['TOPSTA']], [[52577], ['LSP', 'TOPSTA']]]], [17245, [[[155206], ['TOPSTA']], [[155210], ['TOPSTA', 'LSS']], [[155214], ['TOPSTA', 'LSS']]]], [46025, [[[414228], ['LSP', 'TOPSTA']], [[414230], ['LSP', 'STA']], [[414232], ['LSP', 'STA']]]], [13672, [[[123051], ['TOPSTA']], [[123052], ['LSP', 'TOPSTA']], [[123056], ['LSP', 'TOPSTA']]]], [37152, [[[334370], ['STA']], [[334374], ['STA']], [[334375], ['TOPSTA']]]], [17242, [[[155179], ['LSP', 'TOPSTA']], [[155184], ['TOPSTA']], [[155186], ['LSP', 'TOPSTA']]]], [41849, [[[376642], ['TOPSTA']], [[376646], ['TOPSTA']], [[376650], ['TOPSTA']]]], [35795, [[[322157], ['TOPSTA']], [[322159], ['TOPSTA']], [[322164], ['TOPSTA']]]], [26305, [[[236746], ['TOPSTA']], [[236751], ['TOPSTA']], [[236753], ['TOPSTA']]]], [33472, [[[301249], ['STA']], [[301253], ['STA']], [[301257], ['TOPSTA']]]], [5474, [[[49267], ['LSP', 'TOPSTA']], [[49272], ['LSP', 'TOPSTA']], [[49274], ['TOPSTA']]]], [15027, [[[135246], ['LSP', 'TOPSTA']], [[135247], ['LSP', 'TOPSTA']], [[135251], ['TOPSTA']]]], [35653, [[[320880], ['STA']], [[320882], ['TOPSTA']], [[320884], ['STA']]]], [25150, [[[226351], ['STA']], [[226356], ['STA']], [[226358], ['STA']]]], [8438, [[[75944], ['TOPSTA']], [[75946], ['LSP', 'TOPSTA']], [[75951], ['LSP', 'TOPSTA']]]], [45757, [[[411814], ['TOPSTA']], [[411818], ['TOPSTA']], [[411822], ['TOPSTA']]]], [5666, [[[50996], ['N/A']], [[50998], ['STA']], [[51003], ['N/A']]]], [27489, [[[247404], ['TOPSTA', 'LSS']], [[247406], ['TOPSTA']], [[247408], ['LSS']]]], [29198, [[[262784], ['STA']], [[262786], ['TOPSTA']], [[262791], ['STA']]]], [3369, [[[30324], ['LSP', 'TOPSTA']], [[30326], ['TOPSTA']], [[30328], ['LSP', 'TOPSTA']]]], [19149, [[[172343], ['STA']], [[172347], ['TOPSTA']], [[172348], ['STA']]]], [41305, [[[371748], ['TOPSTA', 'LSS']], [[371750], ['LSS']], [[371752], ['TOPSTA']]]], [32971, [[[296740], ['LSS']], [[296745], ['LSS']], [[296747], ['LSS']]]], [1180, [[[10623], ['TOPSTA']], [[10624], ['TOPSTA', 'LSS']], [[10628], ['LSS']]]], [41614, [[[374527], ['LSP', 'TOPSTA']], [[374532], ['LSP', 'TOPSTA', 'LSS']], [[374534], ['NOBOT', 'LSS']]]], [13028, [[[117254], ['STA']], [[117258], ['LSP', 'LSS']], [[117259], ['LSP', 'LSS']]]], [8778, [[[79003], ['STA']], [[79008], ['LSS']], [[79010], ['LSS']]]], [6964, [[[62679], ['TOPSTA']], [[62681], ['LSS']], [[62683], ['LSS']]]], [31520, [[[283682], ['TOPSTA']], [[283684], ['TOPSTA']], [[283689], ['TOPSTA']]]], [44677, [[[402095], ['TOPSTA']], [[402097], ['TOPSTA']], [[402102], ['TOPSTA']]]], [16849, [[[151642], ['TOPSTA']], [[151647], ['STA']], [[151649], ['STA']]]], [46508, [[[418574], ['STA']], [[418578], ['STA']], [[418579], ['TOPSTA']]]], [20379, [[[183414], ['LSP', 'TOPSTA']], [[183415], ['LSP', 'TOPSTA', 'LSS']], [[183419], ['LSP', 'TOPSTA', 'LSS']]]], [20910, [[[188191], ['TOPSTA']], [[188196], ['TOPSTA']], [[188198], ['TOPSTA']]]], [41382, [[[372439], ['LSS']], [[372444], ['LSP', 'TOPSTA', 'LSS']], [[372446], ['LSP', 'TOPSTA', 'LSS']]]], [2189, [[[19704], ['TOPSTA']], [[19706], ['TOPSTA']], [[19708], ['TOPSTA']]]], [10501, [[[94510], ['TOPSTA']], [[94515], ['LSS']], [[94517], ['LSS']]]], [37426, [[[336837], ['STA']], [[336839], ['STA']], [[336841], ['TOPSTA']]]], [13416, [[[120745], ['LSP', 'TOPSTA', 'LSS']], [[120750], ['LSS']], [[120752], ['STA']]]], [10485, [[[94367], ['STA']], [[94369], ['STA']], [[94374], ['STA']]]], [15139, [[[136254], ['LSP', 'TOPSTA']], [[136256], ['LSP', 'TOPSTA']], [[136258], ['TOPSTA']]]], [19561, [[[176050], ['TOPSTA']], [[176054], ['TOPSTA']], [[176058], ['TOPSTA']]]], [9339, [[[84053], ['TOPSTA']], [[84057], ['STA']], [[84058], ['STA']]]], [20053, [[[180480], ['STA']], [[180481], ['TOPSTA']], [[180485], ['STA']]]], [41124, [[[370117], ['LSP', 'STA']], [[370122], ['LSP', 'TOPSTA']], [[370124], ['LSP', 'STA']]]], [2940, [[[26461], ['TOPSTA']], [[26466], ['STA']], [[26468], ['STA']]]], [19103, [[[171930], ['LSP', 'TOPSTA']], [[171932], ['TOPSTA']], [[171934], ['LSP', 'TOPSTA']]]], [3222, [[[28999], ['STA']], [[29003], ['TOPSTA']], [[29007], ['STA']]]], [7189, [[[64704], ['STA']], [[64705], ['STA']], [[64709], ['LSP', 'TOPSTA']]]], [1358, [[[12224], ['TOPSTA']], [[12228], ['STA']], [[12229], ['STA']]]], [11205, [[[100847], ['STA']], [[100849], ['TOPSTA']], [[100854], ['STA']]]], [9514, [[[85629], ['TOPSTA']], [[85630], ['TOPSTA']], [[85634], ['TOPSTA']]]], [37597, [[[338374], ['TOPSTA']], [[338379], ['LSP', 'TOPSTA']], [[338381], ['LSP', 'TOPSTA']]]], [21723, [[[195510], ['TOPSTA', 'LSS']], [[195511], ['TOPSTA', 'LSS']], [[195515], ['TOPSTA']]]], [43325, [[[389926], ['LSP', 'TOPSTA']], [[389930], ['TOPSTA']], [[389934], ['LSP', 'TOPSTA']]]], [26905, [[[242146], ['STA']], [[242150], ['TOPSTA']], [[242154], ['STA']]]], [35860, [[[322743], ['LSP', 'TOPSTA']], [[322745], ['LSP', 'TOPSTA']], [[322747], ['TOPSTA']]]], [5571, [[[50141], ['STA']], [[50145], ['STA']], [[50146], ['STA']]]], [1807, [[[16266], ['LSP', 'TOPSTA']], [[16267], ['TOPSTA']], [[16271], ['LSP', 'TOPSTA']]]]]
    times2 = [[45147, [119.64185214042664, 110.93364024162292, 68.23804712295532]], [35529, [669.6490581035614, 94.81695008277893, 709.502946138382]], [19405, [106.59244203567505, 210.88771390914917, 617.8764998912811]], [21400, [67.54768800735474, 150.69524002075195, 149.80915093421936]], [9078, [64.75260806083679, 90.88503193855286, 91.6175971031189]], [18513, [91.40050196647644, 150.97673797607422, 146.99971985816956]], [12547, [84.05125880241394, 71.6084098815918, 67.95906090736389]], [40330, [81.5237193107605, 83.19979286193848, 71.66498875617981]], [16217, [89.69657278060913, 116.82347893714905, 110.10502219200134]], [14633, [66.1332540512085, 90.99265098571777, 94.00083923339844]], [2498, [69.51089262962341, 223.40243291854858, 216.17932891845703]], [34453, [233.31077408790588, 65.0344409942627, 202.32985281944275]], [7369, [69.59844088554382, 82.45620679855347, 80.2625789642334]], [12701, [90.36657094955444, 84.40098595619202, 78.9956271648407]], [36873, [67.06822872161865, 68.1882848739624, 70.08416080474854]], [17681, [92.14423489570618, 67.32422018051147, 118.93220400810242]], [15311, [84.36215090751648, 87.97714114189148, 72.37125563621521]], [31956, [70.72783303260803, 71.32095193862915, 69.20530295372009]], [44102, [112.02787804603577, 108.22787594795227, 100.50231409072876]], [45811, [127.97166204452515, 100.12954115867615, 113.52290201187134]], [32890, [164.69935178756714, 151.8915572166443, 152.62738394737244]], [25133, [146.10261917114258, 62.382538080215454, 149.4518039226532]], [39000, [59.60682487487793, 91.26792287826538, 96.09875893592834]], [18123, [63.7340989112854, 198.85757112503052, 200.31168007850647]], [14894, [203.59928178787231, 90.00776696205139, 203.53603291511536]], [30315, [63.495904207229614, 66.06091022491455, 63.70245599746704]], [42637, [74.7255470752716, 67.80333805084229, 71.49138402938843]], [46533, [96.83730006217957, 91.68171525001526, 69.2635269165039]], [11747, [62.82764482498169, 99.08576011657715, 102.02689814567566]], [21022, [73.03947877883911, 92.0443160533905, 77.44750905036926]], [45557, [94.02819514274597, 99.73337197303772, 68.59247779846191]], [1559, [65.09916186332703, 94.88815903663635, 95.48632073402405]], [24379, [668.3284773826599, 99.88362693786621, 662.0733609199524]], [12996, [78.57122993469238, 112.12171292304993, 103.78985404968262]], [4525, [120.28771305084229, 113.73079800605774, 82.37071895599365]], [41581, [77.22158193588257, 88.84094595909119, 74.53733110427856]], [34167, [85.23128509521484, 77.13868188858032, 85.37701201438904]], [40509, [69.60090899467468, 75.17367100715637, 79.00155401229858]], [20722, [104.95570182800293, 69.48427200317383, 110.31709909439087]], [26830, [77.32894229888916, 72.45263314247131, 71.81864404678345]], [24968, [66.71765089035034, 71.54813313484192, 75.26708698272705]], [5841, [68.86105275154114, 71.58614683151245, 71.20972299575806]], [17245, [62.52356696128845, 219.07700204849243, 244.9939022064209]], [46025, [86.39352703094482, 105.47410082817078, 102.13181495666504]], [13672, [65.72757077217102, 74.43127703666687, 72.61021709442139]], [37152, [113.03454113006592, 99.87882018089294, 71.51455092430115]], [17242, [78.37856698036194, 71.22794604301453, 74.59059286117554]], [41849, [71.21043109893799, 80.18200612068176, 75.21416115760803]], [35795, [73.98591208457947, 70.40540385246277, 72.14485597610474]], [26305, [65.76566910743713, 78.81995916366577, 80.01225900650024]], [33472, [103.43204689025879, 99.32069873809814, 72.95839309692383]], [5474, [71.12742805480957, 72.2438268661499, 63.51945900917053]], [15027, [71.3316171169281, 70.39561605453491, 67.01471209526062]], [35653, [92.33022713661194, 69.3017430305481, 111.3168671131134]], [25150, [111.58803081512451, 109.61395311355591, 88.85224413871765]], [8438, [66.99501585960388, 75.4324631690979, 73.40234684944153]], [45757, [71.75135588645935, 73.37735486030579, 65.25131916999817]], [5666, [732.6780850887299, 104.89572095870972, 738.2242958545685]], [27489, [238.6039891242981, 80.28336691856384, 172.3206090927124]], [29198, [123.75264096260071, 95.39119982719421, 116.3401231765747]], [3369, [77.22201704978943, 72.67622184753418, 100.59018278121948]], [19149, [99.61953783035278, 74.26639699935913, 97.02082014083862]], [41305, [217.99529600143433, 167.7687051296234, 70.38587379455566]], [32971, [160.4140408039093, 182.01754808425903, 167.01166415214539]], [1180, [69.13595604896545, 214.05829095840454, 161.12229895591736]], [41614, [69.91027092933655, 220.13374710083008, 197.50797200202942]], [13028, [103.45818090438843, 184.9131109714508, 184.6055691242218]], [8778, [91.39377307891846, 159.30760431289673, 160.87024188041687]], [6964, [69.31837296485901, 164.56155610084534, 162.23530983924866]], [31520, [65.69614696502686, 70.90243196487427, 78.32770991325378]], [44677, [67.38719701766968, 73.68128180503845, 77.73207497596741]], [16849, [67.58080625534058, 101.65367221832275, 106.80570697784424]], [46508, [121.45953106880188, 108.80765795707703, 85.49681305885315]], [20379, [84.11043620109558, 272.1434941291809, 226.93387699127197]], [20910, [71.35066986083984, 77.04578614234924, 88.1346685886383]], [41382, [162.98921418190002, 248.82782697677612, 259.7049231529236]], [2189, [70.74960279464722, 75.02744817733765, 78.72465586662292]], [10501, [79.76633810997009, 158.5913598537445, 164.03243017196655]], [37426, [98.51228284835815, 115.91363406181335, 84.20329999923706]], [13416, [242.03167510032654, 178.76290893554688, 103.12051701545715]], [10485, [111.85252118110657, 115.55747699737549, 96.48392391204834]], [15139, [73.31060981750488, 74.46084690093994, 85.78416299819946]], [19561, [79.43401288986206, 70.26240301132202, 82.29435515403748]], [9339, [66.81089615821838, 129.54192686080933, 127.48070096969604]], [20053, [94.32460904121399, 76.76542210578918, 116.56863903999329]], [41124, [109.8064227104187, 78.11506199836731, 97.96141004562378]], [2940, [59.85053110122681, 90.22608494758606, 89.18706583976746]], [19103, [71.77005004882812, 66.58166003227234, 67.06874394416809]], [3222, [88.03245711326599, 63.534746170043945, 85.30950021743774]], [7189, [85.85126495361328, 88.1563138961792, 67.62016725540161]], [1358, [62.708000898361206, 88.12205576896667, 89.17157793045044]], [11205, [88.75064873695374, 67.8731677532196, 89.90057396888733]], [9514, [68.36375308036804, 68.03567600250244, 63.958401918411255]], [37597, [66.45266699790955, 68.92298316955566, 71.57995915412903]], [21723, [193.59007692337036, 196.19698691368103, 63.053709983825684]], [43325, [70.5363290309906, 69.21230101585388, 70.86066508293152]], [26905, [86.61411309242249, 78.8992931842804, 103.11885619163513]], [35860, [74.21826100349426, 78.57601571083069, 70.09185194969177]], [5571, [99.49493503570557, 100.38392305374146, 98.76116681098938]], [1807, [71.05958771705627, 69.22675681114197, 69.64467883110046]]]
