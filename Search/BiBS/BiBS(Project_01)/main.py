import getopt
import sys

from search.algorithms import State
from search.map import Map
from search.search import *


def main():
    """
    Function for testing your A* and Dijkstra's implementation. There is no need to edit this file.
    Run it with a -help option to see the options available. 
    """
    optlist, _ = getopt.getopt(sys.argv[1:], 'h:m:r:', ['testinstances', 'plots', 'help'])

    plots = False
    for o, a in optlist:
        if o in ("-help"):
            print("Examples of Usage:")
            print("Solve set of test instances: main.py --testinstances")
            print("Solve set of test instances and generate plots: main.py --testinstances --plots")
            exit()
        elif o in ("--testinstances"):
            test_instances = "test-instances/testinstances.txt"
                              
    gridded_map = Map("dao-map/brc000d.map")
    
    nodes_expanded_dijkstra = []    
    nodes_expanded_bibs = []
    
    start_states = []
    goal_states = []
    solution_costs = []
       
    file = open(test_instances, "r")
    for instance_string in file:
        list_instance = instance_string.split(",")
        start_states.append(State(int(list_instance[0]), int(list_instance[1])))
        goal_states.append(State(int(list_instance[2]), int(list_instance[3])))
        
        solution_costs.append(float(list_instance[4]))
    file.close()
        
    for i in range(0, len(start_states)):    
        start = start_states[i]
        goal = goal_states[i]
        # print(f"\n\nnodes={start}")

        # Calling BiBS strategy based on given states and the map;
        bibs = BiBS(start, goal, gridded_map)
        cost, expanded_bibs = bibs.search()
        print(f"\tBIBS: cost={cost}, expansion={expanded_bibs}")
        nodes_expanded_bibs.append(expanded_bibs)
        
        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by Bi-HS and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()
    
    print('Finished running all experiments.')


if __name__ == "__main__":
    main()
