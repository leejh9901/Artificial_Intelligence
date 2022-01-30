# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from collections import deque
import heapq
from math import inf
import copy
from typing import final

# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        
        self.distances   = {
                # (i, j): DISTANCE(i, j)
                # Manhattan Distance
               (i, j): abs(i[0] - j[0]) + abs(i[1] - j[1])
                # (i, j): self.compute_mst_weight()

               for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)



# get the path from gVal and closedList
def getPath(current, gVal, closedList):

        path = [current]
        
        while closedList[current] is not None:
            path.append(closedList[current])
            current = closedList[current]

        path.reverse()

        return path



def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    closedList = {}     # key: node, value: previous node
    queue = deque()
    gVal = {}
    path = []
    
    gVal[maze.start] = 0
    closedList[maze.start] = None
    queue.append(maze.start)
    
    while queue:
        temp = queue.pop()
        current = temp
        
    
        if current == maze.waypoints[0]:
            
            path = getPath(current, gVal, closedList)
            return path
        
        for neighbor in maze.neighbors(current[0], current[1]):
            if neighbor not in closedList:
                closedList[neighbor] = current
                gVal[neighbor] = gVal[current] + 1
                queue.appendleft(neighbor)

            if  (gVal[current] +1 < gVal[neighbor]) :
                closedList[neighbor] = current
                gVal[neighbor] = gVal[current] + 1
    
    path = getPath(current, gVal, closedList)
    return path

    


# Computes the h value
def ManhattanDist(p1, p2):
    return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

# Computes the g value
def DistFromStart(start, end): 
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    openList = []       # 0: h val, 1: node              Keeps track of nodes that need to be examined
    closedList = {}     # already visited  0: node, 1: visited from where            Keeps track of ndoes that are examined
    gVal = {}           #  0: node, 1: g value                   (key: node, value: gVal)
 
    # examine the starting node
    gVal[maze.start] = 0
    heapq.heappush(openList, (gVal[maze.start] + ManhattanDist(maze.start, maze.waypoints[0]), maze.start))
    closedList[maze.start] = None

    while openList:
        current = openList[0][1]    # node (e.g, maze.start for the first loop)
        if current == maze.waypoints[0]:
            return getPath(current, gVal, closedList)

        heapq.heappop(openList)     # no need to store the value bc it's already sotred in current

        for neighbor in maze.neighbors(current[0], current[1]):
            if neighbor not in closedList:
                gVal[neighbor] = gVal[current] + 1
                closedList[neighbor] = current

                heapq.heappush(openList, (gVal[neighbor] + ManhattanDist(neighbor, maze.waypoints[0]), neighbor))
               
            if  (gVal[current] +1 < gVal[neighbor]) :
                closedList[neighbor] = current
                gVal[neighbor] = gVal[current] + 1
               
    return getPath(current, gVal, closedList)



def findShortestD(remaining_waypoints, current):
    minD = inf
    (minI, minJ) = (-1, -1)
    for (i, j) in remaining_waypoints:
        dist = abs(i - current[0]) + abs(j - current[1])
        if dist < minD :
            minD = dist
            (minI, minJ) = (i, j)

    return [(minI, minJ), minD]

def getHeuristic(mst, current, distToD):
    return  distToD + mst.compute_mst_weight()



def getAstar(maze, start, dest):
    newMaze = copy.deepcopy(maze)
    newMaze.start = start
    newMaze.waypoints = [dest]

    return astar_single(newMaze)



def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    start = maze.start
    remaining_waypoints = list(maze.waypoints)
    final_path = [start]
    mstWeights = []
    shortestD = []
    
    

    while remaining_waypoints:
        hval = 0
        idx = -1
        if len(remaining_waypoints) > 1:
            mstWeights = []
            shortestD = []
            for i in range(len(remaining_waypoints)):
                shortestD.append(ManhattanDist(start, remaining_waypoints[i]))
            
            currentMstWeight = MST(remaining_waypoints).compute_mst_weight()

            idx = shortestD.index(min(shortestD))
            hval = shortestD[idx] + currentMstWeight

        else:
            hval = ManhattanDist(start, remaining_waypoints[0])
            mstWeights = [0]
            idx = 0

        currentNode = remaining_waypoints[idx]
        # return []
        #


        # astar_single
        #
            
        openList = []       # 0: h val, 1: node              Keeps track of nodes that need to be examined
        closedList = {}     # already visited  0: node, 1: visited from where            Keeps track of ndoes that are examined
        gVal = {}           #  0: node, 1: g value                   (key: node, value: gVal)
        temp_path = deque()

        # examine the starting node
        gVal[start] = 0
        heapq.heapify(openList)
        heapq.heappush(openList, ((gVal[start] + hval), start))
        closedList[start] = None

        while openList:
            current = openList[0][1]    # node (e.g, maze.start for the first loop)
            if current == currentNode:
                temp_path = getPath(current, gVal, closedList)
                break

            heapq.heappop(openList)     # no need to store the value bc it's already sotred in current

            for neighbor in maze.neighbors(current[0], current[1]):
                if neighbor not in closedList:
                    gVal[neighbor] = gVal[current] + 1
                    closedList[neighbor] = current

                    heapq.heappush(openList, (gVal[neighbor] + ManhattanDist(neighbor, current), neighbor))
                
                if  (gVal[current] +1 < gVal[neighbor]) :
                    closedList[neighbor] = current
                    gVal[neighbor] = gVal[current] + 1
                
        
        temp_path = getPath(current, gVal, closedList)
        final_path.extend(list(temp_path)[1:])

        start = currentNode
        remaining_waypoints.pop(idx)
    
    # print(final_path)
    return final_path







def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    