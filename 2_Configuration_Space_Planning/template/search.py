# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush



def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)


def getPath(current, gVal, closedList):
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """

    path = [current]
    
    while closedList[current] is not None:
        path.append(closedList[current])
        current = closedList[current]

    path.reverse()

    return path


def bfs(maze, ispart1=False):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """
    
    closedList = {}     # key: node, value: previous node
    queue = deque()
    gVal = {}
    path = []
    isValid = False

    start = maze.getStart()
    objectives = maze.getObjectives()

    gVal[start] = 0
    closedList[start] = None
    queue.append(start)
    
    while queue:
        temp = queue.pop()
        current = temp
        
    
        if maze.isObjective(current[0], current[1], current[2], ispart1):
            path = getPath(current, gVal, closedList)
            isValid = True
            break
        
        else:
            neighbors = maze.getNeighbors(current[0], current[1], current[2], ispart1)
            for neighbor in neighbors:
                if neighbor not in closedList:
                    closedList[neighbor] = current
                    gVal[neighbor] = gVal[current] + 1
                    queue.appendleft(neighbor)

                if  (gVal[current] +1 < gVal[neighbor]) :
                    closedList[neighbor] = current
                    gVal[neighbor] = gVal[current] + 1
    
    if (isValid):
        path = getPath(current, gVal, closedList)
        # print(path)
        if maze.isValidPath(path):
            return path
   
    return None
    