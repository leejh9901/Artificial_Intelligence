# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien


#  The shortest distance between a line segment and a point
#  Reference: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
#             https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def dist_between_line_and_point(x0, y0, x1, y1, x2, y2):    # (x0, y0) : pt

    dx = x2 - x1
    dy = y2 - y1

    if ((dx**2 + dy**2) == 0):
        return math.sqrt(x0**2 + y0**2)
    
    temp_dist = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)
    
        
    # Check if the point is on the line segment
    if (temp_dist < 0):
        closestX = x1
        closestY = y1
        dx = x0 - x1
        dy = y0 - y1
    
    elif (temp_dist > 1):
        closestX = x2
        closestY = y2
        dx = x0 - x2
        dy = y0 - y2

    else:
        closestX = x1 + temp_dist * dx
        closestY = y1 + temp_dist * dy
        dx = x0 - closestX
        dy = y0 - closestY

    return math.sqrt(dx**2 + dy**2)



def dist_between_points(x1,y1, x2,y2):
    return abs(math.sqrt((x1-x2)**2 + (y1-y2)**2))


def ccw(x1,y1, x2,y2, x3,y3):
    direction = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)

    if (direction > 0):
        return 1
    elif (direction < 0):
        return -1
    else:
        return 0


# Check if the two line segments intersect
# Return True if two line segments intersect
# Reference: 
def line_intersect(x1,y1, x2,y2, x3,y3, x4,y4):
    # Check 
    if ((ccw(x1,y1, x2,y2, x3,y3) *  ccw(x1,y1, x2,y2, x4,y4)) < 0) and  ((ccw(x3,y3, x4,y4, x1,y1) *  ccw(x3,y3, x4,y4, x2,y2)) < 0):
        return True

    return False



def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    # calculate granularity / sqrt(2)
    #   store it in a local variable
    # check the alien shape
    #   if shape is ball
    #       check the radius only (width)
    #   if shape is oblong
    #       check both length and width
    # 
    # CHECK
    #   Circle
    #       line from a point (min_distance < radius+buffer) ?
    #   Oblong
    #       head and tail   --> line from a point (min_distance < radius+buffer) ?
    #       line(head--tail) intersects with wall?


    granularity_offset = (granularity / math.sqrt(2)) 
    shape = alien.is_circle()
    centeroid = alien.get_centroid()
    head_and_tail = alien.get_head_and_tail()   #[(x_head, y_head), (x_tail, y_tail)]
    radius = alien.get_width()      # radius of the current alien shape

    # HORIZONTAL, VERTICAL
    if (shape == False):
        for wall in walls:
            # Alien_HEAD
            dist = dist_between_line_and_point(head_and_tail[0][0], head_and_tail[0][1], wall[0], wall[1], wall[2], wall[3])
            if (dist < radius + granularity_offset) or np.isclose(dist, radius+granularity_offset) :
                return True

            # Alien_TAIL
            dist = dist_between_line_and_point(head_and_tail[1][0], head_and_tail[1][1], wall[0], wall[1], wall[2], wall[3])
            if (dist < radius + granularity_offset) or np.isclose(dist, radius+granularity_offset):
                return True

            # Alien_LINE_INTERSECT
            if (line_intersect(head_and_tail[0][0], head_and_tail[0][1], head_and_tail[1][0], head_and_tail[1][1], wall[0], wall[1], wall[2], wall[3])):
                return True

            # Wall_HEAD
            dist = dist_between_line_and_point(wall[0], wall[1], head_and_tail[0][0], head_and_tail[0][1], head_and_tail[1][0], head_and_tail[1][1])
            if (dist < radius + granularity_offset) or np.isclose(dist, radius + granularity_offset) :
                return True

            # Wall_TAIL
            dist = dist_between_line_and_point(wall[2], wall[3], head_and_tail[0][0], head_and_tail[0][1], head_and_tail[1][0], head_and_tail[1][1])
            if (dist < radius + granularity_offset) or np.isclose(dist, radius + granularity_offset):
                return True

    
    # BALL
    if (shape == True):
        for wall in walls:
            dist = dist_between_line_and_point(centeroid[0], centeroid[1], wall[0], wall[1], wall[2], wall[3])

            if (dist < radius + granularity_offset) or np.isclose(dist, radius + granularity_offset):
                return True

    return False



def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """

    shape = alien.is_circle()
    centeroid = alien.get_centroid()
    head_and_tail = alien.get_head_and_tail()   #[(x_head, y_head), (x_tail, y_tail)]
    radius = alien.get_width()      # radius of the current alien shape    

     # HORIZONTAL, VERTICAL
    if (shape == False):
        for goal in goals:
            # Alien_HEAD
            dist = dist_between_points(head_and_tail[0][0], head_and_tail[0][1], goal[0], goal[1])
            if (dist < radius + goal[2]) or np.isclose(dist, radius + goal[2]) :
                return True

            # Alien_TAIL
            dist = dist_between_points(head_and_tail[1][0], head_and_tail[1][1], goal[0], goal[1])
            if (dist < radius + goal[2]) or np.isclose(dist, radius + goal[2]):
                return True

            # Alien_LINE_INTERSECT
            dist = dist_between_line_and_point(goal[0], goal[1], head_and_tail[0][0], head_and_tail[0][1], head_and_tail[1][0], head_and_tail[1][1])
            if (dist < radius + goal[2]) or np.isclose(dist, radius + goal[2]):
                return True

 
    # BALL
    if (shape == True):
        for goal in goals:
            dist = dist_between_points(centeroid[0], centeroid[1], goal[0], goal[1])
            if (dist < radius + goal[2]) or np.isclose(dist, radius + goal[2]):
                return True


    return False




def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """

    granularity_offset = (granularity / math.sqrt(2)) 
    shape = alien.is_circle()
    centeroid = alien.get_centroid()
    head_and_tail = alien.get_head_and_tail()   #[(x_head, y_head), (x_tail, y_tail)]
    radius = alien.get_width()      # radius of the current alien shape

    walls = [(0, 0, 0, window[1]), (0, 0, window[0], 0), (window[0], 0, window[0], window[1]), (0, window[1], window[0], window[1])]
    if ((centeroid[0] > 0 or np.isclose(centeroid[0], 0)) and (centeroid[0] < window[0] or np.isclose(centeroid[0], window[0])) ) and ((centeroid[1] > 0 or np.isclose(centeroid[1], 0)) and (centeroid[1] < window[1] or np.isclose(centeroid[1], window[1]))):

        # HORIZONTAL, VERTICAL
        if (shape == False):
            for wall in walls:
                # Alien_HEAD
                dist = dist_between_line_and_point(head_and_tail[0][0], head_and_tail[0][1], wall[0], wall[1], wall[2], wall[3])
                if (dist < radius or np.isclose(dist, radius)):
                    return False

                # Alien_TAIL
                dist = dist_between_line_and_point(head_and_tail[1][0], head_and_tail[1][1], wall[0], wall[1], wall[2], wall[3])
                if (dist < radius or np.isclose(dist, radius)):
                    return False

                # Alien_LINE_INTERSECT
                if (line_intersect(head_and_tail[0][0], head_and_tail[0][1], head_and_tail[1][0], head_and_tail[1][1], wall[0], wall[1], wall[2], wall[3])):
                    return False

        
        # BALL
        if (shape == True):
            for wall in walls:
                dist = dist_between_line_and_point(centeroid[0], centeroid[1], wall[0], wall[1], wall[2], wall[3])

                if (dist < radius) or np.isclose(dist, radius):
                    return False


    return True

if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0) 
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, expected: {truths[0]}'
        assert touch_goal_result == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, expected: {truths[1]}'
        assert in_window_result == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, expected: {truths[2]}'

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")