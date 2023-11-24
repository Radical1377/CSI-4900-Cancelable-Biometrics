from math import sqrt

def circumcircle(trig):
    
    # Output circle
    circle = [0, 0, 0] 
    
    trig = list(trig)

    # Triangle Points
    a = (trig[0][0], trig[0][1])
    b = (trig[1][0], trig[1][1])
    c = (trig[2][0], trig[2][1])
    
    # Select appropriate lines that are non-horizontal
    line_1 = {"p1":None, "p2":None}
    line_2 = {"p1":None, "p2":None}
    
    if a[1] - b[1] == 0:
        line_1["p1"], line_1["p2"] = a, c
        line_2["p1"], line_2["p2"] = b, c
    elif a[1] - c[1] == 0:
        line_1["p1"], line_1["p2"] = a, b
        line_2["p1"], line_2["p2"] = c, b
    elif b[1] - c[1] == 0:
        line_1["p1"], line_1["p2"] = b, a
        line_2["p1"], line_2["p2"] = c, a
    else:
        line_1["p1"], line_1["p2"] = a, c
        line_2["p1"], line_2["p2"] = b, c

    # Calculate midpoints of lines 1 and 2
    mid_1 = ((line_1["p1"][0]+line_1["p2"][0]) / 2, (line_1["p1"][1]+line_1["p2"][1]) / 2)
    mid_2 = ((line_2["p1"][0]+line_2["p2"][0]) / 2, (line_2["p1"][1]+line_2["p2"][1]) / 2)
    
    # Calculate the slopes of lines perpendicular to lines 1 and 2
    slop_1 = -(line_1["p1"][0]-line_1["p2"][0]) / (line_1["p1"][1]-line_1["p2"][1])
    slop_2 = -(line_2["p1"][0]-line_2["p2"][0]) / (line_2["p1"][1]-line_2["p2"][1])

    # Calculate the y-axis intersection points of lines 1 and 2
    yint_1 = mid_1[1] - (slop_1 * mid_1[0])
    yint_2 = mid_2[1] - (slop_2 * mid_2[0])

    # Calculate the coordinate and radius of the circumcircle
    circle[0] = (yint_2 - yint_1) / (slop_1 - slop_2)
    circle[1] = (slop_1 * circle[0]) + yint_1
    circle[2] = sqrt(pow((circle[0] - a[0]), 2) + pow((circle[1] - a[1]), 2))
    return circle

def isInCircum(point, trig):
    circle = circumcircle(trig)
    return sqrt(pow((circle[0]-point[0]), 2) + pow((circle[1]-point[1]), 2)) <= circle[2]
