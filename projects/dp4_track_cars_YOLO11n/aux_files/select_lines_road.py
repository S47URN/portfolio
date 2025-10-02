import numpy as np
import cv2


def calc_slope_intercept(lines):
    
    # Convert to slope-intercept
    params = []
    for x1, y1, x2, y2 in lines:
        if abs(x2-x1) < 1e-6:  # vertical
            slope = 1e6
        else:
            slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1
        params.append(((x1,y1,x2,y2), slope, intercept))
    return params


def are_similar(line1, line2, slope_thresh=0.1, intercept_thresh=30):
    _, s1, i1 = line1
    _, s2, i2 = line2
    return abs(s1 - s2) <= slope_thresh and abs(i1 - i2) <= intercept_thresh

# just for almost horizontal lines, consider geometric distance
# vertical difference
def line_distance(line1, line2, slope_thresh=0.1, vertical_thresh=30):
    
    (x1_1,y1_1,x2_1,y2_1), m1, b1 = line1
    (x1_2,y1_2,x2_2,y2_2), m2, b2 = line2
    # middle point inside lines
    x_ref = sum([x1_1, x1_2, x2_1, x2_2])/4
    
    y1 = m1 * x_ref + b1
    y2 = m2 * x_ref + b2
    
    slope_diff = abs(m1 - m2)
    y_diff = abs(y1 - y2)
    
    return slope_diff <= slope_thresh and y_diff <= vertical_thresh

# group similar ones
# DBScan can be suitable to
# Using Custom Grouping
def grouping_lines(lines, funct=are_similar, kws={"slope_thresh":0.1, "vertical_thresh":30}):
    groups = []
    for line in lines:
        added = False
        for group in groups:
            if any(funct(line, gline, **kws) for gline in group):
                group.append(line)
                added = True
                break
        if not added:
            groups.append([line])
    
    #for idx, group in enumerate(groups):
    #    print(f"Group {idx}: {group}")

    return groups


def fit_group(group):
        
    # Fit one line through all points in this group
    group = np.array([el[0] for el in group])

    if len(group)==1:
        return group[0].tolist()
    
    xs = np.concatenate([group[:,0], group[:,2]])
    ys = np.concatenate([group[:,1], group[:,3]])
    
    # line in x = m*y + b form
    poly = np.polyfit(xs, ys, deg=1)
    m, b = poly
    
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min = int(m*x_min + b)
    y_max = int(m*x_max + b)
    
    return (x_min, y_min, x_max, y_max)



def group_similar_lines(lines):
    # group lines based on slope and intercept similarity
    #slope_thresh=0.1; dist_thresh=30

    params = calc_slope_intercept(lines) # [(x1,y1,x2,y2), slope, intercept]
    # creato groups
    road_sides_lines = [el for el in params if abs(el[1])>=0.25 and abs(el[1])<=0.5] # el[1] == "slope" 
    road_limit_lines = [el for el in params if abs(el[1])<0.1]
    
    groups_sides = grouping_lines(road_sides_lines, funct=are_similar, kws={"slope_thresh":0.3, "intercept_thresh":15})
    groups_limit = grouping_lines(road_limit_lines, funct=line_distance, kws={"slope_thresh":0.1, "vertical_thresh":15})
    
    new_side = []; new_limit = []
    for group in groups_limit:
        x1,y1,x2,y2 = fit_group(group)
        dist = abs(x1-x2)
        #print((x1,y1,x2,y2), dist )
        new_limit.append([(x1,y1,x2,y2), dist])
    
    # estimate horizon and limit road lines
    new_limit = sorted(new_limit, key = lambda x: x[1])
    
    if len(new_limit)==0:
        # size and y-position constrains
        horizon, limit_line = None, None
    elif len(new_limit)==1:
        horizon, limit_line = new_limit[-1][0], None
    else:
        horizon    = (new_limit[-1][0] if new_limit[-1][1] > 800 else None) if new_limit[-1][0][1]<400 else None
        limit_line = (new_limit[-2][0] if new_limit[-1][1] > 300 else None) if new_limit[-2][0][1]<400 else None
    
    #print("Horizon", horizon)
    #print("Limit", limit_line)
    
    new_lines=[]
    for group in groups_sides:
        new_lines.append(fit_group(group))

    new_lines = calc_slope_intercept(new_lines)
    left_side  = []; right_side = []
    center_h = 1280//2 # width of frame
    for el in new_lines:
        line, slope, intercept = el
        x1,y1,x2,y2 = line # w= 1280
        if slope<0:
            x1,x2 = sorted([x1,x2])
            line_w = x2-x1
            
            x1 = center_h if x1 <= center_h else x1
            line_sep = x2 - x1
            
            if not (line_sep > 0.5*line_w): #more than 50% of line shouldn't be to left
                left_side.append(el)
        else:
            x1,x2 = sorted([x1,x2])
            line_w = x2-x1
            
            x2 = x2 if x2<= center_h else center_h
            line_sep = x2 - x1

            if not (line_sep > 0.5*line_w): #more than 50% of line shouldn't be to left
                right_side.append(el)
    
    left_side  = fit_group(left_side) if len(left_side) > 0 else None
    right_side = fit_group(right_side) if len(right_side) > 0 else None

    return {"horizon": horizon, "limit": limit_line, "left":left_side, "right":right_side}


# Find Roads


def return_m_b(line):
    x1, y1, x2, y2 = line
    xs = [x1,x2]; ys = [y1,y2]
    poly = np.polyfit(xs, ys, deg=1)
    m, b = poly
    return m, b
    
def fit_points(line, y_var=False):
    
    m, b = return_m_b(line)
    #x_min, x_max = int(min(xs)), int(max(xs))
    #y_min = int(m*x_min + b)
    #y_max = int(m*x_max + b)

    if y_var:
        return lambda y: (y - b)/m 
    else: 
        return lambda x: m*x + b

def line_inter_x(p1, p2):
    # x intersection between two points
    # p1 => e.g. dir_lines["limit"] = x1,y1, x2,y2
    
    m1,b1 = return_m_b(p1)
    m2,b2 = return_m_b(p2)
    
    return (b2-b1)/(m1 - m2)

def calculate_lines(frame):
    """ Preprocess Frame to obtain the road lines"""
    # 1. Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0) #(5,5)
    edges = cv2.Canny(blur, 90, 120) # 50, 150
    
    # 2. Focus on region of interest (lower half of image)
    h, w = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi = np.array([[(0, h), (w, h), 
                     (w, h//12), (0, h//12)]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    edges_roi = cv2.bitwise_and(edges, mask)
    
    # 3. Detect lines with Hough
    lines = np.squeeze(cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=100,
                                    minLineLength=80, maxLineGap=50))
    
    dir_lines = group_similar_lines(lines)
    #{"horizon": horizon, "limit": limit_line, "left":left_side, "right":right_side}
    return dir_lines
