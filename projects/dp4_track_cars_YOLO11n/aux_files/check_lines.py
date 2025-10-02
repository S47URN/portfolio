import numpy as np
import pandas as pd
from select_lines_road import (return_m_b, fit_points, 
                                line_inter_x, calculate_lines)
import cv2


def smooth_edges_variable_window(data, window_size=21):
    half_window = window_size // 2
    smoothed_data = np.full_like(data, np.nan, dtype=float)
    n = len(data)

    for i in range(n):
        # Determine the dynamic window size for the current point
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Calculate the mean of the dynamic window
        smoothed_data[i] = np.mean(data[start:end])
        
    return smoothed_data


def smooth_line_points_variable_window(df_col, window_size=21):

    data = np.array(df_col.tolist())
    half_window = window_size // 2
    
    smoothed_data = np.full_like(data, np.nan, dtype=float)
    n = len(data)
    
    for i in range(n):
        # Determine the dynamic window size for the current point
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Calculate the mean of the dynamic window
        smoothed_data[i] = np.mean(data[start:end], axis=0)

    smoothed_series = pd.Series(list(smoothed_data), index=df_col.index, dtype=object)
    
    return smoothed_series


def sanity_check_lines(_temp):
    
    # correct horizon-limit, sometimes horizon was mismatched as limit of the road
    # check it by association
    _change_idx = _temp["horizon"].isna()
    _temp.loc[_change_idx, "horizon"] = _temp.loc[_change_idx, "limit"]
    _temp.loc[_change_idx, "limit"]   = None
    
    ################# consider no empty horizon
    _temp["horizon"] = _temp["horizon"].ffill().bfill()
    _temp["horizon_h"] = _temp["horizon"].apply(lambda x: (x[1]+x[3])/2)
    # Smooth horizon, smooth lines variation
    data = _temp["horizon_h"].dropna()
    # Apply a centered rolling mean
    smoothed_result = smooth_edges_variable_window(data, window_size=21)
    _temp.loc[data.index,"horizon_h1"] = smoothed_result
    ##################
    #Calculate slopes, intercepts
    f_mb = lambda x: return_m_b(x) if x is not None else x
    
    # Consider some missing road limit line
    _temp["limit_h"] = _temp["limit"].apply(lambda x: (x[1]+x[3])/2 if x is not None else None)
    #correct limit road
    _temp.loc[_temp["limit_h"]<_temp["horizon_h1"], ["limit","limit_h"]] = None
    
    #check distance horizon-limit road
    #_temp["diff"] = abs(_temp["horizon_h1"] - _temp["limit_h"])
    #_temp.loc[_temp["diff"]<40, "limit_h"]  = None
    # fill gaps
    _temp.loc[_temp["limit_h"].isna(), "limit_h"] = _temp.loc[_temp["limit_h"].isna(), "horizon_h1"] + 50.0
    # Smooth horizon, smooth lines variation, Apply a centered rolling mean
    smoothed_result = smooth_edges_variable_window(_temp["limit_h"], window_size=31)
    _temp.loc[data.index,"limit_h1"] = smoothed_result

    # fill dummy lines to correct
    dummy = _temp.loc[_temp["limit"].isna(), "horizon"]
    _temp.loc[_temp["limit"].isna(), "limit"] = dummy.apply(lambda x: x + np.array([0,50,0,50]))
    
    # Smooth line points
    _temp["limit"]   = smooth_line_points_variable_window(_temp["limit"], window_size=71)
    _temp["horizon"] = smooth_line_points_variable_window(_temp["horizon"], window_size=71)
    
    # Recalculate Limit, horizon h
    _temp["limit_h"] = _temp["limit"].apply(lambda x: (x[1]+x[3])/2 )
    _temp["horizon_h"] = _temp["horizon"].apply(lambda x: (x[1]+x[3])/2 )
    
    ###################
    
    # Update y coordinates
    def update_y(df, old_h, new_h, colum_param):
        # y_new = (h_new - h) + y_old
        _incre = (df[new_h] - df[old_h]).apply(lambda x: np.array([0,x,0,x])) 
        return df[colum_param] + _incre  

    # Adjust correct height
    _temp["horizon"] = update_y(_temp, "horizon_h", "horizon_h1", "horizon")
    _temp["limit"]   = update_y(_temp, "limit_h", "limit_h1", "limit")
    
    ######################## Review Side Line
    # Checking that left and ride lines always exists
    _temp.loc[:, ["left","right"]] = _temp.loc[:, ["left","right"]].ffill().bfill()
    
    _temp[['l_m', 'l_b']] = pd.DataFrame(_temp["left"].apply(lambda x: f_mb(x)).tolist(), index=_temp.index)
    _temp[['r_m', 'r_b']] = pd.DataFrame(_temp["right"].apply(lambda x: f_mb(x)).tolist(), index=_temp.index)
    
    # strong smooth for slopes and intercepts
    _temp['l_m_sm'] = smooth_edges_variable_window(_temp['l_m'], window_size=71)
    _temp['r_m_sm'] = smooth_edges_variable_window(_temp['r_m'], window_size=71)
    
    #Considering intersectino
    _temp['l_b_sm'] = smooth_edges_variable_window(_temp['l_b'], window_size=71)
    _temp['r_b_sm'] = smooth_edges_variable_window(_temp['r_b'], window_size=71)
    
    #Update points
    # y = mx + b
    # x = (y - b) / m
    # left [x1=0,y1=b, x2=-b/m, y2=0]
    # 0, _temp["l_b_sm"], - _temp["l_b_sm"]/_temp["l_m_sm"], 0
    _temp["left"] = _temp.apply(lambda row: np.array([0, row['l_b_sm'], -row['l_b_sm'] / row['l_m_sm'], 0]), axis=1)
    _temp["right"] = _temp.apply(lambda row: np.array([0, row['r_b_sm'], -row['r_b_sm'] / row['r_m_sm'], 0]), axis=1)
    
    return _temp[["frame","left","right","limit","horizon"]]


def plot_road_lines(frame, dir_lines=None):

    h = frame.shape[0]
    w = frame.shape[1]
    
    if dir_lines is None: dir_lines = calculate_lines(frame)
    #{"horizon": horizon, "limit": limit_line, "left":left_side, "right":right_side}
    

    fr = frame.copy()

    # generate sides
    ###  functions
    if dir_lines["left"] is not None:
        left_function = fit_points(dir_lines["left"], y_var=False)
        left_function_inv  = fit_points(dir_lines["left"], y_var=True)

        y1 = 0; x1 = int(left_function_inv(y1))
        y2 = h; x2 = int(left_function_inv(y2))
        cv2.line(fr, (x1, y1), (x2, y2), (0,250,0), 4)  # draw green line
        cv2.putText(fr, "Left", (x1-150,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,150,250), 3)

    
    if dir_lines["right"] is not None:
        right_function = fit_points(dir_lines["right"], y_var=False)
        right_function_inv = fit_points(dir_lines["right"], y_var=True)
    
        y1 = 0; x1 = int(right_function_inv(y1))
        y2 = h; x2 = int(right_function_inv(y2))
        cv2.line(fr, (x1, y1), (x2, y2), (0,250,0), 4)  # draw green line
        cv2.putText(fr, "Right", (x1+100,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (150,0,250), 3)

    if dir_lines["limit"] is not None:
        limit_function_inv = fit_points(dir_lines["limit"], y_var=True)
        limit_function     = fit_points(dir_lines["limit"], y_var=False)
        #generate limit
        x1 = 0; y1 = int(limit_function(x1))
        x2 = w; y2 = int(limit_function(x2))
        cv2.line(fr, (x1, y1), (x2, y2), (0,250,0), 4)  # draw green line
        cv2.putText(fr, "Limit", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,250,0), 3)
    
    
    if dir_lines["horizon"] is not None:
        #generate horizon
        hor_function = fit_points(dir_lines["horizon"], y_var=False)
        x1 = 0; y1 = int(hor_function(x1))
        x2 = w; y2 = int(hor_function(x2))
        cv2.line(fr, (x1, y1), (x2, y2), (250,0,0), 4)  # draw green line
        cv2.putText(fr, "Horizon", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (250,0,0), 3)

    
    #Cross_points
    if dir_lines["limit"] is not None:
        if dir_lines["right"] is not None:
            cx_r = line_inter_x(dir_lines["limit"], dir_lines["right"])
            cy_r = limit_function(cx_r)
            cv2.circle(fr, (int(cx_r), int(cy_r)), 15, (250,250,0), -1)
        
        if dir_lines["left"] is not None:
            cx_l = line_inter_x(dir_lines["limit"], dir_lines["left"])
            cy_l = limit_function(cx_l)
            cv2.circle(fr, (int(cx_l), int(cy_l)), 15, (250,0,250), -1)
            
    return fr