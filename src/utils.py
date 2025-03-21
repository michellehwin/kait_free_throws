import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
import traceback 

import itertools
import math
from statistics import mean
from statistics import stdev

import sys

import globals



######################################################### general functions #############################################################


def read_raw_data(data_export, file_name):


    if data_export == globals.FE_data_export:

        player = file_name.split('/')[6][:-11]

        df = pd.read_csv(file_name, skiprows=6)

        if player == 'UTFE11':
            col_list = list(df.columns)[0:2] + list(df.columns)[363:366] + list(df.columns[409:422])
            # print(col_list)
            col_name_mapper = dict(zip(col_list, globals.column_names_alt))
            df = df.rename(col_name_mapper, axis=1)
            df = df[globals.column_names_alt]
        elif player == 'UTFE12':
            col_list = list(df.columns)[0:2] + list(df.columns)[25:28] + list(df.columns[9:22])
            # print(col_list)
            col_name_mapper = dict(zip(col_list, globals.column_names_alt))
            df = df.rename(col_name_mapper, axis=1)
            df = df[globals.column_names_alt]
        else:
            col_list = list(df.columns)[0:2] + list(df.columns[6:9]) + list(df.columns[52:65])
            # print(col_list)
            col_name_mapper = dict(zip(col_list, globals.column_names_alt))
            df = df.rename(col_name_mapper, axis=1)
            df = df[globals.column_names_alt]

        return df
    
    elif data_export == globals.MN_data_export:

        player = file_name.split('/')[6][:-11]

        df = pd.read_csv(file_name, skiprows=6)
        
        col_list = list(df.columns)[0:2] + list(df.columns)[370:373] + list(df.columns[424:443])
        # print(col_list)
        col_name_mapper = dict(zip(col_list, globals.column_names_alt_2))
        df = df.rename(col_name_mapper, axis=1)
        df = df[globals.column_names_alt_2]

        return df
    
    elif data_export == globals.TR_FEB_23_data_export:

        player = file_name.split('/')[6][:-11]
        
        if player in ['TR01', 'TR12']:
            
            df = pd.read_csv(file_name, skiprows=6)
        
            col_list = list(df.columns)[0:2] + list(df.columns)[6:9] + list(df.columns[52:65])
            # print(col_list)
            col_name_mapper = dict(zip(col_list, globals.column_names_alt))
            df = df.rename(col_name_mapper, axis=1)
            df = df[globals.column_names_alt]

            return df

        elif player in ['TR07', 'TR10', 'TR20', 'TR21']:

            df = pd.read_csv(file_name, skiprows=6)
        
            col_list = list(df.columns)[0:2] + list(df.columns)[25:28] + list(df.columns[9:22])
            # print(col_list)
            col_name_mapper = dict(zip(col_list, globals.column_names_alt))
            df = df.rename(col_name_mapper, axis=1)
            df = df[globals.column_names_alt]

            return df


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def convert_m_to_ft(x):

    return (x / 1000) * 3.28084


######################################################### estimation functions ########################################################


def fit_x(t, vx, x0, ax):
    return vx * t + x0 + 0.5 * (ax) * t ** 2

def fit_y(t, vy, y0, ay):
    return vy * t + y0 + .5 * (ay) * t ** 2

def fit_z(t, vz, z0, az):
    return vz * t + z0 - 0.5 * (az) * t ** 2


def arc_coordinates(x, y, z, t):

    fitted_x = []
    fitted_y = []
    fitted_z = []

    # find entry time, where z = 10

    # fit a polynomial onto the z coordinates as a function of time
    coeffs_z = poly.polyfit(t, z, 2)

    # create a list of time values after the end of the cropped t data to check for when the ball crosses z = 10ft
    t_extended = list(np.linspace(t[-1], t[-1] + 1, num=len(t) * 100))

    diff_z = 1000
    estimated_entry_t = 0
    estimated_entry_z = 0

    # loop through the additional time, checking the difference between the estimated z and 10ft at each point
    for i in range(len(t_extended)):

        cur_z = poly.polyval(t_extended[i], coeffs_z)
        cur_diff_z = abs(cur_z - 10)

        if cur_diff_z < diff_z:
            diff_z = cur_diff_z
            estimated_entry_t = t_extended[i]
            estimated_entry_z = cur_z

    # given frame rate of fps and the additional time to entry, find the additional number of data points needed to estimate
    additional_time_to_entry = estimated_entry_t - t[-1]
    additional_frames = 240 * additional_time_to_entry

    print('End of Crop Z:', z[-1])
    print('Estimated Entry Z:', estimated_entry_z)
    print('End of Crop T:', t[-1])
    print('Estimated Entry T:', estimated_entry_t)
    print('Additonal Time to Entry:', additional_time_to_entry)
    print('Additional Frames:', additional_frames)

    # produce the full polynomial from release to entry
    additional_points = round(additional_frames)
    t_extended_to_entry = list(np.linspace(t[0], estimated_entry_t, num=len(t) + additional_points))

    # z
    coeffs_z = poly.polyfit(t, z, 2)
    for i in range (len(t_extended_to_entry)):
        fitted_z.append(poly.polyval(t_extended_to_entry[i], coeffs_z))

    # y
    coeffs_y = poly.polyfit(t, y, 2)
    for i in range (len(t_extended_to_entry)):
        fitted_y.append(poly.polyval(t_extended_to_entry[i], coeffs_y))

    # x
    coeffs_x = poly.polyfit(t, x, 2)
    for i in range (len(t_extended_to_entry)):
        fitted_x.append(poly.polyval(t_extended_to_entry[i], coeffs_x))

    
    return fitted_x, fitted_y, fitted_z, t_extended_to_entry    
    


def return_vector(x2, x1, z2, z1, y2, y1):

    return [x2 - x1, z2 - z1, y2 - y1]

def vector_magnitude(vector):

    return np.linalg.norm(np.array(vector))

def unit_vector(vector, mag):

    #return [(x / mag) for x in vector]
    return vector / mag


def release_velocity(x_poly, y_poly, z_poly, t_poly):

    x1 = x_poly[0]
    x2 = x_poly[1]
    
    y1 = y_poly[0]
    y2 = y_poly[1]

    z1 = z_poly[0]
    z2 = z_poly[1]

    t1 = t_poly[0]
    t2 = t_poly[1]
    
    dt = (t2 - t1)
    dx = (x2 - x1) / dt
    dy = (y2 - y1) / dt
    dz = (z2 - z1) / dt  
    # print('\n-------------------------------\n')
    # print(f'DT: {dt}\nDX: {dx}\nDY: {dy}\nDZ: {dz}\n')  
    # print('\n-------------------------------\n')

    v = math.sqrt((dx**2) + (dy**2) + (dz**2))

    return v

def release_angle(x_poly, y_poly, z_poly):

    x1 = x_poly[0]
    x2 = x_poly[1]

    y1 = y_poly[0]
    y2 = y_poly[1]

    z1 = z_poly[0]
    z2 = z_poly[1]
    
    dx = (x2 - x1)
    dy = (y2 - y1) 
    dz = (z2 - z1)

    dxy = math.sqrt((dx**2) + (dy**2)) 

    angle = math.degrees(math.atan(dz / dxy))

    return angle

def entry_angle(x_poly, y_poly, z_poly):

    x1 = x_poly[-1]
    x2 = x_poly[-2]

    y1 = y_poly[-1]
    y2 = y_poly[-2]

    z1 = z_poly[-1]
    z2 = z_poly[-2]
    
    dx = (x2 - x1)
    dy = (y2 - y1) 
    dz = (z2 - z1)

    dxy = math.sqrt((dx**2) + (dy**2)) 

    angle = math.degrees(math.atan(dz / dxy))

    if angle < 0:
        angle *= -1

    return angle

def entry_direction(Rx, Ry, Ex, Ey, Cx, Cy):

    # Cy = globals.center_of_rim_coords[1]
    # Cx = globals.center_of_rim_coords[0]

    # Ry = y_poly[0]
    # Rx = x_poly[0]

    # Ey = y_poly[-1]
    # Ex = x_poly[-1]

    RC_Vector = return_vector(Cx, Rx, Cy, Ry, 0, 0)
    RC_magnitude = vector_magnitude(RC_Vector)
    RC_Unit_Vector = list(unit_vector(RC_Vector, RC_magnitude))

    # print('RC Vector: {}'.format(RC_Vector))
    # print('RC Magnitude: {}'.format(RC_magnitude))
    # print('RC Unit Vector: {}'.format(RC_Unit_Vector))

    Up_Vector = [0, 0, 1]
    RP_Vector = list(np.cross(RC_Unit_Vector, Up_Vector))

    # print('RP Vector: {}'.format(RP_Vector))

    RE_Vector = return_vector(Ex, Rx, Ey, Ry, 0, 0)
    RE_magnitude = vector_magnitude(RE_Vector)
    RE_Unit_Vector = list(unit_vector(RE_Vector, RE_magnitude))

    # print('RE Vector: {}'.format(RE_Vector))
    # print('RE Magnitude: {}'.format(RE_magnitude))
    # print('RE Unit Vector: {}'.format(RE_Unit_Vector))

    dot = np.dot(RP_Vector, RE_Unit_Vector)

    # print('Dot Product: {}'.format(dot))

    direction = ''

    if dot > 0:
        direction = 'Right'

    if dot < 0:
        direction = 'Left'
    
    if dot == 0:
        direction = 'Center'

    return direction

def left_right(Rx, Ry, Ex, Ey, Cx, Cy, direction):

    # Cy = globals.center_of_rim_coords[1]
    # Cx = globals.center_of_rim_coords[0]

    # Ry = y_poly[0]
    # Rx = x_poly[0]

    # Ey = y_poly[-1]
    # Ex = x_poly[-1]

    RC_Vector = return_vector(Cx, Rx, Cy, Ry, 0, 0)
    RC_magnitude = vector_magnitude(RC_Vector)
    RC_Unit_Vector = list(unit_vector(RC_Vector, RC_magnitude))

    RE_Vector = return_vector(Ex, Rx, Ey, Ry, 0, 0)
    RE_magnitude = vector_magnitude(RE_Vector)
    RE_Unit_Vector = list(unit_vector(RE_Vector, RE_magnitude))

    dot = np.dot(RE_Unit_Vector, RC_Unit_Vector)

    theta = math.degrees(math.acos(dot))

    # print('Left Right Angle: {}'.format(theta))

    d = math.sin(math.radians(theta)) * RE_magnitude

    if direction == 'Left':
        d = d * -1

    return d


def front_back(Rx, Ry, Ex, Ey, Cx, Cy, Sx, Sy):

    # Cy = globals.center_of_rim_coords[1]
    # Cx = globals.center_of_rim_coords[0]

    # Ry = y_poly[0]
    # Rx = x_poly[0]

    # Ey = y_poly[-1]
    # Ex = x_poly[-1]

    # Sy = ss_coords[1]
    # Sx = ss_coords[0]

    SR_Vector = return_vector(Sx, Rx, Sy, Ry, 0, 0)
    SR_magnitude = vector_magnitude(SR_Vector)
    SR_Unit_Vector = list(unit_vector(SR_Vector, SR_magnitude))

    # print('SR Vector: {}'.format(SR_Vector))
    # print('SR Magnitude: {}'.format(SR_magnitude))
    # print('SR Unit Vector: {}'.format(SR_Unit_Vector))

    SE_Vector = return_vector(Ex, Sx, Ey, Sy, 0, 0)
    
    # print('SE Vector: {}'.format(SE_Vector))

    d = np.dot(SR_Unit_Vector, SE_Vector)

    return d

def short_long(d):

    sl = ''

    if d > 0:
        sl = 'Long'
    else:
        sl = 'Short'
    
    return sl

def shot_distance(Rx, Ry, Cx, Cy):

    # Cy = globals.center_of_rim_coords[1]
    # Cx = globals.center_of_rim_coords[0]
 
    # Ry = y_poly[0]
    # Rx = x_poly[0]

    dy = Cy - Ry
    dx = Cx - Rx

    d = math.sqrt((dy**2) + (dx**2))

    return d


def findCircle(x1, y1, x2, y2, x3, y3) :
    
    x12 = x1 - x2
    x13 = x1 - x3
 
    y12 = y1 - y2
    y13 = y1 - y3
 
    y31 = y3 - y1
    y21 = y2 - y1
 
    x31 = x3 - x1
    x21 = x2 - x1
 
    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)
 
    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)
 
    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)
 
    f = (((sx13) * (x12) + (sy13) * (x12) + (sx21) * (x13) + (sy21) * (x13)) / (2 * ((y31) * (x12) - (y21) * (x13))))
             
    g = (((sx13) * (y12) + (sy13) * (y12) + (sx21) * (y13) + (sy21) * (y13)) / (2 * ((x31) * (y12) - (x21) * (y13))))
 
    c = (-pow(x1, 2) - pow(y1, 2) - 2 * g * x1 - 2 * f * y1)
 
    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c
 
    # r is the radius
    r = round(math.sqrt(sqr_of_r), 5)
 
    # print("Centre = (", h, ", ", k, ")")
    # print("Radius = ", r)

    return [h, k], r


def find_rim_coords(data_export, file_name):

    df = read_raw_data(data_export, file_name)

    if data_export == globals.FE_data_export:

        rim_marker_1_x = df['Rim:Marker001_Position_X'].mean()
        rim_marker_1_y = df['Rim:Marker001_Position_Z'].mean()
        rim_marker_1_z = df['Rim:Marker001_Position_Y'].mean()

        rim_marker_2_x = df['Rim:Marker002_Position_X'].mean()
        rim_marker_2_y = df['Rim:Marker002_Position_Z'].mean()
        rim_marker_2_z = df['Rim:Marker002_Position_Y'].mean()

        rim_marker_3_x = df['Rim:Marker003_Position_X'].mean()
        rim_marker_3_y = df['Rim:Marker003_Position_Z'].mean()
        rim_marker_3_z = df['Rim:Marker003_Position_Y'].mean()

        rim_marker_4_x = df['Rim:Marker004_Position_X'].mean()
        rim_marker_4_y = df['Rim:Marker004_Position_Z'].mean()
        rim_marker_4_z = df['Rim:Marker004_Position_Y'].mean()

        print('\n\n----------------\n\nHOOP MARKERS:\n\n----------------\n\n')
        print(f'MARKER 1: ({rim_marker_1_x}, {rim_marker_1_y}, {rim_marker_1_z})')
        print(f'MARKER 2: ({rim_marker_2_x}, {rim_marker_2_y}, {rim_marker_2_z})')
        print(f'MARKER 3: ({rim_marker_3_x}, {rim_marker_3_y}, {rim_marker_3_z})')
        print(f'MARKER 4: ({rim_marker_4_x}, {rim_marker_4_y}, {rim_marker_4_z})')

        # rim_coords_list_z = [rim_marker_1_z, rim_marker_4_z]
        # rim_coords_z = convert_m_to_ft(mean(rim_coords_list_z))
        # print(rim_coords_z)

        rim_coords_z = 10

        rim_coords_list_x = []
        rim_coords_list_y = []
        radius_list = []

        coords, r = findCircle(rim_marker_1_x, rim_marker_1_y, rim_marker_2_x, rim_marker_2_y, rim_marker_3_x, rim_marker_3_y)
        rim_coords_list_x.append(coords[0])
        rim_coords_list_y.append(coords[1])
        radius_list.append(r)

        coords, r = findCircle(rim_marker_1_x, rim_marker_1_y, rim_marker_2_x, rim_marker_2_y, rim_marker_4_x, rim_marker_4_y)
        rim_coords_list_x.append(coords[0])
        rim_coords_list_y.append(coords[1])
        radius_list.append(r)

        coords, r = findCircle(rim_marker_1_x, rim_marker_1_y, rim_marker_3_x, rim_marker_3_y, rim_marker_4_x, rim_marker_4_y)
        rim_coords_list_x.append(coords[0])
        rim_coords_list_y.append(coords[1])
        radius_list.append(r)

        coords, r = findCircle(rim_marker_4_x, rim_marker_4_y, rim_marker_2_x, rim_marker_2_y, rim_marker_3_x, rim_marker_3_y)
        rim_coords_list_x.append(coords[0])
        rim_coords_list_y.append(coords[1])
        radius_list.append(r)


        rim_coords_x = convert_m_to_ft(mean(rim_coords_list_x))
        print(rim_coords_x)

        rim_coords_y = convert_m_to_ft(mean(rim_coords_list_y))
        print(rim_coords_y)

        # radius = convert_m_to_ft(mean(radius_list))
        radius = .75
        print(radius)

        return rim_coords_x, rim_coords_y, rim_coords_z, radius
    
    elif data_export == globals.MN_data_export:

        # find rim center
        rim_marker_1_x = df['Rim:Marker001_Position_X'].mean()
        rim_marker_1_y = df['Rim:Marker001_Position_Z'].mean()
        rim_marker_1_z = df['Rim:Marker001_Position_Y'].mean()

        rim_marker_2_x = df['Rim:Marker002_Position_X'].mean()
        rim_marker_2_y = df['Rim:Marker002_Position_Z'].mean()
        rim_marker_2_z = df['Rim:Marker002_Position_Y'].mean()

        rim_marker_3_x = df['Rim:Marker003_Position_X'].mean()
        rim_marker_3_y = df['Rim:Marker003_Position_Z'].mean()
        rim_marker_3_z = df['Rim:Marker003_Position_Y'].mean()

        rim_marker_4_x = df['Rim:Marker004_Position_X'].mean()
        rim_marker_4_y = df['Rim:Marker004_Position_Z'].mean()
        rim_marker_4_z = df['Rim:Marker004_Position_Y'].mean()

        rim_marker_5_x = df['Rim:Marker005_Position_X'].mean()
        rim_marker_5_y = df['Rim:Marker005_Position_Z'].mean()
        rim_marker_5_z = df['Rim:Marker005_Position_Y'].mean()

        rim_marker_6_x = df['Rim:Marker006_Position_X'].mean()
        rim_marker_6_y = df['Rim:Marker006_Position_Z'].mean()
        rim_marker_6_z = df['Rim:Marker006_Position_Y'].mean()

        print('\n\n----------------\n\nHOOP MARKERS:\n\n----------------\n\n')
        print(f'MARKER 1: ({rim_marker_1_x}, {rim_marker_1_y}, {rim_marker_1_z})')
        print(f'MARKER 2: ({rim_marker_2_x}, {rim_marker_2_y}, {rim_marker_2_z})')
        print(f'MARKER 3: ({rim_marker_3_x}, {rim_marker_3_y}, {rim_marker_3_z})')
        print(f'MARKER 4: ({rim_marker_4_x}, {rim_marker_4_y}, {rim_marker_4_z})')
        print(f'MARKER 5: ({rim_marker_5_x}, {rim_marker_5_y}, {rim_marker_5_z})')
        print(f'MARKER 6: ({rim_marker_6_x}, {rim_marker_6_y}, {rim_marker_6_z})')

        rim_marker_1_list = [rim_marker_1_x, rim_marker_1_y, rim_marker_1_z]
        rim_marker_2_list = [rim_marker_2_x, rim_marker_2_y, rim_marker_2_z]
        rim_marker_3_list = [rim_marker_3_x, rim_marker_3_y, rim_marker_3_z]
        rim_marker_4_list = [rim_marker_4_x, rim_marker_4_y, rim_marker_4_z]
        rim_marker_5_list = [rim_marker_5_x, rim_marker_5_y, rim_marker_5_z]
        rim_marker_6_list = [rim_marker_6_x, rim_marker_6_y, rim_marker_6_z]

        rim_marker_list = [rim_marker_1_list, rim_marker_2_list, rim_marker_3_list, rim_marker_4_list, rim_marker_5_list, rim_marker_6_list]

        # rim_coords_list_z = [rim_marker_1_z, rim_marker_4_z]
        # rim_coords_z = convert_m_to_ft(mean(rim_coords_list_z))
        # print(rim_coords_z)

        rim_coords_z = 10

        rim_coords_list_x = []
        rim_coords_list_y = []
        radius_list = []

        # for elem in itertools.combinations([1,2,3,4,5,6], 3):

        #     print(elem)

        keep_comb = [[1, 2, 3], [1, 2, 5], [1, 3, 4],
                     [1, 3, 5], [1, 3, 6], [1, 4, 6],
                     [2, 3, 5], [3, 4, 6]]


        for elem in itertools.combinations(rim_marker_list, 3):
            comb_cur = [(rim_marker_list.index(elem[0]) + 1), (rim_marker_list.index(elem[1]) + 1), (rim_marker_list.index(elem[2]) + 1)]
            if comb_cur in keep_comb:
                print(f'P{(rim_marker_list.index(elem[0]) + 1)}: {elem[0]}')
                print(f'P{(rim_marker_list.index(elem[1]) + 1)}: {elem[1]}')
                print(f'P{(rim_marker_list.index(elem[2]) + 1)}: {elem[2]}')
                coords, r = findCircle(elem[0][0], elem[0][1], elem[1][0], elem[1][1], elem[2][0], elem[2][1])
                rim_coords_list_x.append(coords[0])
                rim_coords_list_y.append(coords[1])
                radius_list.append(r)
            else:
                continue

        rim_coords_x = convert_m_to_ft(mean(rim_coords_list_x))
        print(rim_coords_x)

        rim_coords_y = convert_m_to_ft(mean(rim_coords_list_y))
        print(rim_coords_y)

        # radius = convert_m_to_ft(mean(radius_list))
        radius = .75
        print(radius)

        return rim_coords_x, rim_coords_y, rim_coords_z, radius


def sweet_spot_angle(x_poly, y_poly, rim_coords_x, rim_coords_y):

    Cy = rim_coords_y
    Cx = rim_coords_x

    Ry = y_poly[0]
    Rx = x_poly[0]

    ss_dy = Cy - Ry
    ss_dx = Cx - Rx

    theta = math.degrees(math.atan(ss_dy/ss_dx))

    return theta

def sweet_spot_coords(x_poly, y_poly, rim_coords_x, rim_coords_y, theta):

    Cy = rim_coords_y
    Cx = rim_coords_x

    Ry = y_poly[0]
    Rx = x_poly[0]

    dx = math.cos(math.radians(theta)) * 0.1666667
    dy = math.sin(math.radians(theta)) * 0.1666667

    if Rx > Cx:
        Sx = Cx - dx
        Sy = Cy - dy
    else:
        Sx = Cx + dx
        Sy = Cy + dy

    return Sx, Sy


def metrics(data_export, file_name):

    df = read_raw_data(data_export, file_name)

    player = file_name.split('/')[6][:-11]
    shot_att = file_name.split('/')[6][:-4]

    print('-------- Dropping Duplicates ----------')

    print(len(df))

    df = df.drop_duplicates(subset=['Ball_Position_X', 'Ball_Position_Y', 'Ball_Position_Z'])

    print(len(df))

    x_pos_array = df['Ball_Position_X'].apply(lambda x: convert_m_to_ft(x)).to_numpy() 
    y_pos_array = df['Ball_Position_Z'].apply(lambda x: convert_m_to_ft(x)).to_numpy()
    z_pos_array = df['Ball_Position_Y'].apply(lambda x: convert_m_to_ft(x)).to_numpy()
    time_seq = df['Time'].to_numpy()
    frame_seq = df['Frame'].to_numpy()
    

    diffs = np.abs(np.diff(x_pos_array, prepend=0, append=0))
    keep_idx = np.where((diffs[:-1] <= 5) | (diffs[1:] <= 5))[0]
    x_pos_array = x_pos_array[keep_idx]
    y_pos_array = y_pos_array[keep_idx]
    z_pos_array = z_pos_array[keep_idx]
    time_seq = time_seq[keep_idx]
    frame_seq = frame_seq[keep_idx]

    # changed it to 6ft because 5ft was causing some noise in finding the release frame
    five_ft = np.argmax(z_pos_array > 6)
    x_pos_array = x_pos_array[five_ft:]
    y_pos_array = y_pos_array[five_ft:]
    z_pos_array = z_pos_array[five_ft:]
    time_seq = time_seq[five_ft:]
    frame_seq = frame_seq[five_ft:]

    v_x_array = np.gradient(x_pos_array, time_seq, edge_order=1)
    v_y_array = np.gradient(y_pos_array, time_seq, edge_order=1)
    v_z_array = np.gradient(z_pos_array, time_seq, edge_order=1)
    v_sum_array = v_x_array + v_z_array + v_y_array

    
    # instead of checking from the beginning to the apex of the shot, I only check from the beginning to the first time the z coordinate passes 10ft
    # this is because there is some data where there are gaps near the apex
    good_start = False
    try:
        z_apex = np.argmax(z_pos_array)
        releas_window_idx = np.where((z_pos_array > 9.9) & (z_pos_array < 10.1))
        # start = np.argmax(v_sum_array[:z_apex])
        start = np.argmax(v_sum_array[:releas_window_idx[0][0]])
        good_start = True
    except Exception as e:
        if True:
            print(f"Error finding start: {e}")
            traceback.print_exc()
        start = 0

    frame_start = frame_seq[start]

    good_stop = False
    stop = -1
    for j in range(start, len(z_pos_array)-1):

        # check if it hit the back board and bounced back, stop it before it hits the back borad

        if (y_pos_array[j + 1] < y_pos_array[j]) and (y_pos_array[j + 2] < y_pos_array[j + 1]) and (y_pos_array[j + 3] < y_pos_array[j + 2]) and (y_pos_array[j + 4] < y_pos_array[j + 3]) and (j > z_apex):
            print('bounced backwards ----------')
            stop = j
            good_stop = True
            break

        if z_pos_array[j] > 10.45 and z_pos_array[j + 1] <= 10.45:
            stop = j
            good_stop = True
            break

    if stop == -1:
        stop = len(z_pos_array) - 1

    # for i in range (10):
    #     print(f'post stop frame{i} ----------')
    #     print(f'x: {x_pos_array[j + i]}\ny: {y_pos_array[j + i]}\nz: {z_pos_array[j + i]}')

    frame_stop = frame_seq[stop]

    if not (good_start and good_stop):
        print('BAD DATA - Check Graph')

    print(frame_start)
    print(frame_stop)

   
    t = time_seq[start:stop] - time_seq[start]
    x = x_pos_array[start:stop]
    y = y_pos_array[start:stop]
    z = z_pos_array[start:stop]

    fitted_x, fitted_y, fitted_z, fitted_t = arc_coordinates(x, y, z, t)

    df_ts_dict = {'t': fitted_t, 'x': fitted_x, 'y': fitted_y, 'z': fitted_z}

    df_ts = pd.DataFrame(df_ts_dict)

    df_ts.to_csv(f'../timeseries/{data_export[-4:]}/{player}/{shot_att}_TS.csv', index=False)

    params_x, cov_x = curve_fit(fit_x, t, x)
    params_y, cov_y = curve_fit(fit_y, t, y)
    params_z, cov_z = curve_fit(fit_z, t, z)

    params = np.concatenate((params_x, params_y, params_z))
    params = {'player': player, 'shot': shot_att, 'vx': params[0], 'release_x_pm': params[1], 'release_x_poly': fitted_x[0], 'ax': params[2], 
              'vy': params[3], 'release_y_pm': params[4], 'release_y_poly': fitted_y[0], 'ay': params[5], 'vz': params[6], 
              'release_z_pm': params[7], 'release_z_poly': fitted_z[0], 'az': params[8], 'frame_start': frame_start, 'frame_stop': frame_stop}
    
    fitted_x_pm = fit_x(t, *params_x)
    fitted_y_pm = fit_y(t, *params_y)
    fitted_z_pm = fit_z(t, *params_z)

    rim_coords_x, rim_coords_y, rim_coords_z, radius = find_rim_coords(data_export, file_name)

    sweet_spot_angle_value_pm = sweet_spot_angle(fitted_x_pm, fitted_y_pm, rim_coords_x, rim_coords_y)
    Sx_pm, Sy_pm = sweet_spot_coords(fitted_x_pm, fitted_y_pm, rim_coords_x, rim_coords_y, sweet_spot_angle_value_pm)

    sweet_spot_angle_value_poly = sweet_spot_angle(fitted_x, fitted_y, rim_coords_x, rim_coords_y)
    Sx_poly, Sy_poly = sweet_spot_coords(fitted_x, fitted_y, rim_coords_x, rim_coords_y, sweet_spot_angle_value_poly)

    print(sweet_spot_angle_value_poly)
    print(Sx_poly, Sy_poly)

    params['sweet_spot_pm'] = [Sx_pm, Sy_pm, rim_coords_z]
    params['sweet_spot_poly'] = [Sx_poly, Sy_poly, fitted_z[-1]]

    parameters = pd.Series(params)

    parameters['apex_t'] = parameters['vz'] / parameters['az']
    parameters['apex_x'] = parameters['release_x_pm'] + parameters['vx'] * parameters['apex_t']
    parameters['apex_y'] = parameters['release_y_pm'] + parameters['vy'] * parameters['apex_t']
    parameters['apex_z_pm'] = parameters['release_z_pm'] + parameters['vz'] * parameters['apex_t'] - 0.5 * parameters['az'] * parameters['apex_t']**2
    parameters['apex_z_poly'] = max(fitted_z)
    parameters['entry_t_pm'] = (parameters['vz'] + np.sqrt(parameters['vz']**2 - 2 * parameters['az'] * 10 + 2 * parameters['az'] * parameters['release_z_pm'])) / parameters['az']
    parameters['entry_t_poly'] = fitted_t[-1]
    parameters['entry_x_pm'] = parameters['release_x_pm'] + parameters['vx'] * parameters['entry_t_pm']
    parameters['entry_x_poly'] = fitted_x[-1]
    parameters['entry_y_pm'] = parameters['release_y_pm'] + parameters['vy'] * parameters['entry_t_pm']
    parameters['entry_y_poly'] = fitted_y[-1]
    parameters['entry_z_pm'] = rim_coords_z
    parameters['entry_z_poly'] = fitted_z[-1]

    sweet_spot_x_pm = parameters['sweet_spot_pm'][0]
    sweet_spot_y_pm = parameters['sweet_spot_pm'][1]
    sweet_spot_z_pm = parameters['sweet_spot_pm'][2]
    sweet_spot_x_poly = parameters['sweet_spot_poly'][0]
    sweet_spot_y_poly = parameters['sweet_spot_poly'][1]
    sweet_spot_z_poly = parameters['sweet_spot_poly'][2]
    parameters['dist_from_sweet_pm'] = np.sqrt((parameters['entry_x_pm'] - sweet_spot_x_pm)**2 + (parameters['entry_y_pm'] - sweet_spot_y_pm)**2 + (parameters['entry_z_pm'] - sweet_spot_z_pm)**2)
    parameters['dist_from_sweet_poly'] = np.sqrt((parameters['entry_x_poly'] - sweet_spot_x_poly)**2 + (parameters['entry_y_poly'] - sweet_spot_y_poly)**2 + (parameters['entry_z_poly'] - sweet_spot_z_poly)**2)
    parameters['release_v_mag_pm'] = (parameters['vx']**2 + parameters['vy']**2 + parameters['vz']**2)**0.5
    parameters['release_v_mag_poly'] = release_velocity(fitted_x, fitted_y, fitted_z, fitted_t)
    parameters['launch_angle_pm'] = 90 - np.arccos(parameters['vz'] / parameters['release_v_mag_pm']) * 180 / np.pi
    parameters['launch_angle_poly'] = release_angle(fitted_x, fitted_y, fitted_z)
    parameters['off_center_angle_pm'] = np.arctan(parameters['vy'] / parameters['vx']) * 180 / np.pi
    parameters['off_center_angle_poly'] = sweet_spot_angle_value_poly
    parameters['vz_f'] = parameters['vz'] - parameters['az'] * parameters['entry_t_pm']
    parameters['entry_angle_pm'] = np.arctan(parameters['vz_f'] / parameters['release_v_mag_pm']) * 180 / np.pi
    parameters['entry_angle_poly'] = entry_angle(fitted_x, fitted_y, fitted_z)

    parameters['entry_direction_pm'] = entry_direction(fitted_x_pm[0], fitted_y_pm[0], parameters['entry_x_pm'], parameters['entry_y_pm'], rim_coords_x, rim_coords_y)
    parameters['entry_direction_poly'] = entry_direction(fitted_x[0], fitted_y[0], parameters['entry_x_poly'], parameters['entry_y_poly'], rim_coords_x, rim_coords_y)
    parameters['left_right_pm'] = left_right(fitted_x_pm[0], fitted_y_pm[0], parameters['entry_x_pm'], parameters['entry_y_pm'], rim_coords_x, rim_coords_y, parameters['entry_direction_pm']) * 12
    parameters['left_right_poly'] = left_right(fitted_x[0], fitted_y[0], parameters['entry_x_poly'], parameters['entry_y_poly'], rim_coords_x, rim_coords_y, parameters['entry_direction_poly']) * 12
    parameters['front_back_pm'] = front_back(fitted_x_pm[0], fitted_y_pm[0], parameters['entry_x_pm'], parameters['entry_y_pm'], rim_coords_x, rim_coords_y, parameters['sweet_spot_pm'][0], parameters['sweet_spot_pm'][1]) * 12
    parameters['front_back_poly'] = front_back(fitted_x[0], fitted_y[0], parameters['entry_x_poly'], parameters['entry_y_poly'], rim_coords_x, rim_coords_y, parameters['sweet_spot_poly'][0], parameters['sweet_spot_poly'][1]) * 12
    parameters['short_long_pm'] = short_long(parameters['front_back_pm'])
    parameters['short_long_poly'] = short_long(parameters['front_back_poly'])
    parameters['release_height_pm'] = fitted_z_pm[0]
    parameters['release_height_poly'] = fitted_z[0]
    parameters['shot_distance_pm'] = shot_distance(fitted_x_pm[0], fitted_y_pm[0], rim_coords_x, rim_coords_y)
    parameters['shot_distance_poly'] = shot_distance(fitted_x[0], fitted_y[0], rim_coords_x, rim_coords_y)

    return parameters.to_dict()    

######################################################### graph functions #############################################################


def graph_shot_yz(data1, data2):

    # data1 = [convert_m_to_ft(i) for i in data1]
    # data2 = [convert_m_to_ft(i) for i in data2]
	
    fig = plt.figure(figsize = (35, 20))
    # fig = plt.figure()
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.scatter(data1, data2, c = 'mediumslateblue', edgecolors = 'ghostwhite')
    
    # x_lim, y_lim = x_y_lim(shot_type)
    x_lim = [-15, 20]
    y_lim = [0, 20]

    x_range = np.arange(x_lim[0], x_lim[1], step=1)
    y_range = np.arange(y_lim[0], y_lim[1], step=1)

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.xticks(x_range)
    plt.yticks(y_range)

    plt.vlines(x_range, ymax=y_lim[1], ymin=y_lim[0], linewidths=0.2, linestyles='--', colors='ghostwhite')
    plt.hlines(y_range, xmax=x_lim[1], xmin=x_lim[0], linewidths=0.2, linestyles='--', colors='ghostwhite')

    return fig

def graph_shot_yz_overlap(data_under1, data_under2, data_over1, data_over2):

    # data_under1 = [convert_m_to_ft(i) for i in data_under1]
    # data_under2 = [convert_m_to_ft(i) for i in data_under2]

    # data_over1 = [convert_m_to_ft(i) for i in data_over1]
    # data_over2 = [convert_m_to_ft(i) for i in data_over2]
	
    fig = plt.figure(figsize = (35, 20))
    plt.style.use('dark_background')
    ax = plt.gca()
    
    plt.plot(data_under1, data_under2, c='mediumslateblue', marker='o', mec='ghostwhite')
    plt.plot(data_over1, data_over2, c='aqua')
    
    #x_lim, y_lim = x_y_lim(shot_type)
    x_lim = [-15, 20]
    y_lim = [0, 20]

    x_range = np.arange(x_lim[0], x_lim[1], step=1)
    y_range = np.arange(y_lim[0], y_lim[1], step=1)

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.xticks(x_range)
    plt.yticks(y_range)

    plt.vlines(x_range, ymax=y_lim[1], ymin=y_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')
    plt.hlines(y_range, xmax=x_lim[1], xmin=x_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')
    # for line in x_range:
    #     plt.vlines()

    return fig

def graph_shot_xy(data1, data2):

    # data1 = [convert_m_to_ft(i) for i in data1]
    # data2 = [convert_m_to_ft(i) for i in data2]
	
    fig = plt.figure(figsize = (20, 35))
    # fig = plt.figure()
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.scatter(data1, data2, c = 'mediumslateblue', edgecolors = 'ghostwhite')
    
    # x_lim, y_lim = x_y_lim(shot_type)
    x_lim = [-10, 10]
    y_lim = [-15, 20]

    x_range = np.arange(x_lim[0], x_lim[1], step=1)
    y_range = np.arange(y_lim[0], y_lim[1], step=1)

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.xticks(x_range)
    plt.yticks(y_range)

    plt.vlines(x_range, ymax=y_lim[1], ymin=y_lim[0], linewidths=0.2, linestyles='--', colors='ghostwhite')
    plt.hlines(y_range, xmax=x_lim[1], xmin=x_lim[0], linewidths=0.2, linestyles='--', colors='ghostwhite')

    return fig


def graph_shot_tz(data1, data2):

    # data1 = [convert_m_to_ft(i) for i in data1]
    # data2 = [convert_m_to_ft(i) for i in data2]
	
    fig = plt.figure(figsize = (20, 35))
    # fig = plt.figure()
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.scatter(data1, data2, c = 'mediumslateblue', edgecolors = 'ghostwhite')
    
    # x_lim, y_lim = x_y_lim(shot_type)
    # x_lim = [-10, 10]
    # y_lim = [-15, 20]

    # x_range = np.arange(x_lim[0], x_lim[1], step=1)
    # y_range = np.arange(y_lim[0], y_lim[1], step=1)

    # plt.xlim(x_lim[0], x_lim[1])
    # plt.ylim(y_lim[0], y_lim[1])
    # plt.xticks(x_range)
    # plt.yticks(y_range)

    # plt.vlines(x_range, ymax=y_lim[1], ymin=y_lim[0], linewidths=0.2, linestyles='--', colors='ghostwhite')
    # plt.hlines(y_range, xmax=x_lim[1], xmin=x_lim[0], linewidths=0.2, linestyles='--', colors='ghostwhite')

    return fig


def graph_shot_xy_overlap(data_under1, data_under2, data_over1, data_over2):

    # data_under1 = [convert_m_to_ft(i) for i in data_under1]
    # data_under2 = [convert_m_to_ft(i) for i in data_under2]

    # data_over1 = [convert_m_to_ft(i) for i in data_over1]
    # data_over2 = [convert_m_to_ft(i) for i in data_over2]
	
    fig = plt.figure(figsize = (20, 35))
    plt.style.use('dark_background')
    ax = plt.gca()
    
    plt.plot(data_under1, data_under2, c='mediumslateblue', marker='o', mec='ghostwhite')
    plt.plot(data_over1, data_over2, c='aqua')
    
    #x_lim, y_lim = x_y_lim(shot_type)
    x_lim = [-10, 10]
    y_lim = [-15, 20]

    x_range = np.arange(x_lim[0], x_lim[1], step=1)
    y_range = np.arange(y_lim[0], y_lim[1], step=1)

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.xticks(x_range)
    plt.yticks(y_range)

    plt.vlines(x_range, ymax=y_lim[1], ymin=y_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')
    plt.hlines(y_range, xmax=x_lim[1], xmin=x_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')
    # for line in x_range:
    #     plt.vlines()

    return fig

def ring_chart(Rx, Ry, Ex, Ey, Sx, Sy, Cx, Cy, r):

    # Rx = convert_m_to_in(Rx)
    # Ry = convert_m_to_in(Ry)
    # Ex = convert_m_to_in(Ex)
    # Ey = convert_m_to_in(Ey)
    # Sx = convert_m_to_in(Sx)
    # Sy = convert_m_to_in(Sy)

    Rx = Rx * 12
    Ry = Ry * 12
    Ex = Ex * 12
    Ey = Ey * 12
    Sx = Sx * 12
    Sy = Sy * 12
    Cx = Cx * 12
    Cy = Cy * 12
    r = r * 12
    

    # center_in = [convert_m_to_in(x) for x in globals.center_of_rim_coords]
    # rim_radius_in = convert_m_to_in(globals.rim_radius)

    fig = plt.figure(figsize = (20, 20))
    plt.style.use('dark_background')
    circle = plt.Circle((Cx, Cy), r, color='orange', linewidth = 2, fill=False)
    ax = plt.gca()
    ax.add_artist(circle)

    x_lim, y_lim = [-20, 20], [Cy - 20, Cy + 20]

    x_range = np.arange(x_lim[0], x_lim[1], step=1)
    y_range = np.arange(y_lim[0], y_lim[1], step=1)

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.xticks(x_range)
    plt.yticks(y_range)

    plt.vlines(x_range, ymax=y_lim[1], ymin=y_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')
    plt.hlines(y_range, xmax=x_lim[1], xmin=x_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')


    plt.plot([Rx, Sx], [Ry, Sy], c='darkslateblue', linewidth=0.75)

    ax.scatter([Rx], [Ry], c='tomato', edgecolors = 'ghostwhite')
    ax.scatter([Ex], [Ey], c='mediumspringgreen', edgecolors = 'ghostwhite')
    ax.scatter([Sx], [Sy], c='mediumslateblue', edgecolors = 'ghostwhite')
    ax.scatter([Cx], [Cy], c='mediumslateblue', edgecolors = 'ghostwhite')



    return fig

def release_ring_chart(Rx, Ry, Ex, Ey, Sx, Sy, Cx, Cy, r):

    # Rx = convert_m_to_ft(Rx)
    # Ry = convert_m_to_ft(Ry)
    # Ex = convert_m_to_ft(Ex)
    # Ey = convert_m_to_ft(Ey)
    # Sx = convert_m_to_ft(Sx)
    # Sy = convert_m_to_ft(Sy)

    # print(Rx, Ry)
    # print(Ex, Ey)

    # center_in = [convert_m_to_ft(x) for x in globals.center_of_rim_coords]
    # rim_radius_in = convert_m_to_ft(globals.rim_radius)

    fig = plt.figure(figsize = (20, 20))
    plt.style.use('dark_background')
    circle = plt.Circle((Cx, Cy), r, color='orange', linewidth = 2, fill=False)
    ax = plt.gca()
    ax.add_artist(circle)

    x_lim, y_lim = [-10, 10], [-5, 15]

    x_range = np.arange(x_lim[0], x_lim[1], step=1)
    y_range = np.arange(y_lim[0], y_lim[1], step=1)

    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.xticks(x_range)
    plt.yticks(y_range)

    plt.vlines(x_range, ymax=y_lim[1], ymin=y_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')
    plt.hlines(y_range, xmax=x_lim[1], xmin=x_lim[0], linewidths=0.2, linestyles='--',  colors='ghostwhite')


    plt.plot([Rx, Sx], [Ry, Sy], c='darkslateblue', linewidth=0.75)

    ax.scatter([Rx], [Ry], c='tomato', edgecolors = 'ghostwhite')
    ax.scatter([Ex], [Ey], c='mediumspringgreen', edgecolors = 'ghostwhite')
    ax.scatter([Sx], [Sy], c='mediumslateblue', edgecolors = 'ghostwhite')
    ax.scatter([Cx], [Cy], c='mediumslateblue', edgecolors = 'ghostwhite')



    return fig


def graph_shot(graph_type, data1, data2):

    if graph_type == 'yz':
        return graph_shot_yz(data1, data2)

    elif graph_type == 'xy':
        return graph_shot_xy(data1, data2)
    
    elif graph_type == 'tz':
        return graph_shot_tz(data1, data2)
    
def graph_shot_overlap(graph_type, data_under1, data_under2, data_over1, data_over2):

    if graph_type == 'yz':
        return graph_shot_yz_overlap(data_under1, data_under2, data_over1, data_over2)

    elif graph_type == 'xy':
        return graph_shot_xy_overlap(data_under1, data_under2, data_over1, data_over2)


    
def save_graph(graph, file_name):

    graph.savefig(file_name, bbox_inches='tight')




def graph_shot_player(data_export, player, att, file_name):

    df = read_raw_data(data_export, file_name)

    # print(file_name)

    # print(f'{globals.graph_path}/{player}/{player} _{att}_RAW_YZ')

    x_raw = list(df['Ball_Position_X'].apply(lambda x: convert_m_to_ft(x)))
    y_raw = list(df['Ball_Position_Z'].apply(lambda x: convert_m_to_ft(x)))
    z_raw = list(df['Ball_Position_Y'].apply(lambda x: convert_m_to_ft(x)))

    raw_yz_fig = graph_shot('yz', y_raw, z_raw)
    save_graph(raw_yz_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_RAW_YZ')
    plt.close(raw_yz_fig)

    raw_xy_fig = graph_shot('xy', x_raw, y_raw)
    save_graph(raw_xy_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_RAW_XY')
    plt.close(raw_xy_fig)

    
    print('-------- Dropping Duplicates ----------')

    print(len(df))

    df = df.drop_duplicates(subset=['Ball_Position_X', 'Ball_Position_Y', 'Ball_Position_Z'])

    print(len(df))


    x_pos_array = df['Ball_Position_X'].apply(lambda x: convert_m_to_ft(x)).to_numpy() 
    y_pos_array = df['Ball_Position_Z'].apply(lambda x: convert_m_to_ft(x)).to_numpy()
    z_pos_array = df['Ball_Position_Y'].apply(lambda x: convert_m_to_ft(x)).to_numpy()
    time_seq = df['Time'].to_numpy()
    frame_seq = df['Frame'].to_numpy()

    diffs = np.abs(np.diff(x_pos_array, prepend=0, append=0))
    keep_idx = np.where((diffs[:-1] <= 5) | (diffs[1:] <= 5))[0]
    x_pos_array = x_pos_array[keep_idx]
    y_pos_array = y_pos_array[keep_idx]
    z_pos_array = z_pos_array[keep_idx]
    time_seq = time_seq[keep_idx]
    frame_seq = frame_seq[keep_idx]

    five_ft = np.argmax(z_pos_array > 6)
    x_pos_array = x_pos_array[five_ft:]
    y_pos_array = y_pos_array[five_ft:]
    z_pos_array = z_pos_array[five_ft:]
    time_seq = time_seq[five_ft:]
    frame_seq = frame_seq[five_ft:]

    v_x_array = np.gradient(x_pos_array, time_seq, edge_order=1)
    v_y_array = np.gradient(y_pos_array, time_seq, edge_order=1)
    v_z_array = np.gradient(z_pos_array, time_seq, edge_order=1)
    v_sum_array = v_x_array + v_z_array + v_y_array



    
    good_start = False
    try:
        z_apex = np.argmax(z_pos_array)
        releas_window_idx = np.where((z_pos_array > 9.9) & (z_pos_array < 10.1))
        # start = np.argmax(v_sum_array[:z_apex])
        start = np.argmax(v_sum_array[:releas_window_idx[0][0]])
        good_start = True
    except Exception as e:
        if True:
            print(f"Error finding start: {e}")
            traceback.print_exc()
        start = 0

    frame_start = frame_seq[start]

    good_stop = False
    stop = -1
    for j in range(start, len(z_pos_array)-4):

        if (y_pos_array[j + 1] < y_pos_array[j]) and (y_pos_array[j + 2] < y_pos_array[j + 1]) and (y_pos_array[j + 3] < y_pos_array[j + 2]) and (y_pos_array[j + 4] < y_pos_array[j + 3]) and (j > z_apex):
            print('-------- bounced backwards ----------')
            stop = j
            good_stop = True
            break

        if z_pos_array[j] > 10.45 and z_pos_array[j + 1] <= 10.45:
            stop = j
            good_stop = True
            break

    if stop == -1:
        stop = len(z_pos_array) - 1

    # for i in range (10):
    #     print(f'post stop frame{i} ----------')
    #     print(f'x: {x_pos_array[j + i]}\ny: {y_pos_array[j + i]}\nz: {z_pos_array[j + i]}')

    frame_stop = frame_seq[stop]

    if not (good_start and good_stop):
        print('BAD DATA - Check Graph')

    print(frame_start)
    print(frame_stop)

   
    t = time_seq[start:stop] - time_seq[start]
    x = x_pos_array[start:stop]
    y = y_pos_array[start:stop]
    z = z_pos_array[start:stop]

    cropped_yz_fig = graph_shot('yz', y, z)
    save_graph(cropped_yz_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_CROPPED_YZ')
    plt.close(cropped_yz_fig)

    cropped_xy_fig = graph_shot('xy', x, y)
    save_graph(cropped_xy_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_CROPPED_XY')
    plt.close(cropped_xy_fig)


    fitted_x, fitted_y, fitted_z, fitted_t = arc_coordinates(x, y, z, t)

    params = {'x0': fitted_x[0], 'y0': fitted_y[0], 'z0': fitted_z[0], 'frame_start': frame_start, 'frame_stop': frame_stop}

    fitted_yz_fig = graph_shot_overlap('yz', y, z, fitted_y, fitted_z)
    save_graph(fitted_yz_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_FITTED_YZ')
    plt.close(fitted_yz_fig)

    fitted_xy_fig = graph_shot_overlap('xy', x, y, fitted_x, fitted_y)
    save_graph(fitted_xy_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_FITTED_XY')
    plt.close(fitted_xy_fig)    


    rim_coords_x, rim_coords_y, rim_coords_z, radius = find_rim_coords(data_export, file_name)

    sweet_spot_angle_value = sweet_spot_angle(fitted_x, fitted_y, rim_coords_x, rim_coords_y)
    Sx, Sy = sweet_spot_coords(fitted_x, fitted_y, rim_coords_x, rim_coords_y, sweet_spot_angle_value)

    print(sweet_spot_angle_value)
    print(Sx, Sy)

    params['sweet_spot'] = [Sx, Sy, rim_coords_z]


    parameters = pd.Series(params)

    parameters['entryt'] = fitted_t[-1]
    parameters['entryx'] = fitted_x[-1]
    parameters['entryy'] = fitted_y[-1]
    parameters['entryz'] = rim_coords_z


    ring_chart_fig = ring_chart(fitted_x[0], fitted_y[0], parameters['entryx'], parameters['entryy'], Sx, Sy, rim_coords_x, rim_coords_y, radius)
    save_graph(ring_chart_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_RING')
    plt.close(ring_chart_fig)

    release_ring_chart_fig = release_ring_chart(fitted_x[0], fitted_y[0], parameters['entryx'], parameters['entryy'], Sx, Sy, rim_coords_x, rim_coords_y, radius)
    save_graph(release_ring_chart_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_RELEASE_RING')
    plt.close(release_ring_chart_fig)

    # params_x, cov_x = curve_fit(fit_x, t, x)
    # params_y, cov_y = curve_fit(fit_y, t, y)
    # params_z, cov_z = curve_fit(fit_z, t, z)

    # params = np.concatenate((params_x, params_y, params_z))
    # params = {'vx': params[0], 'x0': params[1], 'ax': params[2], 'vy': params[3], 
    #         'y0': params[4], 'ay': params[5], 'vz': params[6], 'z0': params[7], 
    #         'az': params[8], 'frame_start_idx': start}
    
    # fitted_x = fit_x(t, *params_x)
    # fitted_y = fit_y(t, *params_y)
    # fitted_z = fit_z(t, *params_z)

    # fitted_yz_fig = graph_shot_overlap('yz', y, z, fitted_y, fitted_z)
    # save_graph(fitted_yz_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_FITTED_YZ')
    # plt.close(fitted_yz_fig)

    # fitted_xy_fig = graph_shot_overlap('xy', x, y, fitted_x, fitted_y)
    # save_graph(fitted_xy_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_FITTED_XY')
    # plt.close(fitted_xy_fig)


    # rim_coords_x, rim_coords_y, rim_coords_z, radius = find_rim_coords(data_export, file_name)

    # sweet_spot_angle_value = sweet_spot_angle(fitted_x, fitted_y, rim_coords_x, rim_coords_y)
    # Sx, Sy = sweet_spot_coords(fitted_x, fitted_y, rim_coords_x, rim_coords_y, sweet_spot_angle_value)

    # print(sweet_spot_angle_value)
    # print(Sx, Sy)

    # params['sweet_spot'] = [Sx, Sy, rim_coords_z]

    # parameters = pd.Series(params)

    # parameters['apext'] = parameters['vz'] / parameters['az']
    # parameters['apexx'] = parameters['x0'] + parameters['vx'] * parameters['apext']
    # parameters['apexy'] = parameters['y0'] + parameters['vy'] * parameters['apext']
    # parameters['apexz'] = parameters['z0'] + parameters['vz'] * parameters['apext'] - 0.5 * parameters['az'] * parameters['apext']**2
    # parameters['entryt'] = (parameters['vz'] + np.sqrt(parameters['vz']**2 - 2 * parameters['az'] * 10 + 2 * parameters['az'] * parameters['z0'])) / parameters['az']
    # parameters['entryx'] = parameters['x0'] + parameters['vx'] * parameters['entryt']
    # parameters['entryy'] = parameters['y0'] + parameters['vy'] * parameters['entryt']
    # parameters['entryz'] = rim_coords_z

    # ring_chart_fig = ring_chart(fitted_x[0], fitted_y[0], parameters['entryx'], parameters['entryy'], Sx, Sy, rim_coords_x, rim_coords_y, radius)
    # save_graph(ring_chart_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_RING')
    # plt.close(ring_chart_fig)

    # release_ring_chart_fig = release_ring_chart(fitted_x[0], fitted_y[0], parameters['entryx'], parameters['entryy'], Sx, Sy, rim_coords_x, rim_coords_y, radius)
    # save_graph(release_ring_chart_fig, file_name=f'{globals.graph_path}/{player}/{player}_{att}_RELEASE_RING')
    # plt.close(release_ring_chart_fig)