import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
import traceback 
import time


import math
from statistics import mean
from statistics import stdev

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import *
import globals



def main():

    start_time = time.time()

    # ---------------- 1 Player -----------------------

    data_export = globals.MN_data_export
    player_num = 9
    player = globals.MN_player_list[player_num]

    player_files = [x for x in os.listdir(f'{globals.data_path}/{data_export}/{data_export}/{player}') if 'FT' in x]
    print(player_files)

    for f in player_files:

        print(f'------------ {f} -------------')

        att = f.split('_')[2][:-4]

        graph_shot_player_pm(data_export, player, att, f'{globals.data_path}/{data_export}/{data_export}/{player}/{f}')


    # ---------------- All Players ----------------------

    # data_export = globals.MN_data_export
    # player_list = globals.MN_player_list
    
    # for i in range(len(player_list)):

    #     player = player_list[i]

    #     player_files = [x for x in os.listdir(f'{globals.data_path}/{data_export}/{data_export}/{player}') if 'FT' in x]
    #     print(player_files)

    #     for f in player_files:

    #         att = f.split('_')[2][:-4]
            
    #         graph_shot_player(data_export, player, att, f'{globals.data_path}/{data_export}/{data_export}/{player}/{f}')


    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
