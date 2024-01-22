import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os
from tkinter import Tk
from tkinter import filedialog
import sys

# ROOT is the directory one level above the __file__ directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(ROOT, 'Bin', 'parameters.json')
UNITS_PATH = os.path.join(ROOT, 'Bin', 'units.json')
STATS_PATH = os.path.join(ROOT, 'Bin', 'stats_needed.json')
ALL_STATS_PATH = os.path.join(ROOT, 'Bin', 'all_stats.json')
OUTPUT_PATH = os.path.join(ROOT, 'Output')

def batch_output():
    existed_dir = os.listdir(OUTPUT_PATH)
    batch_num = 1
    while True:
        output_dir = 'Batch_' + str(batch_num)
        print('Trying to name to output dir as {}'.format(output_dir))
        if output_dir not in existed_dir:
            break
        else:
            batch_num += 1
    output_dir = os.path.join(OUTPUT_PATH, output_dir)
    print('Output dir is {}'.format(output_dir))
    os.mkdir(output_dir)
    return output_dir

def select_files(root_path = None):
    if root_path == None:
        root_path = os.getcwd()
    elif not os.path.isdir(root_path):
        print(f"Directory {root_path} does not exist")
        sys.exit()
    root = Tk()
    default_dir = os.path.join(root_path, 'Input')
    # filetypes are excel and csv, and all files
    filetypes = [('Excel', '*.xlsx'), ('CSV', '*.csv'), ('All files', '*')]
    # open file dialog
    filenames = filedialog.askopenfilenames(initialdir = default_dir, title = "Select files", filetypes = filetypes)
    if len(filenames) == 0:
        print('No file selected')
        sys.exit()
    if len(filenames) == 1:
        filenames = filenames[0]
    root.destroy()
    return filenames

def load_stats(stat_path = STATS_PATH):
    with open(stat_path) as f:
        stats = json.load(f)
    return stats

def load_units(unit_path = UNITS_PATH):
    with open(unit_path) as f:
        units = json.load(f)
    return units

def load_params(params_path = PARAMS_PATH):
    with open(params_path) as f:
        params = json.load(f)
    # change value type of params from string to int or float
    for key in params:
        if '.' in params[key]:
            params[key] = float(params[key])
        else:
            params[key] = int(params[key])
    return params

def load_df(input_df, params):
    #if the first row first value is "scorer", then remove it
    if 'scorer' in input_df.columns:
        input_df.columns = input_df.iloc[0]
        input_df = input_df.drop(0)
    #get unique values of row index 0
    row_0 = input_df.iloc[0].unique()
    row_0 = row_0 [1:]
    new_columns = []
    for x in row_0:
        new_columns.append(x+"_X")
        new_columns.append(x+"_Y")
        new_columns.append(x+"_likelihood")

    length = params['DURATION'] * params['FPS'] + 1

    def get_CF(input_df, input_num, col1, coln):
        # Take the data of crawfish1 from df to a new df called CF1
        CF = input_df.iloc[2:,col1:coln]
        # Rename the columns
        CF.columns = new_columns
        # Remove columns with _likelihood
        CF = CF.loc[:,~CF.columns.str.contains('_likelihood')]
        # Take only params['DURATION'] * params['FPS'] + 1 rows
        CF = CF.iloc[:length+1]
        # Check if CF1 have rows with NaN
        if CF.isnull().values.any():
            print(f"CF{input_num} has NaN")
            # fill it with previous value
            CF = CF.fillna(method='ffill')
            print(f'Filled NaN in CF{input_num} with previous value')
        else:
            print(f"CF{input_num} has no NaN")
        # Reset Index
        CF = CF.reset_index(drop=True)

        # Change all values in CF1 and CF2 to float
        CF = CF.astype(float)

        return CF
    
    CF1 = get_CF(input_df, 1, 1, len(row_0)*3+1)
    CF2 = get_CF(input_df, 2, len(row_0)*3+1, len(row_0)*6+1)

    # # Take the data of crawfish1 from df to a new df called CF1
    # CF1 = input_df.iloc[2:,1:len(row_0)*3+1]
    # # Rename the columns
    # CF1.columns = new_columns
    # # Remove columns with _likelihood
    # CF1 = CF1.loc[:,~CF1.columns.str.contains('_likelihood')]
    # # Check if CF1 have rows with NaN
    # if CF1.isnull().values.any():
    #     print("CF1 has NaN")
    #     raise ValueError("CF1 has NaN")
    #     sys.exit()
    # else:
    #     print("CF1 has no NaN")
    # # Reset Index
    # CF1 = CF1.reset_index(drop=True)

    # # Take the data of crawfish2 from df to a new df called CF2
    # CF2 = input_df.iloc[2:,len(row_0)*3+1:]
    # # Rename the columns
    # CF2.columns = new_columns
    # # Remove columns with _likelihood
    # CF2 = CF2.loc[:,~CF2.columns.str.contains('_likelihood')]
    # # Check if CF2 have rows with NaN
    # if CF2.isnull().values.any():
    #     print("CF2 has NaN")
    #     raise ValueError("CF2 has NaN")
    #     sys.exit()
    # else:
    #     print("CF2 has no NaN")
    # # Reset Index
    # CF2 = CF2.reset_index(drop=True)

    # # Change all values in CF1 and CF2 to float
    # CF1 = CF1.astype(float)
    # CF2 = CF2.astype(float)


    return CF1, CF2

def cheliped_stat(input_df, params):
    output_dict = {}

    output_dict['distance'] = [] # distance between chelipeds in centimeter
    output_dict['EC'] = []  # 1 = EC, 0 = no EC

    # iterate through rows of CF1
    for index, row in input_df.iterrows():
        # distance=(SQRT((C4-E4)^2+(D4-F4)^2)/$A$4)
        try:
            distance = math.sqrt((row['LeftPincer_X'] - row['RightPincer_X'])**2 + (row['LeftPincer_Y'] - row['RightPincer_Y'])**2)/params['CONVERSION_RATE']
        except TypeError:
            # if the value is nan, print it out
            print(row['LeftPincer_X'], row['RightPincer_X'], row['LeftPincer_Y'], row['RightPincer_Y'], params['CONVERSION_RATE'] )
            # and their type
            print(type(row['LeftPincer_X']), type(row['RightPincer_X']), type(row['LeftPincer_Y']), type(row['RightPincer_Y']), type(params['CONVERSION_RATE']))
        output_dict['distance'].append(distance)
        if distance > params['EC_THRESHOLD']:
            output_dict['EC'].append(1)
        else:
            output_dict['EC'].append(0)
        # if index == 0 :
        #     output_dict['EC event counts'].append(0)
        # elif output_dict['EC'][index] == output_dict['EC'][index-1] or output_dict['EC'][index] == 0:
        #     output_dict['EC event counts'].append(0)
        # else:
        #     output_dict['EC event counts'].append(1)

    output_dict['EC events'] = {}
    for i in range(len(output_dict['EC'])):
        if i == 0:
            if output_dict['EC'][i] == 1:
                start_point = i
            continue
        if output_dict['EC'][i] == 1 and output_dict['EC'][i-1] == 0:
            start_point = i
        elif output_dict['EC'][i] == 0 and output_dict['EC'][i-1] == 1:
            end_point = i
            output_dict['EC events'][(start_point, end_point-1)] = end_point - start_point     

    output_dict['EC event counts'] = len(output_dict['EC events'])

    output_dict['distance'][:5]

    output_dict['avg distance'] = np.mean(output_dict['distance'])

    output_dict['closest distance'] = np.min(output_dict['distance'])

    output_dict['furthest distance'] = np.max(output_dict['distance'])

    output_dict['EC percentage'] = np.sum(output_dict['EC'])/len(output_dict['EC'])*100

    output_dict['total EC time'] = int(np.sum(output_dict['EC'])/params['FPS'])   # in seconds

    return output_dict

def movement_stat(input_df, params): 
    output_dict = {}

    output_dict['distance'] = []  # distance between 2 frames in centimeter
    output_dict['Speed'] = [] # Speed in cm/s
    output_dict['slow movements'] = [] # Speed < THRESHOLD_1
    output_dict['medium movements'] = [] # THRESHOLD_1 <= Speed < THRESHOLD_2
    output_dict['rapid movements'] = [] # Speed >= THRESHOLD_2
    output_dict['distance to center'] = [] # distance to center in centimeter (dtc)

    # iterate through rows of CF1
    for index, row in input_df.iterrows():
        # distance to center = (SQRT(($A$6-C4)^2+($A$8-D4)^2)/$A$4)
        dtc = math.sqrt((params['CENTER_X'] - row['Rostrum_X'])**2 + (params['CENTER_Y'] - row['Rostrum_Y'])**2)/params['CONVERSION_RATE']
        output_dict['distance to center'].append(dtc)

        # distance=SQRT((C5-C4)^2+(D5-D4)^2)/$A$4
        if index == 0:
            continue
        else:
            distance = math.sqrt((row['Rostrum_X'] - input_df.iloc[index-1]['Rostrum_X'])**2 + (row['Rostrum_Y'] - input_df.iloc[index-1]['Rostrum_Y'])**2)/params['CONVERSION_RATE']
        output_dict['distance'].append(distance)

        # Speed=E4*$A$10
        Speed = distance*params['FPS']
        output_dict['Speed'].append(Speed)

        # check for Movements type:
        if Speed < params['SPEED_THRESHOLD_1']:
            output_dict['slow movements'].append(1)
            output_dict['medium movements'].append(0)
            output_dict['rapid movements'].append(0)
        elif params['SPEED_THRESHOLD_1'] <= Speed < params['SPEED_THRESHOLD_2']:
            output_dict['slow movements'].append(0)
            output_dict['medium movements'].append(1)
            output_dict['rapid movements'].append(0)
        else:
            output_dict['slow movements'].append(0)
            output_dict['medium movements'].append(0)
            output_dict['rapid movements'].append(1)

    output_dict['total distance'] = np.sum(output_dict['distance']) # in cm
    output_dict['avg speed'] = np.mean(output_dict['Speed']) # in cm/s

    # time ratio = # of Movements / count (in percentage)
    output_dict['freeze time percentage'] = np.sum(output_dict['slow movements'])/len(output_dict['slow movements'])*100
    output_dict['swimming time percentage'] = np.sum(output_dict['medium movements'])/len(output_dict['medium movements'])*100
    output_dict['rapid movements time percentage'] = np.sum(output_dict['rapid movements'])/len(output_dict['rapid movements'])*100

    # Average distance to center
    output_dict['avg distance to center'] = np.mean(output_dict['distance to center'])

    return output_dict

def interaction_stat(input_1, input_2, params):
    inter_cf = {}

    # distance=(SQRT((C4-Q4)^2+(D4-R4)^2)/$A$4)
    inter_cf['distance'] = []
    inter_cf['interactions'] = []

    for i in range(len(input_1)):
        distance = math.sqrt((input_1['Rostrum_X'][i] - input_2['Rostrum_X'][i])**2 + (input_1['Rostrum_Y'][i] - input_2['Rostrum_Y'][i])**2)/params['CONVERSION_RATE']
        inter_cf['distance'].append(distance)
        if distance < params['INTERACTION_THRESHOLD']:
            inter_cf['interactions'].append(1)
        else:
            inter_cf['interactions'].append(0)

    inter_cf['avg distance'] = np.mean(inter_cf['distance'])

    inter_cf['closest distance'] = np.min(inter_cf['distance'])

    inter_cf['furthest distance'] = np.max(inter_cf['distance'])

    inter_cf['interactions percentage'] = np.sum(inter_cf['interactions'])/len(inter_cf['interactions'])*100

    inter_cf['interaction events'] = {}
    for i in range(len(inter_cf['interactions'])):
        if i == 0:
            if inter_cf['interactions'][i] == 1:
                start_point = i
            continue
        if inter_cf['interactions'][i] == 1 and inter_cf['interactions'][i-1] == 0:
            start_point = i
        elif inter_cf['interactions'][i] == 0 and inter_cf['interactions'][i-1] == 1:
            end_point = i
            inter_cf['interaction events'][(start_point, end_point-1)] = end_point - start_point

    # get interaction events percentage of longest duration
    try:
        longest_duration = max(inter_cf['interaction events'].values())
    except ValueError:
        longest_duration = 0
    try:
        inter_cf['interaction events percentage'] = longest_duration/len(inter_cf['interactions'])*100
    except ZeroDivisionError:
        inter_cf['interaction events percentage'] = 0

    # get longest_duration in seconds
    inter_cf['longest duration'] = longest_duration/params['FPS']

    return inter_cf

def fighting_stat(input_1, input_2, params):
    output_dict = {}

    output_dict['left side distance'] = []
    output_dict['right side distance'] = []
    output_dict['avg distances'] = []
    output_dict['fighting'] = []
    output_dict['fighting events'] = {}

    for i in range(len(input_1)):
        left_side_distance = math.sqrt((input_1['LeftPincer_X'][i] - input_2['RightPincer_X'][i])**2 + (input_1['LeftPincer_Y'][i] - input_2['RightPincer_Y'][i])**2)/params['CONVERSION_RATE']
        right_side_distance = math.sqrt((input_1['RightPincer_X'][i] - input_2['LeftPincer_X'][i])**2 + (input_1['RightPincer_Y'][i] - input_2['LeftPincer_Y'][i])**2)/params['CONVERSION_RATE']
        avg_distance = (left_side_distance + right_side_distance)/2

        output_dict['left side distance'].append(left_side_distance)
        output_dict['right side distance'].append(right_side_distance)
        output_dict['avg distances'].append(avg_distance)

        if left_side_distance < params['FIGHTING_THRESHOLD'] and right_side_distance < params['FIGHTING_THRESHOLD']:
            output_dict['fighting'].append(1)
        else:
            output_dict['fighting'].append(0)

        # count fighting events
        if i == 0:
            if output_dict['fighting'][i] == 1:
                start_point = i
            continue
        if output_dict['fighting'][i] == 1 and output_dict['fighting'][i-1] == 0:
            start_point = i
        elif output_dict['fighting'][i] == 0 and output_dict['fighting'][i-1] == 1:
            end_point = i
            output_dict['fighting events'][(start_point, end_point-1)] = end_point - start_point

    output_dict['fighting event counts'] = len(output_dict['fighting events'])
    output_dict['fighting time in frames'] = np.sum(output_dict['fighting'])
    output_dict['fighting time in seconds'] = output_dict['fighting time in frames']/params['FPS']
    output_dict['fighting time percentage'] = np.sum(output_dict['fighting'])/len(output_dict['fighting'])*100
    try:
        output_dict['longest fighting time in seconds'] = max(output_dict['fighting events'].values())/params['FPS']
    except ValueError:
        print('No fighting events found')
        output_dict['longest fighting time in seconds'] = 0

    return output_dict

def chasing_stat(chaser, chased, params):
    output_dict = {}

    output_dict['distance'] = []  # distance from chaser Rostrum to chased Telson in cm
    output_dict['chasing'] = []
    output_dict['chasing events'] = {}

    for i in range(len(chaser)):
        # distance = (SQRT((C4-E4)^2+(D4-F4)^2)/$A$4)
        distance = math.sqrt((chaser['Rostrum_X'][i] - chased['Telson_X'][i])**2 + (chaser['Rostrum_Y'][i] - chased['Telson_Y'][i])**2)/params['CONVERSION_RATE']
        output_dict['distance'].append(distance)

        if distance < params['CHASING_THRESHOLD']:
            output_dict['chasing'].append(1)
        else:
            output_dict['chasing'].append(0)

        # count chasing events
        if i == 0:
            if output_dict['chasing'][i] == 1:
                start_point = i
            continue
        if output_dict['chasing'][i] == 1 and output_dict['chasing'][i-1] == 0:
            start_point = i
        elif output_dict['chasing'][i] == 0 and output_dict['chasing'][i-1] == 1:
            end_point = i
            output_dict['chasing events'][(start_point, end_point-1)] = end_point - start_point

    output_dict['avg distance'] = np.mean(output_dict['distance'])
    output_dict['closest distance'] = np.min(output_dict['distance'])
    output_dict['furthest distance'] = np.max(output_dict['distance'])
    output_dict['chasing duration percentage'] = np.sum(output_dict['chasing'])/len(output_dict['chasing'])*100
    try:
        output_dict['longest chasing event'] = max(output_dict['chasing events'].values())/params['FPS']
    except:
        print('No chasing event found')
        output_dict['longest chasing event'] = 0
    return output_dict

def draw_graph(name, dataframe1, dataframe2, params, target = 'CF1', width = 20, height = 8):
    # set style
    plotParams = {'figure.figsize': (width, height),
                    'font.family': 'Times New Roman',
                    'ytick.labelsize': 15,
                    'xtick.labelsize': 20,
                    'axes.labelsize': 20,
                    'axes.titlesize': 20,
                    'legend.fontsize': 15,
                    'lines.linewidth': 2,
                    'lines.markersize': 10,
                    'axes.grid': True,
                    'grid.linestyle': '--',
                    'grid.linewidth': 0.5}
    
    # plt.rcParams.update(plotParams)

    #target is either CF1 or CF2
    assert target == 'CF1' or target == 'CF2', 'target must be either CF1 or CF2'
    if target == 'CF1':
        target_df = dataframe1
    else:
        target_df = dataframe2

    data = None
    x_label = None
    y_label = None
    title = None

    if name == 'Pincers stat':
        target_dict = cheliped_stat(target_df, params)
        # plot distance to frames
        data = pd.DataFrame({'distance': target_dict['distance']}, index = range(len(target_dict['distance'])))
        x_label = 'Frames'
        y_label = 'distance (cm)'
        title = 'distance between left and right pincers of ' + target
        # figure = plt.figure()
        # plt.plot(data)
        # plt.xlabel('Frames')
        # plt.ylabel('distance (cm)')
        # plt.title('distance between left and right pincers of ' + target)
    
    elif name == 'Interaction stat':
        target_dict = interaction_stat(dataframe1, dataframe2, params)
        # plot distance to frames
        data = pd.DataFrame({'distance': target_dict['distance']}, index = range(len(target_dict['distance'])))
        x_label = 'Frames'
        y_label = 'distance (cm)'
        title = 'distance between two crayfishes'
        # plt.plot(data)
        # plt.xlabel(x_label)
        # plt.ylabel(y_label)
        # plt.title(title)

    elif name == 'Fighting stat':
        target_dict = fighting_stat(dataframe1, dataframe2, params)
        # plot average distance to frames
        data = pd.DataFrame({'distance': target_dict['avg distances']}, index = range(len(target_dict['avg distances'])))
        x_label = 'Frames'
        y_label = 'distance (cm)'
        title = 'Average distance between two crayfishes'
        # plt.plot(data)
        # plt.xlabel('Frames')
        # plt.ylabel('distance (cm)')
        # plt.title('Average distance between two crayfishes')

    elif name == 'Chasing stat':
        if target == 'CF1':
            chaser = dataframe1
            chased = dataframe2
        else:
            chaser = dataframe2
            chased = dataframe1
        target_dict = chasing_stat(chaser, chased, params)
        # plot distance to frames
        data = pd.DataFrame({'distance': target_dict['distance']}, index = range(len(target_dict['distance'])))
        chaser_num = target.split('CF')[1]
        chased_num = '2' if chaser_num == '1' else '1'
        x_label = 'Frames'
        y_label = 'distance (cm)'
        title = f"distance between Rostrum of CrayFish {chaser_num} and Telson of Crayfish {chased_num}"
        # plt.plot(data)
        # plt.xlabel('Frames')
        # plt.ylabel('distance (cm)')
        # plt.title(f"distance between Rostrum of CrayFish {chaser_num} and Telson of Crayfish {chased_num}")

    # plt.show()
    return data, x_label, y_label, title, plotParams

# Make a function to uppercase the first letter of each word in a string
def upper_first_letter(string):
    return ' '.join([word[0].upper() + word[1:] for word in string.split()])



class Analyzer():
    def __init__ (self, data_paths):
        # Load params, units
        self.params = load_params()
        with open(UNITS_PATH, 'r') as f:
            self.units = json.load(f)

        # Define file_nums, tasks, targets and excel_name_dict
        self.file_nums = len(data_paths)
        self.tasks = ['pincer', 'movement', 'interaction', 'fighting', 'chasing']
        self.tasks_need_extra_input = ['pincer', 'movement', 'chasing']
        self.tasks_no_need_extra_input = [x for x in self.tasks if x not in self.tasks_need_extra_input]
        self.targets = ['CF1', 'CF2']
        self.stats_needed = load_stats()
        self.excel_name_dict = {}
        for i, data_path in enumerate(data_paths):
            self.excel_name_dict[i] = os.path.basename(data_path)

        # Load excel file in to df and clean it, separate it into two df of two crayfishes
        self.CF1s = [None] * self.file_nums
        self.CF2s = [None] * self.file_nums
        self.pincer_dict_CF1s = [None] * self.file_nums
        self.pincer_dict_CF2s = [None] * self.file_nums
        self.movement_dict_CF1s = [None] * self.file_nums
        self.movement_dict_CF2s = [None] * self.file_nums
        self.interaction_dicts = [None] * self.file_nums
        self.fighting_dicts = [None] * self.file_nums
        self.chasing_dict_CF1s = [None] * self.file_nums
        self.chasing_dict_CF2s = [None] * self.file_nums
        
        for i, data_path in enumerate(data_paths):
            print('Analyzing ' + self.excel_name_dict[i] + '...')
            # if data_path is excel
            if data_path.endswith('.xlsx'):
                # check if excel file contains multiple sheets
                if len(pd.ExcelFile(data_path).sheet_names) > 1:
                    print('Warning: Excel file contains multiple sheets, only the first sheet will be analyzed.')
                # if excel file has multiple sheets, select the first sheet, no matter what the name is
                sheet_0_name = pd.ExcelFile(data_path).sheet_names[0]
                temp_df = pd.read_excel(data_path, sheet_name=sheet_0_name)
                # temp_df = pd.read_excel(data_path)
            elif data_path.endswith('.csv'):
                temp_df = pd.read_csv(data_path)
            self.CF1s[i], self.CF2s[i] = load_df(temp_df, self.params)
            self.pincer_dict_CF1s[i] = cheliped_stat(self.CF1s[i], self.params)
            self.pincer_dict_CF2s[i] = cheliped_stat(self.CF2s[i], self.params)
            self.movement_dict_CF1s[i] = movement_stat(self.CF1s[i], self.params)
            self.movement_dict_CF2s[i] = movement_stat(self.CF2s[i], self.params)
            self.interaction_dicts[i] = interaction_stat(self.CF1s[i], self.CF2s[i], self.params)
            self.fighting_dicts[i] = fighting_stat(self.CF1s[i], self.CF2s[i], self.params)
            self.chasing_dict_CF1s[i] = chasing_stat(self.CF1s[i], self.CF2s[i], self.params)
            self.chasing_dict_CF2s[i] = chasing_stat(self.CF2s[i], self.CF1s[i], self.params)
    
    def retrieve(self, file_num, task, target, stat):
        assert task in self.tasks, f"task must be one of {self.tasks}"
        assert target in self.targets, f"target must be one of {self.targets}"
        assert file_num < self.file_nums, f"file_num must be within 0 and {self.file_nums - 1}"
        
        if task == 'pincer':
            if target == 'CF1':
                return self.pincer_dict_CF1s[file_num][stat]
            else:
                return self.pincer_dict_CF2s[file_num][stat]
        elif task == 'movement':
            if target == 'CF1':
                return self.movement_dict_CF1s[file_num][stat]
            else:
                return self.movement_dict_CF2s[file_num][stat]
        elif task == 'interaction':
            return self.interaction_dicts[file_num][stat]
        elif task == 'fighting':
            return self.fighting_dicts[file_num][stat]
        elif task == 'chasing':
            if target == 'CF1':
                return self.chasing_dict_CF1s[file_num][stat]
            else:
                return self.chasing_dict_CF2s[file_num][stat]

    def group_df(self, task, target, stat):
        # group the data of the same task and target into a dataframe
        # columns = ['File num', 'Stat Value']
        # sheet_name = f"{task} {target} {stat}"

        print(f'Grouping data with task = {task}, target = {target}, stat = {stat}')

        output_df = pd.DataFrame(columns = ['File num', 'Stat Value', 'Units', 'Excel name'])
        for i in range(self.file_nums):
            output_df.loc[i] = [i, self.retrieve(i, task, target, stat), self.units[stat], self.excel_name_dict[i]]
        return output_df

    def export_excel(self):
        ALL_DF = {}
        # create empty dataframe for each task, target and stat
        for task, stat_list in self.stats_needed.items():
            if task not in self.tasks:
                noti = f'{task} is not a valid task' + '\n' + f'VALID TASKS: {self.tasks}'
            if task in self.tasks_need_extra_input:
                for target in self.targets:
                    for stat in stat_list:
                        ALL_DF[(task, target, stat)] = self.group_df(task, target, stat)
            else:
                for stat in stat_list:
                    target = 'CF1'
                    ALL_DF[(task, stat)] = self.group_df(task, target, stat)
        
        # Export the dataframe in ALL_DF to different sheets in excel
        output_dir = batch_output()
        output_path = os.path.join(output_dir, 'summary.xlsx')
        engine = 'xlsxwriter'
        writer = pd.ExcelWriter(output_path, engine = engine)
        print(f'Writer initialized, engine = {engine}')
        for key, summary_df in ALL_DF.items():
            name = '-'.join(key)
            # if name length is > 31, excel will not allow it
            if len(name) > 31:
                # abbrieviate the key[2] to 3 initials
                stat_name = key[-1]
                stat_name = [(i[0].upper()+i[1]) for i in stat_name.split(' ')]
                stat_name = ''.join(stat_name)
                sheet_name = '-'.join(key[:-1]) + '-' + stat_name
            else:
                sheet_name = name
            try:
                print(f'Writing summary dataframe to sheet {sheet_name}')
                summary_df.to_excel(writer, sheet_name = sheet_name, index = False)
            except:
                print(f'Error in exporting sheet {sheet_name}')

        writer.save()

        print("Write to summary.xlsx finished")

        # save the current parameters used (.json)
        with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
            json.dump(self.params, f, indent = 4)
        
        # save that stats needed (.json)
        with open(os.path.join(output_dir, 'stats_needed.json'), 'w') as f:
            json.dump(self.stats_needed, f, indent = 4)

        return output_dir

