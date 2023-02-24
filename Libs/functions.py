import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os

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

    output_dict['Total EC time'] = int(np.sum(output_dict['EC'])/params['FPS'])   # in seconds

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
    longest_duration = max(inter_cf['interaction events'].values())
    inter_cf['interaction events percentage'] = longest_duration/len(inter_cf['interactions'])*100

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
    output_dict['longest fighting time in seconds'] = max(output_dict['fighting events'].values())/params['FPS']

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
    output_dict['longest chasing event'] = max(output_dict['chasing events'].values())/params['FPS']

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
