import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import sys

import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import xlsxwriter

from Libs.functions import *

#get dir_path of the file
ROOT = os.path.dirname(os.path.realpath(__file__))
PARAMS_PATH = os.path.join(ROOT, 'Bin', 'parameters.json')
UNITS_PATH = os.path.join(ROOT, 'Bin', 'units.json')



params = load_params(params_path = PARAMS_PATH)
tasks_required_extra_input = ['Pincers stat', 'Movement stat', 'Chasing stat']


with open(UNITS_PATH) as f:
    units = json.load(f)

def create_messagebox(title, message):
    # pop-up a message to notify that the file has been exported
    messagebox = tk.Toplevel(root)
    messagebox.title('Exported')
    # flexibly set the size of messagebox
    messagebox.geometry('+{}+{}'.format(root.winfo_x() + 500, root.winfo_y() + 250))
    messagebox.configure(bg='lightsteelblue2')
    label = tk.Label(messagebox, text=message, font=('Arial', 16, 'bold'))
    label.pack()
    # change the messagebox size to fit the label
    messagebox.update()
    messagebox.after(3000, messagebox.destroy)

def batch_process():
    default_dir = os.path.join(ROOT, 'Input')
    import_file_paths = filedialog.askopenfilenames(initialdir = default_dir, title = "Select files", filetypes = (("all files","*.*"),("excel files","*.xlsx"),("csv files","*.csv")))
    if import_file_paths:
        analyse = Analyzer(import_file_paths)
        try:
            print('Trying to export to Output/Batch_*/summary.xlsx')
            output_dir = analyse.export_excel()
        except:
            print('Error somewhere')

        #create message box to notify user that the excel file is exported
        try:
            create_messagebox('Exported', f'The summary stats are exported to {output_dir}')
        except:
            print('ERROR: Can not create message box')

# when user click Import 
# get excel path from dialog
def getData():
    global CF1
    global CF2
    global OUTPUT_DIR

    default_directory = os.path.join(ROOT, 'Input')
    filetypes = [ ('all files', '*'), ('excel files', '*.xlsx'), ('csv files', '*.csv')]
    import_file_path = filedialog.askopenfilename(initialdir = default_directory, title = "Select file", filetypes = filetypes)

    # Define output directory
    file_name, file_type = os.path.basename(import_file_path).split('.')
    OUTPUT_DIR = os.path.join('Output', file_name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load excel file to dataframe
    if import_file_path.endswith('.xlsx'):
        # check if excel file contains multiple sheets
        if len(pd.ExcelFile(import_file_path).sheet_names) > 1:
            print('Warning: Excel file contains multiple sheets, only the first sheet will be analyzed.')
        # if excel file contains multiple sheets, select the first sheet, no matter its name is
        sheet_0_name = pd.ExcelFile(import_file_path).sheet_names[0]
        df = pd.read_excel(import_file_path, sheet_name=sheet_0_name)
        # df = pd.read_excel(import_file_path)
    elif import_file_path.endswith('.csv'):
        df = pd.read_csv(import_file_path)

    CF1, CF2 = load_df(df)

    # change notify label text to "Excel file imported"
    notify_label.config(text=f'       Data file (.{file_type}) is loaded          ')
    EXCEL_LOADED = True

# when user clicks enter, get the values from the boxes
# and print them out in the console
# and rewrite the parameters in parameters.json file
def get_input():
    global params

    # get values from boxes
    conversion_rate = conversion_rate_widget.get()
    center_x = center_x_widget.get()
    center_y = center_y_widget.get()
    frame_rate = fps_widget.get()
    duration = duration_widget.get()
    ec_threshold = ec_threshold_widget.get()
    speed_threshold_1 = speed_threshold_1_widget.get()
    speed_threshold_2 = speed_threshold_2_widget.get()
    interaction_threshold = interaction_threshold_widget.get()
    fighting_threshold = fighting_threshold_widget.get()
    chasing_threshold = chasing_threshold_widget.get()

    # print values in console
    print('Conversion Rate: ', conversion_rate)
    print('Center X: ', center_x)
    print('Center Y: ', center_y)
    print('Frame Rate: ', frame_rate)
    print('Duration: ', duration)
    print('Extended Claw Threshold: ', ec_threshold)
    print('Speed Threshold 1: ', speed_threshold_1)
    print('Speed Threshold 2: ', speed_threshold_2)
    print('Interaction Threshold: ', interaction_threshold)
    print('Fighting Threshold: ', fighting_threshold)
    print('Chasing Threshold: ', chasing_threshold)

    # rewrite parameters in parameters.json file
    params['CONVERSION_RATE'] = conversion_rate
    params['CENTER_X'] = center_x
    params['CENTER_Y'] = center_y
    params['FPS'] = frame_rate
    params['DURATION'] = duration
    params['EC_THRESHOLD'] = ec_threshold
    params['SPEED_THRESHOLD_1'] = speed_threshold_1
    params['SPEED_THRESHOLD_2'] = speed_threshold_2
    params['INTERACTION_THRESHOLD'] = interaction_threshold
    params['FIGHTING_THRESHOLD'] = fighting_threshold
    params['CHASING_THRESHOLD'] = chasing_threshold

    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=4)

    params = load_params(params_path = PARAMS_PATH)

#####################################

###### BUTTON CONFIGURATION #######
texts = {
    'import' : 'Import Single Excel File',
    'accept' : 'Accept Inputs',
    'export' : 'Export Excel File',
    'display' : 'Display Results',
    'batch': 'Import Several Files for Batch Processing',
    'draw' : 'Draw Graphs',
    'quit' : 'Quit'
}

# get button_width as max(texts, key=len)
button_width = len(max(texts, key=len)) + 2
button_fg = 'white'
button_bg = 'green'
button_font = ('helvetica', 12, 'bold')

#####################################

######## GUI CONFIGURATION ########
# Select the file to be analyzed

SINGLE_FILE = False
INPUTED = False

root = tk.Tk()
root.title('Crayfish Behavior Analysis')

# make a canvas
canvas = tk.Canvas(root, width=450, height=380, bg='lightsteelblue2', relief='raised')
canvas.pack()

# canvas size is fixed at 500x500
canvas.grid_propagate(False)

# create a frame
frame = tk.Frame(canvas, bg='lightsteelblue2')
frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

notify_font = ('Arial', 12, 'bold', 'italic')
notify_label = tk.Label(canvas, text='Please select an excel file for analysis', fg='red', font=notify_font)
notify_label.grid(row=0, column=0, padx=10, pady=10)

# Import button
button_import = tk.Button(canvas, text=texts['import'], command=getData, 
                                bg=button_bg, fg=button_fg, font=button_font)
button_import.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

#############################################################
# MAKE DROP DOWN LIST TO SELECT FROM TASKS (location = canvas) 
# to select from tasks: "Pincers stat", "Movement", "Interaction", "Fighting", "Chasing"

# make a function to create as many line as input

result_font = ('Arial', 12)

def create_text_line(location, position, *args):
    col = 0
    for arg in args:
        using_font = result_font
        # change font to Bold if col = 0
        if col == 0:
            using_font=('Arial', 12, 'bold')
        if col == 2:
            using_font=('Arial', 12, 'italic')
            arg = '( ' + arg + ' )'
        label = tk.Label(location, text=arg, font=using_font)
        label.grid(row=position, column=col, sticky='w', padx=10, pady=10)
        col += 1
    
def export_list_to_excel(file_name, *sheet_dfs, display = True):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    sheet_names = {
        1 : 'Distance in each frame',
        2 : 'Summary data',
        3 : 'Behavior frames and duration'
    }

    sheet_num = 0
    for sheet_df in sheet_dfs:
        sheet_num += 1
        # write each sheet_df to a different sheet if sheet_df is not empty
        if not sheet_df.empty:
            sheet_df.to_excel(writer, sheet_name=sheet_names[sheet_num])

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    if display:
        # pop-up a message to notify that the file has been exported
        messagebox = tk.Toplevel(root)
        messagebox.title('Exported')
        # flexibly set the size of messagebox
        messagebox.geometry('+{}+{}'.format(root.winfo_x() + 500, root.winfo_y() + 250))
        messagebox.configure(bg='lightsteelblue2')
        label = tk.Label(messagebox, text='The file has been exported to ' + file_name, font=('Arial', 16, 'bold'))
        label.pack()
        # change the messagebox size to fit the label
        messagebox.update()
        messagebox.after(2000, messagebox.destroy)

def draw_graphs(name, dataframe1, dataframe2, params, target = 'CF1', width = 10, height = 4):
    print('Drawing graph for ' + name + '...')
    # create a new window to display graph
    graph_window = tk.Toplevel(root)
    graph_window.title(name)
    graph_window.geometry('1000x500')
    graph_window.configure(bg='lightsteelblue2')

    figure = plt.Figure(figsize=(width, height), dpi=100)
    ax = figure.add_subplot(111)
    bar = FigureCanvasTkAgg(figure, graph_window)
    bar.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    data, x_label, y_label, title, _ = draw_graph(name, dataframe1, dataframe2, params, target, width, height)
    data.plot(kind='line', legend=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def result_display(result_dict):
    position = 0
    sheet1_df = pd.DataFrame() # SHEET 1 is for storing the distance/frames
    sheet2_df = pd.DataFrame(columns = ['Type', 'Value', 'Units']) # SHEET 2 is for storing the summary data
    sheet3_df = pd.DataFrame(columns = ['Start-End Frames', 'Duration']) # SHEET 3 is for storing the dictionary data if any
    for key, value in result_dict.items():
        position += 1
        # if value is a list type, add it to sheet1_df_1, column name is key
        if type(value) == list:
            try:
                sheet1_df[key] = value
            except ValueError:
                # when the length of values > length of index, add NaN to the end of existing columns
                if len(value) > len(sheet1_df.index):
                    diff = len(value) - len(sheet1_df.index)
                    for i in range(diff):
                        sheet1_df = sheet1_df.append(pd.Series(dtype=np.float64), ignore_index=True)
                    print('Added {} blank index to sheet1_df to match length of {}'.format(diff, key))
                # when the length of values < length of index, add NaN to the end of new column
                elif len(value) < len(sheet1_df.index):
                    diff = len(sheet1_df.index) - len(value)
                    for i in range(diff):
                        value.append(np.nan)
                    print('Added {} NaN values to value of {} to match length'.format(diff, key))
                try:
                    sheet1_df[key] = value
                except ValueError:
                    print('Error occurs at {}'.format(key))
        # if value is a dictionary type, add it to sheet3_df, column 1 is dictionary keys and column 2 is dictionary values
        elif type(value) == dict:
            for k, v in value.items():
                # sheet3_df = sheet3_df.append({'Start-End Frames': k, 'Duration': v}, ignore_index=True)
                # use pd.concat instead
                sheet3_df = pd.concat([sheet3_df, pd.DataFrame([[k, v]], columns = ['Start-End Frames', 'Duration'])], ignore_index=True)
        # if value is a float type, round it to 4 decimal places, then convert to string
        else:
            if type(value) == float or type(value) == np.float64:
                value = round(value, 4)
            unit = units[key]
            key = upper_first_letter(key)
            # add the result to sheet2_df using pd.concat
            sheet2_df = pd.concat([sheet2_df, pd.DataFrame([[key, value, unit]], columns = ['Type', 'Value', 'Units'])], ignore_index=True)
            value = str(value)
            # create a line to display the result
            create_text_line(stat_result_window, position, key, value, unit)
    # create a button to export the result to excel file
    excel_name = str(task_clicked.get())
    if excel_name in tasks_required_extra_input:
        excel_name += '_' + str(target_clicked.get())
    excel_name = excel_name.replace(' ', '_')
    excel_name = excel_name + '.xlsx'
    excel_path = os.path.join(OUTPUT_DIR, excel_name)
    button_export = tk.Button(stat_result_window, text=texts['export'], command=lambda: export_list_to_excel(excel_path, sheet1_df, sheet2_df, sheet3_df))
    button_export.grid(row=position+1, column=0, padx=10, pady=10)

    # create a button to draw graphs
    if task_clicked.get() != 'Movement stat':
        button_draw = tk.Button(stat_result_window, text=texts['draw'], command=lambda: draw_graphs(task_clicked.get(), CF1, CF2, target = target_clicked.get(), params = params))
        button_draw.grid(row=position+1, column=1, padx=10, pady=10)

def create_dropdown(location, options, clicked, row, col = 0):
    # Create Dropdown menu
    menu = tk.OptionMenu(location, clicked, *options)
    menu.grid(row=row, column=col, padx=10, pady=10)

def confirm_task():
    global target_clicked

    target_clicked = tk.StringVar()
    target_clicked.set('CF1')
    # if user select "Pincers stat"
    if task_clicked.get() in tasks_required_extra_input:
        create_dropdown(canvas, ['CF1', 'CF2'], target_clicked, 4, 0)
        button_enter = tk.Button(canvas, text=texts['display'], command=show)
        button_enter.grid(row=4, column=1, padx=10, pady=10)
    else:
        show()

def show():
    # dropdown_label.config(text = task_clicked.get())

    global stat_result_window

    # Create a window to show the result
    stat_result_window = tk.Toplevel(root)
    stat_result_window.geometry('+500+400')
    stat_result_window.title('Result')
    stat_result_window.geometry('500x500')

    target = target_clicked.get()

    target_1 = CF1 if target == 'CF1' else CF2
    target_2 = CF2 if target == 'CF1' else CF1

    # if user select "Pincers stat"
    if task_clicked.get() == "Pincers stat":
        result_dict = cheliped_stat(target_1, params = params)
        stat_result_window.title('Pincers statistic Result')
        result_display(result_dict)
    if task_clicked.get() == "Movement stat":
        result_dict = movement_stat(target_1, params = params)
        stat_result_window.title('Movement Statistic Result')
        result_display(result_dict)
    if task_clicked.get() == "Interaction stat":
        result_dict = interaction_stat(CF1, CF2, params = params)
        stat_result_window.title('Interaction Statistic Result')
        result_display(result_dict)
    if task_clicked.get() == "Fighting stat":
        result_dict = fighting_stat(CF1, CF2, params = params)
        stat_result_window.title('Fighting Statistic Result')
        result_display(result_dict)
    if task_clicked.get() == "Chasing stat":
        result_dict = chasing_stat(target_1, target_2, params = params)
        stat_result_window.title('Chasing Statistic Result')
        result_display(result_dict)

task_options = [
    "Pincers stat",
    "Movement stat",
    "Interaction stat",
    "Fighting stat",
    "Chasing stat"
]

target_options = [
    "Crayfish 1",
    "Crayfish 2"
]

# datatype of menu text
task_clicked = tk.StringVar()
task_clicked.set("Pincers stat")
# Create Dropdown menu
task_menu = create_dropdown(canvas, task_options, task_clicked, 3, 0)
task_menu = tk.OptionMenu(canvas, task_clicked, *task_options)
task_menu.grid(row=3, column=0, padx=10, pady=10)
  
# Create button, it will change label text
confirm_button = tk.Button(canvas, text = "Confirm", command=confirm_task)
confirm_button.grid(row=3, column=1, padx=2, pady=10)


#############################################################

# Create a new window with boxes to input parameters
window = tk.Toplevel(root)
# change the spawn location of the window
window.geometry('+500+200')
window.title('Input Parameters')
window.geometry('500x500')

# create boxes to input parameters
# PARAMETERS:
    # CONVERSION_RATE = 14.3
    # CENTER_X = 327
    # CENTER_Y = 258
    # FPS = 30
    # DURATION = 600
    # EC_THRESHOLD = 3
    # SPEED_THRESHOLD_1 = 1
    # SPEED_THRESHOLD_2 = 5
    # INTERACTION_THRESHOLD = 3
    # FIGHTING_THRESHOLD = 3
    # CHASING_THRESHOLD = 3

def create_input_box(display_text, location, position, default_value):
    label = tk.Label(location, text=display_text)
    label.grid(row=position, column=0, padx=10, pady=10)
    widget = tk.Entry(location)
    widget.grid(row=position, column=1, padx=10, pady=10)
    widget.insert(0, default_value)
    return widget

conversion_rate_widget = create_input_box('Conversion Rate (cm/pixel):', window, 0, params['CONVERSION_RATE'])
center_x_widget = create_input_box('Center X (pixel):', window, 1, params['CENTER_X'])
center_y_widget = create_input_box('Center Y (pixel):', window, 2, params['CENTER_Y'])
fps_widget = create_input_box('FPS:', window, 3, params['FPS'])
duration_widget = create_input_box('Duration (s):', window, 4, params['DURATION'])
ec_threshold_widget = create_input_box('EC Threshold (pixel):', window, 5, params['EC_THRESHOLD'])
speed_threshold_1_widget = create_input_box('Speed Threshold 1 (cm/s):', window, 6, params['SPEED_THRESHOLD_1'])
speed_threshold_2_widget = create_input_box('Speed Threshold 2 (cm/s):', window, 7, params['SPEED_THRESHOLD_2'])
interaction_threshold_widget = create_input_box('Interaction Threshold (pixel):', window, 8, params['INTERACTION_THRESHOLD'])
fighting_threshold_widget = create_input_box('Fighting Threshold (pixel):', window, 9, params['FIGHTING_THRESHOLD'])
chasing_threshold_widget = create_input_box('Chasing Threshold (pixel):', window, 10, params['CHASING_THRESHOLD'])


# Enter button
button_enter = tk.Button(canvas, text=texts['accept'], command=get_input,
                                bg=button_bg, fg=button_fg, font=button_font)
button_enter.grid(row=2, column=0, padx=10, pady=10)    

# Create a guiding label, wraplength is the width of the label
guide_text = 'If you do batch processing, only Change/Accept Input then click Batch Process'
guide_label = tk.Label(canvas, text=guide_text, wraplength=200, font = ('Arial', 10))
guide_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

# Batch button
button_batch = tk.Button(canvas, text=texts['batch'], command=batch_process,
                                bg=button_bg, fg=button_fg, font=button_font)
# this button on row 6, take up 2 columns 0 and 1
button_batch.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

# Quit button
button_quit = tk.Button(canvas, text=texts['quit'], command=root.destroy, 
                                bg=button_bg, fg=button_fg, font=button_font)
button_quit.grid(row=2, column=1, padx=10, pady=10)


root.mainloop()
