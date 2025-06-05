import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def get_files(folder_path):
    for filepath,dirnames,filenames in os.walk(folder_path):
        for filename in filenames:
            print(f'"{os.path.join(filepath,filename)}", ')

            
def convert_path(input_path):
    input_path = input_path.replace(' \ ', '/')
    input_path = os.path.normpath(input_path)
    input_path = input_path.replace(os.sep, '/')
    return input_path


def convert_paths(xls_path_list):
    return [convert_path(input_path) for input_path in xls_path_list]


def xls2csv(path, output_name, row_start = 0, row_stop = 801):
    read_xls = pd.read_excel(path, header=30)
    trim_data = read_xls.iloc[row_start:row_stop, 0:2]
    output_csv = f"{output_name}_csv.csv"
    trim_data.to_csv(output_csv, encoding="utf-8", index=False)
    return output_csv


def batch_read_csv(path_list, row_start = 0, row_stop = 801):
    file_name_list = []

    for path in path_list:
        dir_str, ext = os.path.splitext(path)
        file_name = dir_str.split('/')[-1]
        file_name_list.append(file_name)

    data_name_list = []
    output_name_csv_list = []

    for output_name in file_name_list:
        output_name_csv = f"{output_name}_csv.csv"
        data_name_list.append(f"data_{output_name}")
        path_name_pair = list(zip(path_list, file_name_list))
        output_name_csv_list.append(output_name_csv)
        for path_name in path_name_pair:
            output_name_csv = xls2csv(path_name[0], path_name[1], row_start, row_stop)
            
    print(f"{output_name_csv_list} Csv files have been saved to the root menu.")
    print(f"{data_name_list} Call variables for more details.")
    print(f"Recommend: \ndata_x, y_list, xy_data = toolbox.get_xy(data_dict, x_source='*', y_source_list={data_name_list})")
    
    data_dict = {}

    for data_name, output_name_csv in zip(data_name_list, output_name_csv_list):
        data_dict[data_name] = pd.read_csv(output_name_csv)

    return data_dict


def get_xy(data_dict, x_source, y_source_list):
    data_x = data_dict[x_source].iloc[:, 0]
    xy_dict = {'x':data_x}
    y_list = []
    for y_source in y_source_list:
        output_y = data_dict[y_source].iloc[:, 1]
        xy_dict[f"y_{y_source}"] = output_y
        y_list.append(f"y_{y_source}")

    print(f"Obtained y_list{y_list}")
    print("Recommend: \nlegend_list = [*]\ntoolbox.autoplot(data_x, xy_data, y_list, legend_list = legend_list, png_file_name = '*', x_label = 'Wavelength(nm)', y_label = 'Absorbance', title = '*')")
    
    return data_x, y_list, xy_dict


def autoplot(data_x, xy_data, y_list, hex_c_list = None, legend_list = None, 
            png_file_name = 'Test', scatter_size = 1, 
            x_label = 'Test X Label', y_label = 'Test Y Label', title = 'Test Title', legend_fontsize = 8, dpi = 900):
    
    #plt.figure(figsize=figuresize)
    plt.xlabel(xlabel = x_label)
    plt.ylabel(ylabel = y_label)
    plt.title(label= title)
    

    if legend_list is None:
        if hex_c_list is None:
            plt.scatter(data_x, xy_data[y], label = y, s = scatter_size)

        else:
            for y, hex_c in zip(y_list, hex_c_list):        
                plt.scatter(data_x, xy_data[y], label = y, s = scatter_size, c = hex_c)

        
    else:
        if hex_c_list is None:
            for y_key, legend in zip(y_list, legend_list):
                plt.scatter(data_x, xy_data[y_key], label = legend, s = scatter_size)

        else:
            for y_key, legend, hex_c in zip(y_list, legend_list, hex_c_list):
                plt.scatter(data_x, xy_data[y_key], label = legend, s = scatter_size, c = hex_c)


def autolineplot(data_x, xy_data, y_list, hex_c_list = None, legend_list = None, 
            png_file_name = 'Test', lw = 2, 
            x_label = 'Test X Label', y_label = 'Test Y Label', title = 'Test Title', legend_fontsize = 8, dpi = 900):
    
    #plt.figure(figsize=figuresize)
    plt.xlabel(xlabel = x_label)
    plt.ylabel(ylabel = y_label)
    plt.title(label= title)
    

    if legend_list is None:
        if hex_c_list is None:
            plt.plot(data_x, xy_data[y], label = y, linewidth = lw)

        else:
            for y, hex_c in zip(y_list, hex_c_list):        
                plt.plot(data_x, xy_data[y], label = y, linewidth = lw, c = hex_c)

        
    else:
        if hex_c_list is None:
            for y_key, legend in zip(y_list, legend_list):
                plt.plot(data_x, xy_data[y_key], label = legend, linewidth = lw)

        else:
            for y_key, legend, hex_c in zip(y_list, legend_list, hex_c_list):
                plt.plot(data_x, xy_data[y_key], label = legend, linewidth = lw, c = hex_c)                


    plt.legend(fontsize = legend_fontsize)
    plt.savefig(fname = png_file_name, dpi = dpi)




def get_shifts(data, keepdims=True, column_start=None, column_stop=None):
    # The shape of inputting data is required to be [wavelengths + untreated curve + treated curves, wavelength range]
    read_ndarray = np.genfromtxt(data, delimiter=',')[:, column_start:column_stop]
    # Turn csv into ndarray, which contains all row data(wavelengths and absorbances)
    # A clip in wavelength range is permitted by specifying column_start and column_stop
    x = read_ndarray[0, :]
    y = read_ndarray[1:, :]
    max_value = np.max(y, axis=1, keepdims=True)
    # Get the max value in every row(axis = 1)
    shift_list = []
    wvs_list = []
    coor = np.where(y == max_value)
    # Return the coordinates where max values locate
    # eg.
    # (array([0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6]), --> row numbers(y, not include x), coor[0]
    # array([79, 80, 82, 83,  84, 115, 130, 142, 151, 152, 153, 155, 156, 172, 189])) --> column numbers
    elements, counts = np.unique(coor[0], return_counts=True)
    # eg.
    # (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), --> elements(y, not include x), each item: ele_idx
    # array([5, 1, 1, 1, 5, 1, 1], dtype=int64)) --> counts
    for ele_idx, count in enumerate(counts):
        if count == 1:
            indexes = np.where(y[ele_idx, :] == np.max(y[ele_idx, :]))
            # Find out column numbers in rows who have the unique max value
            for i in indexes:
                wvs = x[i].item()
            wvs_list.append(wvs)

        else:
            indexes = np.where(y[ele_idx, :] == np.max(y[ele_idx, :]))
            # Find out column numbers in rows who have multiple equal max value
            # Return a multi-element array
            for i in indexes:
                wvs = x[i].mean()
            wvs_list.append(wvs)

    for wv in wvs_list:
        shift = wvs_list[0] - wv
        shift_list.append(shift)

    print(wvs_list)
    print(shift_list)
    print(f"x: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"max_value: {max_value}")
    print(f"coordinates: {coor}")
    print(f"elements: {elements}, counts: {counts}")
    return shift_list
