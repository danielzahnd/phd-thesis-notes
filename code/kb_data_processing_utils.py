# Imports necessary packages to python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from tidegravity import solve_point_corr
import scipy.constants as cs
from nptdms import TdmsFile
from datetime import datetime, timedelta

# Define path
path = r'\\METASFS01.AD.METAS\HOME$\OFFICE\ZADA\PhD Daniel Zahnd\Joule-watt balance project\Python code\Kibble balance test evaluation data'

# Define function to convert strings containing date and time information to a suitable format for tide correction calculations
def convert_to_datetime(date_str, time_str):
    '''
    This function converts date and time information in string format to a suitable format for tide correction calculations using the tidegravity package.

    INPUT
    -----
    date_str (string): Array containing date information
    time_str (string): Array containing time information

    OUTPUT
    ------
    combined_datetime (datetime format): Datetime format object containing date and time for tide correction calculations
    '''
    # Define the format for the input strings
    date_format = '%d.%m.%Y'
    time_format = '%H:%M:%S'
    
    # Parse the date and time strings
    date_part = datetime.strptime(date_str, date_format)
    time_part = datetime.strptime(time_str, time_format).time()
    
    # Combine the date and time parts into a single datetime object
    combined_datetime = datetime.combine(date_part, time_part)
    
    return combined_datetime

# Define function to filter outliers
def filter_outliers(array_y, array_x=np.array([None]), k=3):
    '''
    This function filters outliers out of a set of (x,y(x)) data. If a value y(x) is outside of the k*sigma range around the mean of y-values,
    this value and the corresponding x-value are filtered from the arrays.
    
    INPUT
    -----
    array_x (array): Flattened array of arbitrary length, default is None
    array_y (array): Flattened array of arbitrary length
    k (int): Specifies, which sigma range to use, default is 3

    OUTPUT
    ------
    f_array_x (array): Filtered array_x
    f_array_y (array): Filtered array_y
    '''
    # Case if array_x is None
    if array_x.any() == None:
        # Calculate mean and standard deviation
        mean_arr_y = np.mean(array_y)
        std_arr_y = np.std(array_y)

        # Define empty list with filtered array values
        f_arr_y = []

        # Loop over values in array_y
        for i in range(len(array_y)):
            if abs(array_y[i]-mean_arr_y) < k*std_arr_y:
                f_arr_y.append(array_y[i])
        f_array_y = np.array(f_arr_y)
            
        # Give back the results
        return f_array_y
    
    # Calculate mean value and standard deviation of array_y
    mean_arr_y = np.mean(array_y)
    std_arr_y = np.std(array_y)

    # Define empty list with filtered array values
    f_arr_x = []
    f_arr_y = []

    # Loop over values in array_y
    for i in range(len(array_y)):
        if abs(array_y[i]-mean_arr_y) < k*std_arr_y:
            f_arr_x.append(array_x[i])
            f_arr_y.append(array_y[i])
    f_array_x, f_array_y = np.array(f_arr_x), np.array(f_arr_y)
        
    # Give back the results
    return f_array_x, f_array_y

# Define function to increment start_date and start_time by one second iteratively, such as to find the start_time and start_date of next meas. cycle
def increment_datetime(start_date, start_time):
    '''
    This function iteratively increases start_time and if needed start_date by one second.

    INPUT
    -----
    start_date (str): Start date of measurement campaign, enter in the form year_month_day, e.g. 240724 for July 24, 2024
    start_time (str): Start time of measurement campaign, enter in the form hour_minute_second, e.g. 071434 for 07:14 and 34 seconds

    OUTPUT
    ------
    new_date (str): Start date of measurement campaign + 1 second
    new_time (str): Start time of measurement campaign + 1 second
    '''
    # Combine start_date and start_time into a datetime object
    start_datetime_str = start_date + start_time
    start_datetime = datetime.strptime(start_datetime_str, '%y%m%d%H%M%S')
    
    # Increment the datetime by one second
    incremented_datetime = start_datetime + timedelta(seconds=1)
    
    # Extract the date and time components
    new_date = incremented_datetime.strftime('%y%m%d')
    new_time = incremented_datetime.strftime('%H%M%S')
    
    return new_date, new_time

# Define function to get data for one measurement cycle (3 weightings + 10 up-down movements for induction)
def get_data(start_date, start_time, num_series=1):
    '''
    This function gets the data for one measurement cycle (3 weightings + 10 up-down movements for induction). At the end, the start 
    time and start date of the next measurement cycle after termination of the loaded cycle is looked up and given back as strings.
    
    INPUT
    -----
    num_series (int): Number of measurement cycles to load, default is 1
    start_date (str): Start date of measurement campaign, enter in the form year_month_day, e.g. 240724 for July 24, 2024
    start_time (str): Start time of measurement campaign, enter in the form hour_minute_second, e.g. 071434 for 07:14 and 34 seconds

    OUTPUT
    ------
    data_i (dict): Dictionary containing the loaded weighting data from one measurement cycle as numpy arrays
    data_u (dict): Dictionary containing the loaded induction data from one measurement cycle as numpy arrays
    n_start_date (str): String of new start date after termination of loaded cycle
    n_start_time (str): String of new start time after termination of loaded cycle
    '''
    # Set paths for voltage and current measurement files
    i_path = r'\\metasfs01.ad.metas\5130-M\5133_MFP\20_QS\20_Messplaetze\131_80_BWM\911.00\Data\Comparator' + '\\'
    u_path = r'\\metasfs01.ad.metas\5130-M\5133_MFP\20_QS\20_Messplaetze\131_80_BWM\911.00\Data\Uind' + '\\'

    # Define dictionaries for both u and i files
    data_i = {}    
    data_u = {}

    # Loop over number of cylces
    for i in range(num_series):
        # Define temporary data dictionaries
        temp_data_i = {}
        temp_data_u = {}
        
        # Define lists with filenames and filepaths
        i_filenames = []
        i_filepaths = []
        u_filenames = []
        u_filepaths = []

        # Get filenames and filepaths for weighting measurements
        i_files_n = 0
        while i_files_n < 3:
            for el in os.listdir(i_path + start_date):
                if el == start_date + '_' + start_time + '.tdms':
                    i_filenames.append(el)
                    i_filepaths.append(i_path + '\\' + start_date + '\\' + start_date + '_' + start_time + '.tdms')
                    i_files_n = i_files_n + 1
            start_date, start_time = increment_datetime(start_date=start_date, start_time=start_time)

        # Get filenames and filepaths for induction measurements
        u_files_n = 0
        while u_files_n < 20:
            for el in os.listdir(u_path + start_date):
                if el == start_date + '_' + start_time + '.tdms':
                    u_filenames.append(el)
                    u_filepaths.append(u_path + '\\' + start_date + '\\' + start_date + '_' + start_time + '.tdms')
                    u_files_n = u_files_n + 1
            start_date, start_time = increment_datetime(start_date=start_date, start_time=start_time)

        # Iterate over each weighting file and load it
        for k in range(len(i_filenames)):
            temp_data_i[i_filenames[k]] = load_tmds(i_filepaths[k], channel_num=0)

        # Iterate over each induction file and load it
        for k in range(len(u_filenames)):
            temp_data_u[u_filenames[k]] = load_tmds(u_filepaths[k], channel_num=1)

        # Append data to dictionary
        data_i['Cycle_' + str(i + 1)] = temp_data_i
        data_u['Cycle_' + str(i + 1)] = temp_data_u

        # Increment the start_date and start_time in order to find next cycle
        start_date, start_time = increment_datetime(start_date=start_date, start_time=start_time)

    # Return files, a dictionary for weighting measurements and a second dictionary for induction measurements
    return data_i, data_u

# Define function to load tmds files
def load_tmds(filepath, channel_num):
    '''
    This function loads a tdms file.

    INPUT
    -----
    filepath (str): Complete path to the file
    channel_num (int): Channel, where the important data is stored

    OUTPUT
    ------
    channel_data (dict): Dictionary containing the loaded data as numpy arrays
    '''
    # Load the TDMS file
    tdms_file = TdmsFile.read(filepath)

    # Access groups and channels in the TDMS file
    for group in tdms_file.groups():
        for channel in group.channels():
            data = channel[:]

    # Initialize a dictionary to store data from each channel
    channel_data = {}

    # Iterate over channels
    for channel in tdms_file.groups()[channel_num].channels():
        channel_name = channel.name
        # Extract data from the channel
        data = np.array(channel[:])
        # Store data in the dictionary with the channel name as the key
        channel_data[channel_name] = data

    # Return dictionary containing data
    return channel_data

# Define function to load all tmds files in a folder
def load_data(folderpath, channel_num):
    '''
    This function loads multiple tdms files in a folder and stores them in a dictionary.

    INPUT
    -----
    folderpath (str): Path to the folder containing the data required
    channel_num (int): Channel, where the important data is stored in the tmds file
    
    OUTPUT
    ------
    data (dict): Dictionary containing the data of all files in folderpath
    '''
    # List all files in the folder
    files = os.listdir(folderpath)

    # Define dictionary
    data = {}

    # Iterate over each file and load it
    for file_name in files:
        data[file_name] = load_tmds(folderpath + '\\' + file_name, channel_num=channel_num)

    # Return data
    return data

# Define function to calculate velocities and voltage to velocity ratios
def calc_u_v(all_files):
    '''
    This function calculates the voltage to velocity ratio.

    INPUT
    -----
    all_files (dict): Dictionnary containing all voltage measurement files of one measurement cycle

    OUTPUT
    ------
    all_u_v (array): Voltage to velocity ratios
    all_pos_u_v (array): Associated positions of voltage to velocity ratios
    '''
    # Wavelength of the light used in the interferometer [m]
    λ = 532.2455762E-9

    # Define voltage correction factor arising from calibration of voltmeter against Josephson voltage standard
    a = 1.84e-6
    
    # Define lists with all voltage to velocity ratios and the associated positions for a bunch of 3 weightings + 10 up-downs
    all_u_v = []
    all_pos_u_v = []

    for which_file in range(len(list(all_files.keys()))):
        # Take the file of index whichfile
        file = all_files[list(all_files.keys())[which_file]]

        # Get time of zero position in z
        t_0 = file['TIA3_start'][2]
        t_fringe_up = file['TIA1_start']
        z_fringe_up = np.zeros(len(t_fringe_up))
        t_fringe_down = file['TIA1_stop']
        z_fringe_down = np.zeros(len(t_fringe_down))
        t_u_start = file['TIA2_start']
        z_u_start = np.zeros(len(t_u_start))
        t_u_stop = file['TIA2_stop']
        z_u_stop = np.zeros(len(t_u_stop))
        u = file['DVM_tot']*(1 - a)
        v = np.zeros(len(u))
        p_v = np.zeros(len(v))
        u_v = np.zeros(len(v))

        # Calculate position of each up- or downward fringe
        diff_up = np.inf
        index_up = 0
        for i in range(len(t_fringe_up)): # Get index of upward zero-crossing of interferom. signal closest to time at zero-position
            d = abs(t_fringe_up[i]-t_0)
            if d < diff_up:
                diff_up = d
                index_up = i

        diff_down = np.inf
        index_down = 0
        for i in range(len(t_fringe_down)): # Get index of downward zero-crossing of interferom. signal closest to time at zero-position
            d = abs(t_fringe_down[i]-t_0)
            if d < diff_down:
                diff_down = d
                index_down = i

        # Calculate time differences needed for further calculations
        delta_t_up_down = abs(t_fringe_up[index_up]-t_fringe_down[index_down]) # Calculate difference in time between closest zero-crossings to zero-position
        delta_t_up = abs(t_fringe_up[index_up]-t_0) # Calculate difference in time between closest upward zero-crossing to time at zero-position
        delta_t_down = abs(t_fringe_down[index_down]-t_0) # Calculate difference in time between closest downward zero-crossing to time at zero-position

        # Calculate positions associated to timestamps closest to zero-position
        if t_fringe_up[index_up] < t_0: 
            z_fringe_up[index_up] = 0 - (delta_t_up/delta_t_up_down)*(λ/4) # Calculate position of closest upward zero-crossing
        else: 
            z_fringe_up[index_up] = 0 + (delta_t_up/delta_t_up_down)*(λ/4) # Calculate position of closest upward zero-crossing

        if t_fringe_down[index_down] < t_0:
            z_fringe_down[index_down] = 0 - (delta_t_down/delta_t_up_down)*(λ/4) # Calculate position of closest downward zero-crossing
        else:
            z_fringe_down[index_down] = 0 + (delta_t_down/delta_t_up_down)*(λ/4) # Calculate position of closest downward zero-crossing

        # Calculate positions associated to timestamps of upward zero-crossings for timestamps < zero-position time
        begin = 1
        for i in range(index_up, 0, -1):
            z_fringe_up[i-1] = z_fringe_up[index_up] - begin*(λ/2)
            begin = begin + 1

        # Calculate positions associated to timestamps of upward zero-crossings for timestamps > zero-position time
        begin = 1
        for i in range(index_up, len(z_fringe_up)-1):
            z_fringe_up[i+1] = z_fringe_up[index_up] + begin*(λ/2)
            begin = begin + 1

        # Calculate positions associated to timestamps of downward zero-crossings for timestamps < zero-position time
        begin = 1
        for i in range(index_down, 0, -1):
            z_fringe_down[i-1] = z_fringe_down[index_down] - begin*(λ/2)
            begin = begin + 1
        
        # Calculate positions associated to timestamps of downward zero-crossings for timestamps > zero-position time
        begin = 1
        for i in range(index_down, len(z_fringe_down)-1):
            z_fringe_down[i+1] = z_fringe_down[index_up] + begin*(λ/2)
            begin = begin + 1

        # Calculate starting positions for induction measurements
        for k in range(len(t_u_start)):
            t_0 = t_u_start[k] # Set starting time of i'th voltage induction measurement as zero-time
            diff_up = np.inf
            index_up = 0
            for i in range(len(t_fringe_up)): # Get index of upward zero-crossing of interferom. signal closest to time at beginning of induction measurement
                d = abs(t_fringe_up[i]-t_0)
                if d < diff_up:
                    diff_up = d
                    index_up = i

            diff_down = np.inf
            index_down = 0
            for i in range(len(t_fringe_down)): # Get index of downward zero-crossing of interferom. signal closest to time at beginning of induction measurement
                d = abs(t_fringe_down[i]-t_0)
                if d < diff_down:
                    diff_down = d
                    index_down = i
            
            # Calculate time differences needed for further calculations
            delta_t_up_down = abs(t_fringe_up[index_up]-t_fringe_down[index_down]) # Calculate difference in time between closest zero-crossings to induction meas. start
            delta_t_up = abs(t_fringe_up[index_up]-t_0) # Calculate difference in time between closest upward zero-crossing and induction meas. start
            delta_t_down = abs(t_fringe_down[index_down]-t_0) # Calculate difference in time between closest downward zero-crossing and induction meas. start
            
            # Calculate position of voltage induction measurement start
            if t_fringe_up[index_up] > t_fringe_down[index_down]:
                z_u_start[k] = z_fringe_down[index_down] + (delta_t_down/delta_t_up_down)*(λ/4) # Add fractional distance of λ/4 to zero-crossing before meas. starts

            if t_fringe_down[index_down] > t_fringe_up[index_up]:
                z_u_start[k] = z_fringe_up[index_up] + (delta_t_up/delta_t_up_down)*(λ/4) # Add fractional distance of λ/4 to zero-crossing before meas. starts

        # Calculate ending positions for induction measurements
        for k in range(len(t_u_stop)):
            t_0 = t_u_stop[k] # Set ending time of i'th voltage induction measurement as zero-time
            diff_up = np.inf
            index_up = 0
            for i in range(len(t_fringe_up)): # Get index of upward zero-crossing of interferom. signal closest to time at end of induction measurement
                d = abs(t_fringe_up[i]-t_0)
                if d < diff_up:
                    diff_up = d
                    index_up = i

            diff_down = np.inf
            index_down = 0
            for i in range(len(t_fringe_down)): # Get index of downward zero-crossing of interferom. signal closest to time at end of induction measurement
                d = abs(t_fringe_down[i]-t_0)
                if d < diff_down:
                    diff_down = d
                    index_down = i

            # Calculate time differences needed for further calculations
            delta_t_up_down = abs(t_fringe_up[index_up]-t_fringe_down[index_down]) # Calculate difference in time between closest zero-crossings to induction meas. end
            delta_t_up = abs(t_fringe_up[index_up]-t_0) # Calculate difference in time between closest upward zero-crossing and induction meas. end
            delta_t_down = abs(t_fringe_down[index_down]-t_0) # Calculate difference in time between closest downward zero-crossing and induction meas. end

            # Calculate position of voltage induction measurement end
            if t_fringe_up[index_up] > t_fringe_down[index_down]:
                z_u_stop[k] = z_fringe_down[index_down] + (delta_t_down/delta_t_up_down)*(λ/4) # Add fractional distance of λ/4 to zero-crossing before meas. ends

            if t_fringe_down[index_down] > t_fringe_up[index_up]:
                z_u_stop[k] = z_fringe_up[index_up] + (delta_t_up/delta_t_up_down)*(λ/4) # Add fractional distance of λ/4 to zero-crossing before meas. ends
        
        # Multiply calculated vertical positions times (-1), if direction of movement is reversed
        if which_file % 2 == 0:
            z_fringe_up = z_fringe_up*(-1)
            z_fringe_down = z_fringe_down*(-1)
            z_u_start = z_u_start*(-1)
            z_u_stop = z_u_stop*(-1)

        # Calculate velocities and voltage to velocity ratios with associated positions
        for i in range(len(v)):
            v[i] = (z_u_stop[i]-z_u_start[i])/(t_u_stop[i]-t_u_start[i]) # Calculate average velocity in the i'th voltage induction measurement window
            p_v[i] = z_u_start[i] + ((z_u_stop[i]-z_u_start[i])/2) # Define position of the i'th voltage induction measurement
            u_v[i] = u[i]/v[i] # Calculate voltage to velocity ratio

        # Append calculated values to list containing all values per bunch of 3 weightings + 10 up-downs
        all_u_v.append(u_v)
        all_pos_u_v.append(p_v)

    # Return results
    return all_u_v, all_pos_u_v

# Define function to calculate average Ge for 6h of measurement data
def calculate_Ge(u_files, fit_lolim = -7.0, fit_uplim = 3.0, plot_fits=False, plot_example_fit=False, fit_degree=4, k=3, filter_outl=False):
    '''
    This function calculates the mean value of the voltage to velocity ratio Ge from num_series measurement sets. One measurement set consists of three
    weighting measurements (current measurement), aswell as of 10 upward and 10 downward motions (voltage and position measurement).

    INPUT
    -----
    u_files (dict): Dictionary with induction files
    fit_lolim (float): Lower limit in mm for polynomial fit, default is -7.0 mm
    fit_uplim (float): Upper limit in mm for polynomial fit, default is 3.0 mm
    plot_fits (bool): If true, all fits are plotted, default is False
    plot_example_fit (bool): If true, one fit is plotted as an example, default is False
    fit_degree (int): Order of polynomial fit for measurement data, default is 4
    num_series (int): Number of measurement series stored in the specified dynamic measurement folder path
    k (int): Number of sigma ranges for outlier detection, default is 3
    filter_outl (bool): If true, outliers in the k-sigmal range are filtered from the voltage to velocity ratios, default is False

    OUTPUT
    ------
    mean_Ge (float): Mean calculated value of the voltage to velocity ratio Ge at the weighting position calculated from one measurement set
    std_Ge (float): Standard deviation of Ge calculated from one measurement set
    Ge (array): One Ge value per series
    '''
    # Define list to store calculated Ge values
    Ge = []

    # Initialize cycle counter
    cycle_no = 1

    # Loop over loaded files
    for file in u_files:
        # Load data
        dynamic_files = u_files[file]

        # Calculate voltage to velocity ratios for one series
        u_v, u_v_pos = calc_u_v(dynamic_files)
        u_v, u_v_pos = np.array(u_v).flatten(), np.array(u_v_pos).flatten()*1e3

        # Filter outliers, if filter_outliers == True
        if filter_outl == True:
            u_v_pos_f, u_v_f = filter_outliers(array_x=u_v_pos, array_y=u_v, k=k)
        else:
            u_v_pos_f, u_v_f = u_v_pos, u_v

        # Cut data to the desired regime for fitting
        indices = np.where((u_v_pos_f >= fit_lolim) & (u_v_pos_f <= fit_uplim))
        filtered_u_v_pos_f = u_v_pos_f[indices]
        filtered_u_v_f = u_v_f[indices]

        # Perform fit
        weighting_pos = -3.16715 # Vertical position of weighting [mm]
        coefficients = np.polyfit(filtered_u_v_pos_f, filtered_u_v_f, fit_degree)
        poly_function = np.poly1d(coefficients)
        u_v_pos_f_fit = np.linspace(min(filtered_u_v_pos_f), max(filtered_u_v_pos_f), 1000)
        u_v_f_fit = poly_function(u_v_pos_f_fit)
        u_v_w = poly_function(weighting_pos)

        # Plot fits
        if plot_fits == True:
            plt.title('Voltage to velocity ratio $G_e = U_d/v$')
            plt.scatter(u_v_pos_f, u_v_f, label='Data', s=6)
            plt.plot(u_v_pos_f_fit, u_v_f_fit, color='red', label='Fit $\\mathcal{O}' + f'(z^{fit_degree})$')
            plt.axvline(x=weighting_pos, label='Weighting pos.', color='green')
            plt.xlabel('Position $z$ [mm]')
            plt.ylabel('$G_e$ [Vs$\\text{m}^{-1}$]')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f'Ge_series_{cycle_no}.pdf')
            plt.show()

        # Plot example fit
        if plot_example_fit == True and cycle_no == 1:
            plt.title('Voltage to velocity ratio $G_e = U_d/v$')
            plt.scatter(u_v_pos_f, u_v_f, label='Data', s=6)
            plt.plot(u_v_pos_f_fit, u_v_f_fit, color='red', label='Fit $\\mathcal{O}' + f'(z^{fit_degree})$')
            plt.axvline(x=weighting_pos, label='Weighting pos.', color='green')
            plt.xlabel('Position $z$ [mm]')
            plt.ylabel('$G_e$ [Vs$\\text{m}^{-1}$]')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f'Ge_example.pdf')
            plt.show()

        # Append calculated Ge value
        Ge.append(u_v_w)

        # Print progress
        print(f'Series {cycle_no}/{len(u_files)} done!')

        # Increment cycle number
        cycle_no = cycle_no + 1

    # Calculate average Ge value and associated standard deviation
    mean_Ge = np.mean(np.array(Ge))
    std_Ge = np.std(np.array(Ge))

    return mean_Ge, std_Ge, Ge

# Define function to calculate current to g ratios
def calc_g_i(all_files, tide_corr=True):
    '''
    This function calculates the gravity to current ratio.

    INPUT
    -----
    all_files (dict): Dictionnary containing all current measurement files of one measurement cycle
    tide_corr (boolean): If False, tide corrections for the gravitational acceleration are not considered, default is True

    OUTPUT
    ------
    g_i (float): Gravity to current ratio for measurement files stored in a folder
    std_g_I (folat): Error associated to g_i calculated via Gaussian error propagation
    '''
    # Define constants and arrays
    a = 1.84e-6 # Correction factor for voltage readings
    m = 1.00000000 # Mass [kg]
    R = 99.9999772 # Resistance [Ω]
    g = 9.80588659 # Gravitational acceleration at weighting position [m/s^2]
    lat = 46.92382 # Latitude of Kibble balance [°]
    lon = 7.46447 # Longitude of Kibble balance (West is negative) [°]
    alt = 551.3 # Altitude of watt balance [m]
    
    # Define time interval of weighting
    time = np.array([all_files[list(all_files.keys())[0]]['Time'][0], all_files[list(all_files.keys())[2]]['Time'][-1]]) # First entry: Start time of first weighting value, second entry: End time of last weighting value
    date = np.array([all_files[list(all_files.keys())[0]]['Date'][0], all_files[list(all_files.keys())[2]]['Date'][-1]]) # First entry: Start time of first weighting value, second entry: End time of last weighting value
    datetime_start = convert_to_datetime(date[0], time[0])
    datetime_end = convert_to_datetime(date[1], time[1])

    # Get the difference between datetime_start and datetime_end in seconds
    time_difference = datetime_end - datetime_start
    difference_in_seconds = time_difference.total_seconds()

    # Calculate mean gravity correction for one weighting (n-p-n or p-n-p)
    tide_corrections = solve_point_corr(lat, lon, alt, datetime_start, n=int(difference_in_seconds), increment='S')
    g_corrections = tide_corrections['g0'].values
    g_corr = np.mean(g_corrections)*1e-5

    # Define list with voltage measurements over the resistor R
    u_s = []

    # Define list with load cell readings [g]
    delta_m = [] 

    # Loop over available files
    for which_file in range(len(list(all_files.keys()))):
        # Take the file of index whichfile
        file = all_files[list(all_files.keys())[which_file]]

        # Append voltage data
        u_s.append(file['DVM_tot'])

        # Append load cell data
        delta_m.append(file['LoadCell'])

    # Convert lists to numpy arrays
    u_s = np.array(u_s)*(1 - a) # (1-a) is a calibration factor that arises from calibration against Josephson voltage standard
    delta_m = np.array(delta_m)

    # Define index to begin with
    beg_idx = int(70)

    # Calculate currents
    I = u_s/R
    I_sm1, std_I_sm1 = np.mean(I[0,beg_idx:]), np.std(I[0,beg_idx:])
    I_sm2, std_I_sm2 = np.mean(I[1,beg_idx:]), np.std(I[1,beg_idx:])
    I_sm3, std_I_sm3 = np.mean(I[2,beg_idx:]), np.std(I[2,beg_idx:])
    if I_sm1 > 0:
        I_sm = ((I_sm1 + I_sm3)/2) - I_sm2
    if I_sm1 < 0:
        I_sm = I_sm2 - ((I_sm1 + I_sm3)/2)
    sigma_I_sm = np.sqrt(0.25*std_I_sm1**2 + 0.25*std_I_sm3**2 + std_I_sm2**2)

    # Calculate masses
    delta_m_1, std_delta_m_1 = np.mean(delta_m[0,beg_idx:]), np.std(delta_m[0,beg_idx:])
    delta_m_2, std_delta_m_2 = np.mean(delta_m[1,beg_idx:]), np.std(delta_m[1,beg_idx:])
    delta_m_3, std_delta_m_3 = np.mean(delta_m[2,beg_idx:]), np.std(delta_m[2,beg_idx:])
    delta_m = abs(delta_m_2 - ((delta_m_1 + delta_m_3)/2))*1e-3
    sigma_delta_m = np.sqrt(0.25*std_delta_m_1**2 + 0.25*std_delta_m_3**2 + std_delta_m_2**2)*1e-3

    # Calculate gravity to current ratio and associated error
    if tide_corr == True:
        g_i = (g - g_corr)*(m - delta_m)/I_sm
    else:
        g_i = g*(m - delta_m)/I_sm
    std_g_i = np.sqrt((g/I_sm)**2*sigma_delta_m**2 + ((m-delta_m)*(g/I_sm))**2*sigma_I_sm)
    return g_i, std_g_i

# Define function to calculate average Ge for 6h of measurement data
def calculate_Gm(i_files, num_series=10, tide_corr=True):
    '''
    This function calculates the mean value of the gravity to current ratio Gm from num_series measurement sets. One measurement set consists of three
    weighting measurements (current measurement), aswell as of 10 upward and 10 downward motions (voltage and position measurement).

    INPUT
    -----
    i_files (dict): Dictionary with weighting files
    num_series (int): Number of measurement series stored in the specified dynamic measurement folder path
    tide_corr (boolean): If False, tide corrections for the gravitational acceleration are not considered, default is True

    OUTPUT
    ------
    mean_Gm (float): Mean calculated value of the gravity to current ratio Gm at the weighting position calculated from one measurement set
    sigma_Gm (float): Error of Gm calculated from one measurement set (Gaussian error propagation was applied)
    Gm (array): Gives back all Gm values for one measurement set (10 values)
    '''
    # Define list to store calculated Gm values
    Gm = []
    Gm_sigma = []

    # Initialize cycle counter
    cycle_no = 1

    # Loop over loaded files
    for file in i_files:
        # Load data
        static_files = i_files[file]

        # Calculate gravity to current ratio for one series
        g_i, sigma_g_i = calc_g_i(static_files, tide_corr=tide_corr)

        # Append calculated Ge value
        Gm.append(g_i)
        Gm_sigma.append(sigma_g_i)

        # Print progress
        print(f'Series {cycle_no}/{len(i_files)} done!')

        # Increment cycle number
        cycle_no = cycle_no + 1

    # Calculate average Ge value and associated standard deviation
    mean_Gm = np.mean(np.array(Gm))
    sigma_Gm = np.std(Gm) # Alternative approach: sigma_Gm = np.sqrt(np.sum(np.array(Gm_sigma)**2))

    return mean_Gm, sigma_Gm, np.array(Gm)