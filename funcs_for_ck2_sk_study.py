#%%
#use neo to import either voltage or current clamp data in the correct, scaled units!
def load_neo_file(file_name, **kwargs):
    #################
    # This function is used to load the data from the axograph files
    #################
    import neo
    reader = neo.io.get_io(file_name)
    blocks = reader.read(**kwargs)
    new_blocks = []
    for bl in blocks:
        new_segments = []
        for seg in bl.segments:
            traces = []
            count_traces = 0
            analogsignals = seg.analogsignals

            for sig in analogsignals:
                traces.append({})
                traces[count_traces]['T'] = sig.times.rescale('ms').magnitude
                #need to write an if statement here for conversion
                try:
                    traces[count_traces]['A'] = sig.rescale('pA').magnitude
                except:
                    traces[count_traces]['V'] = sig.rescale('mV').magnitude
                count_traces += 1
            new_segments.append(traces)
        #new_blocks.append(efel_segments)
    return new_segments

#for finding spike times in the voltage trace, specifically when looking at spontaneous firing. This code is utiliezed into numerous other functions
def find_spike_times(voltages, dt, detection_level, min_interval):
    ###############
    # This function is used to find the spike times in the voltage trace
    #It requires the voltage trace, the time step, the detection level and the minimum interval between spikes
    ###############
    spike_times = []
    min_amplitude = -10
    last_spike_time = -min_interval
    for i in range(1, len(voltages)):
        t = i * dt
        if (voltages[i - 1] < detection_level <= voltages[i]) and (voltages[i] - voltages[i - 1] >= min_amplitude) and (t - last_spike_time >= min_interval):
            spike_times.append(t)
            last_spike_time = t
    return spike_times

#general helper function to segment the voltage trace
def segment(values, dx, x_min, x_max):
    ###############
    # This function is used to segment the voltage trace
    #It requires the voltage trace, the time step, the minimum and maximum time
    ###############
    return values[round(x_min / dx):round(x_max / dx)]

#general helper function to find the slopes of the voltage trace
def find_slopes(values, dx):
    ###############
    # This function is used to find the slopes of the voltage trace
    #It requires the voltage trace and the time step
    ###############
    diffs = np.diff(values)
    slopes = [0] * len(values)
    slopes[0] = diffs[0] / dx
    slopes[-1] = diffs[-1] / dx
    for i in range(1, len(values) - 1):
        slopes[i] = (diffs[i - 1] + diffs[i]) / (2 * dx)
    return (slopes)

#dvdt function to calculate the dvdt of the voltage trace, this is useful for finding the threshold of the spike
#for threshold detection we used a threshold of 20 mV/ms
def dvdt(path, sweep):
    table2 = []
    os.chdir(path)
    for filename in os.listdir():
        #check whether file is in the axgx or axgd format
        if filename.endswith(".axgd") or filename.endswith(".axgx"):
            [traces] = efel.io.load_neo_file(filename, stim_start=0, stim_end=10000)
            for data in traces[sweep]:
                times = (data['T']) / 1000
                voltages = (data['V'])
                times -= times[0]
                dt = times[2] - times[1]
                detection_level = -10
                min_interval = 0.0001
                spike_times = find_spike_times(voltages, dt, detection_level, min_interval)
                isis = np.diff(spike_times)
                time_before = .018
                time_after = .015
                times_rel = list(np.arange(-time_before, time_after, dt))
                spike_voltages = []
                for i in range(0, len(spike_times)):
                    if time_before < spike_times[i] < times[-1] - time_after:
                        spike_voltages.append(
                            segment(voltages, dt, spike_times[i] - time_before, spike_times[i] + time_after))
                spike_voltage_arrays = [np.array(x) for x in spike_voltages]
                mean_spike_voltages = [np.mean(k) for k in zip(*spike_voltage_arrays)]
                dvdt_threshold = 10
                dvdt = find_slopes(mean_spike_voltages, dt)
                i = 1
                while dvdt[i] < dvdt_threshold:
                    i += 1
                v0 = mean_spike_voltages[i - 1]
                v1 = mean_spike_voltages[i]
                dvdt0 = dvdt[i - 1]
                dvdt1 = dvdt[i]
                v_threshold = v0 + (v1 - v0) * (dvdt_threshold - dvdt0) / (dvdt1 - dvdt0)
                pandas_dvdt = pd.DataFrame(dvdt)
                pandas_dvdt.rename(columns={0: filename}, inplace=True)  #naming the columns!
                pandas_membrane_voltages = pd.DataFrame(mean_spike_voltages)
            table2.append(pandas_dvdt)
            df_concat = pd.concat(table2, axis=1)

            # df_concat.to_excel('dvdt' + 'master_file.xlsx', index=False)
    return (df_concat)


#this is the adaptation of the blue brain project's code (eFEL) to extact any feature from their library. it takes the path to a folder containing voltage traces and then returns a dataframe with the feature values for each trace
def analyze_feature(path, feature):
    #####
    #this code takes the path to a directory and the eFEL feature, which are well documented, and returns a dataframe of the data for each trace
    table2 = []
    os.chdir(path)
    #sort the dir
    sorted_dir = sorted(os.listdir())
    file_names = []
    for filename in sorted_dir:
        table = pd.DataFrame(columns=[feature])     #create a table that has columns with the name you want
        table.name = feature                        #the tables name
        if filename.endswith(".axgd") or filename.endswith(".axgx"):    #check for the filetype
            [traces] = efel.io.load_neo_file(filename, stim_start=500, stim_end=1500)    #load the trace, and define stim start and stop
            print('Working on file: ', filename)
            file_names.append(filename)
            for data in traces:    #loop through these guys
                #table.rename(columns={feature:filename}, inplace=True) #renaming the columns with the correct file !
                feature_values = efel.getFeatureValues(data, [feature], raise_warnings=None)[0]  #this is the feature extraction
                if feature_values[feature] is not None:
                    # Define the parameters for detection
                    efel.api.setThreshold(-10) # Voltage threshold for detection
                    efel.api.setDerivativeThreshold(10) # dV/dt threshold for detection
                    efel.setIntSetting('strict_stiminterval', True)
                    length = len(table)
                    table.loc[length, feature] = feature_values[feature][0]

                else:
                    efel.api.setThreshold(-10) # Voltage threshold for detection
                    efel.api.setDerivativeThreshold(10) # dV/dt threshold for detection
                    efel.setIntSetting('strict_stiminterval', True)
                    length = len(table)
                    table.loc[length, feature] = feature_values[feature]

            table2.append(table)
            df_concat = pd.concat(table2, axis=1)
    #lets rename the df concat columns with the file names
    df_concat.columns = file_names

    return df_concat




#Cell attached analysis which takes the path to the directory holding the action current traces
#It should be noted that the smoothing used in this function must be manipulated on a cell by cell basis. This is because noise spikes in the positive direction can be mistaken for action potentials
#we have implemented new code to detect cell attached spikes that utilize smoothing with a gaussian filter, differentiating the signal, and then finding the peaks in the first directive with a threshold set based on the interquartile range of the signal, which mitigates the need for a lot of manual inspection and is more robust, though it was not used in the analysis of the paper.
def cell_attached_analysis_2(path, genotype):
    #################
    #this function is used to detect spikes in the cell attached data
    #it requires the path to the directory and the genotype
    #################
    os.chdir(path)
    concat_cv_append = []
    concat_frequency_append = []
    for file_name in os.listdir():
        cv_of_isi_append = []
        frequency_append = []
        if file_name.endswith(".axgd") or file_name.endswith(".axgx"):
            traces = load_neo_file(file_name)
            for cell_attached_data in traces:
                for data in cell_attached_data:
                    times1 = data['T']
                    amps1 = data['A']
                    baseline = np.average(data['A'])  #define a baseline period from which to substract from
                    amps1 = amps1 - baseline  #subtract the baseline from the amplitudes of the first pulse
                    #we generated all the amps into a dataframe, check
                    sampling_rate = 50_000
                    #Filter the signal (savgol)
                    window_length = 75
                    deriv = 0
                    polyorder = 2
                    current_filtered = signal.savgol_filter(x=amps1, window_length=window_length, polyorder=polyorder,
                                                            deriv=deriv, axis=0,
                                                            cval='nearest')  #convolve with a small window, low order polynomial
                    amps1_df = pd.DataFrame(current_filtered)
                    std_trace = np.std(current_filtered)

                    flat_amps_concat_np = current_filtered.flatten('F')
                    flat_amps_concat_np_df = pd.DataFrame(flat_amps_concat_np)
                    median_trace = np.median(flat_amps_concat_np)
                    flat_times = times1.flatten('F')
                    if std_trace < 3.5:
                        width = 0.9 * sampling_rate / 1000
                        peaks, peaks_dict = find_peaks(-flat_amps_concat_np,  # signal
                                                       height=(1.3 * std_trace, 200),
                                                       # Min and max thresholds to detect peaks.
                                                       threshold=None,
                                                       # Min and max vertical distance to neighboring samples.
                                                       distance=(1000),  # Min horizontal distance between peaks.
                                                       prominence=12,
                                                       # Vertical distance between the peak and lowest contour line.
                                                       width=width,
                                                       # Min required width (in bins). E.g. For 50Khz, 10 bins = 5 ms.
                                                       wlen=None,  # Window length to calculate prominence.
                                                       rel_height=1.0,
                                                       # Relative height at which the peak width is measured.
                                                       plateau_size=None)
                        # plt.figure(figsize=(16, 8))
                        # plt.plot(flat_amps_concat_np)
                        # plt.plot(peaks, flat_amps_concat_np[peaks], "x")
                        # plt.show()
                        if len(peaks) > 2:
                            isi_s = np.diff(peaks, axis=0, prepend=peaks[0])[1:] / sampling_rate
                            isis_pd = pd.DataFrame(isi_s)
                            isi_std = np.std(isi_s)
                            isi_mean = np.mean(isi_s)
                            cv_of_isi = isi_std / isi_mean
                            cv_array = np.array(cv_of_isi)
                            #frequency based on ISIs
                            frequency = np.mean(1 / isi_s)
                            frequency_based_on_time = len(peaks) / 10
                            cv_of_isi_append.append(cv_of_isi)
                            frequency_append.append(frequency)
            mean_cv = np.mean(cv_of_isi_append)
            mean_cv_array = np.array(mean_cv, ndmin=2)
            mean_cv_df = pd.DataFrame(mean_cv_array)
            mean_cv_df['file_name'] = file_name
            mean_cv_df['Genotype'] = genotype
            concat_cv_append.append(mean_cv_df)
            mean_frequency = np.mean(frequency_append)
            mean_frequency_array = np.array(mean_frequency, ndmin=2)
            mean_frequency_df = pd.DataFrame(mean_frequency_array)
            mean_frequency_df['file_name'] = file_name
            mean_frequency_df['Genotype'] = genotype
            concat_frequency_append.append(mean_frequency_df)
    cv_append_concat = pd.concat(concat_cv_append)
    cv_append_concat.rename(columns={0: 'CV of ISI'}, inplace=True)
    cv_append_concat.to_excel('CV of ISI' + '.xlsx', index=False)
    frequency_append_concat = pd.concat(concat_frequency_append)
    frequency_append_concat.rename(columns={0: 'Mean Frequency (Hz)'}, inplace=True)
    frequency_append_concat.to_excel('Cell Attached Frequency (Hz)' + '.xlsx', index=False)
    return display(cv_append_concat), display(frequency_append_concat)


#this is a workhorse function that is used to extract the average interspike intervals from a voltage trace. it requires a lot of inputs.
#it resamples the ISI into 1000 points to allow for averaging within and across cells. in the end it returns the mean ISI, the mean voltage trace, the total.
#it requires rescaling to achieve correct shape, which is largely handled in the plotting function below
def collect_isis_global_excel(path1, sample_rate, sweep_start, sweep_end, time_start_s, time_end_s, metadata):
    #################
    # This function is used to extract the average interspike intervals from a voltage trace
    #it requires the path to the directory, the sample rate, the sweep start and end, the time start and end, and the metadata
    #################
    os.chdir(path1)
    num_points = 1000
    sampling_rate = sample_rate
    time_start = time_start_s * sampling_rate
    time_end = time_end_s * sampling_rate
    global_mean_traces = []
    global_mean_isi = []
    filename_list = []
    file_names = sorted(os.listdir())  # Sort the file names

    for filename in file_names:
        if filename.endswith(".axgd") or filename.endswith(".axgx"):
            print('Working on ' + filename)
            filename_list.append(filename)
            [traces] = efel.io.load_neo_file(filename, stim_start=0, stim_end=10000)
            traces = traces[sweep_start:sweep_end]
            if len(traces) > 0:
                file_results = []
                average_isi = []
                for trace in traces:
                    for p in trace:
                        times = (p['T'])
                        times = times[time_start:time_end]  #selected for the first 10 seconds of each sweep
                        voltages = (p['V']).flatten()
                        voltages = voltages[time_start:time_end]  #selected for the first 10 seconds of each sweep
                        times -= times[0]
                        dt = times[2] - times[1]
                        detection_level = -10
                        min_interval = 0.0001
                        spike_times = find_spike_times(voltages, dt, detection_level, min_interval)
                        interspike_intervals = np.diff(spike_times)
                        if len(interspike_intervals) > 0:
                            avg_isi_1 = np.mean(interspike_intervals)
                            average_isi.append(avg_isi_1)
                            resampled_isi_voltage_arrays = []
                            for i in range(len(interspike_intervals)):
                                start_time = spike_times[i]
                                end_time = spike_times[i] + interspike_intervals[i]
                                times_rel = np.linspace(0, 1, num_points)
                                isi_voltage_array = segment(voltages, dt, start_time, end_time)
                                resampled_isi_voltage_arrays.append(
                                    np.interp(times_rel, np.linspace(0, 1, len(isi_voltage_array)), isi_voltage_array))
                            mean_resampled_isi_voltages = np.mean(
                                np.concatenate(resampled_isi_voltage_arrays).reshape(len(resampled_isi_voltage_arrays),
                                                                                     num_points), axis=0)
                            file_results.append(mean_resampled_isi_voltages)
            isi_file_mean = np.mean(np.array(average_isi))
            global_mean_isi.append(isi_file_mean)
            file_mean = np.mean(np.array(file_results), axis=0)
            global_mean_traces.append(file_mean)
    filename_df = pd.DataFrame(filename_list)
    filename_df.rename(columns={0: 'filename'}, inplace=True)
    global_mean_isis = np.array(global_mean_isi)
    global_mean_isi_df = pd.DataFrame(global_mean_isis)
    global_mean_isi_df = pd.concat([filename_df, global_mean_isi_df], axis=1)
    global_mean_isi_df.set_index('filename', inplace=True)
    global_mean_isi_df.sort_index(inplace=True)
    global_mean_isis = global_mean_isi_df.values[:, 0]
    global_mean_isi_df.to_excel(metadata + '_global_mean_isi.xlsx')
    isi_final_mean = np.mean(global_mean_isi, axis=0)
    isi_final_mean = np.array([[isi_final_mean]])
    isi_final_mean_df = pd.DataFrame(isi_final_mean)
    display(isi_final_mean_df)
    isi_final_mean_df.to_excel(metadata + '_isi_final_mean.xlsx')

    global_mean_traces = np.array(global_mean_traces)
    global_mean_traces_df = pd.DataFrame(global_mean_traces)
    global_mean_traces_df = pd.concat([filename_df, global_mean_traces_df], axis=1)
    global_mean_traces_df.set_index('filename', inplace=True)
    global_mean_traces_df.sort_index(inplace=True)
    global_mean_traces = np.array(global_mean_traces_df)
    global_mean_traces_df.to_excel(metadata + '_global_mean_traces.xlsx')

    final_mean_trace = np.mean(global_mean_traces, axis=0)
    final_mean_trace_df = pd.DataFrame(final_mean_trace)
    final_mean_trace_df.to_excel(metadata + '_final_mean_trace.xlsx')

    file_results = np.array(file_results)
    file_results_df = pd.DataFrame(file_results)
    file_results_df = pd.concat([filename_df, file_results_df], axis=1)
    file_results_df.set_index('filename', inplace=True)
    file_results_df.sort_index(inplace=True)
    file_results_df.to_excel(metadata + '_total_trajectories.xlsx')
    print('Analysis complete')
    return final_mean_trace, isi_final_mean, global_mean_traces, file_results, global_mean_isis


#this is a plotting function that is used to plot the voltage traces and the ISIs. it requires the mean trace, the mean ISI, the total trajectory voltages, the metadata, and the save path
#it uses the resampled average trace and the average ISI to plot the voltage trace and the ISI
#it uses matplotlib to plot the data
def plot_voltage_trajectories_two(mean_trace1, mean_trace2, avg_isi1, avg_isi2, total_trajectory_voltages1, total_trajectory_voltages2, mean_trace1_label, mean_trace2_label, mean_trace1_color, mean_trace2_color, save, metadata):

    # Calculate first baseline statistics
    dVdt1 = np.gradient(mean_trace1, axis=0)
    slope1 = np.mean(dVdt1 / 0.1)
    avg_isis1 = np.linspace(0, avg_isi1 * len(mean_trace1), len(mean_trace1))
    #ci1 = 1.96 * np.std(np.array(total_trajectory_voltages1), axis=0) / np.sqrt(len(total_trajectory_voltages1))

    # Calculate second baseline statistics
    dVdt2 = np.gradient(mean_trace2, axis=0)
    slope2 = np.mean(dVdt2 / 0.1)
    avg_isis2 = np.linspace(0, avg_isi2 * len(mean_trace2), len(mean_trace2))
    #ci2 = 1.96 * np.std(np.array(total_trajectory_voltages2), axis=0) / np.sqrt(len(total_trajectory_voltages2))

    # Calculate duration of each trace
    num_points = 1000
    duration1 = (len(mean_trace1) * avg_isi1) / num_points-1
    duration2 = (len(mean_trace2) * avg_isi2) / num_points-1




    # Calculate scaling factors for x-axis
    scale1 = duration1 / len(mean_trace1)
    scale2 = duration2 / len(mean_trace2)


    # plot mean traces with actual durations
    fig, ax = plt.subplots(figsize=(2.75, 2.75))
    ax.plot(np.arange(0, duration1, scale1)[:1000], mean_trace1, color=mean_trace1_color, label=mean_trace1_label, linewidth=0.5)
    ax.plot(np.arange(0, duration2, scale2)[:1000], mean_trace2, color=mean_trace2_color, label=mean_trace2_label, linewidth=0.5)
    # ax.fill_between(np.arange(0, duration1, scale1), mean_trace1 - ci1, mean_trace1 + ci1, alpha=0.3,
    #                 color='#848482', label='95% CI (Trace 1)')
    # ax.fill_between(np.arange(0, duration2, scale2), mean_trace2 - ci2, mean_trace2 + ci2, alpha=0.3,
    #                 color='red', label='95% CI (Trace 2)')
    ax.set_xlabel('Time from spike (ms)', fontsize=9)
    ax.set_ylabel('Membrane Potential (mV)', fontsize=9)
    # ax.set_title('ISI voltage trajectories')
    ax.legend(loc='lower right', borderpad=0.4, labelspacing=.1, fontsize=7)
    ax.tick_params(axis='both', which='both', labelsize=8, width=0.5)  # You can adjust the font size as needed

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.50)
    ax.spines['left'].set_linewidth(0.50)

    min_value = min(np.min(mean_trace1), np.min(mean_trace2))
    #ax.legend(fontsize=14)
    plt.legend(frameon=False, edgecolor='none', fontsize='large', handlelength=2, handleheight=1, loc='lower right')
    #set major ticks on the x and y axis

    #x.tick_params(axis='x', labelsize=14)
    plt.ylim(min_value-1, -30)

    if save == True:
        os.chdir('/Users/HBLANKEN/Library/CloudStorage/OneDrive-UniversityofOklahoma/Beckstead Lab/DA-AD paper files/Grand collection of axograph files/Noise Traces/Figures')
        plt.savefig(metadata + '.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()


#this is a plotting function that is used to plot the voltage traces and the ISIs. it requires the mean trace, the mean ISI, the total trajectory voltages, the metadata, and the save path
def calculate_slope_and_mean_voltage(global_mean_traces, global_mean_isis, save_path, metadata):
    os.chdir(save_path)
    slopes = []
    mean_voltages = []
    min_voltages = []
    for i, trace in enumerate(global_mean_traces):
        if len(trace) == 0:
            print(f"Empty trace found at index {i}")
            continue
        isi = global_mean_isis[i]

        # Scale the trace based on the actual length of the ISI
        time = np.linspace(0, isi, len(trace))

        slope_start_voltage_idx = int(0.3 * len(trace))  # find index of minimum voltage
        end_idx_slope = int(0.7 * len(trace))
        # print('Start Slope:', slope_start_voltage_idx)
        # print('End Slope:', end_idx_slope)
        # print('Time:', time[slope_start_voltage_idx:end_idx_slope])

        min_voltage_index = np.argmin(trace)
        mean_voltage_end_idx = int(0.98 * len(trace))

        plt.plot(time[slope_start_voltage_idx:end_idx_slope], trace[slope_start_voltage_idx:end_idx_slope])

        # Calculate the slope using the scaled trace
        slope = np.gradient(trace[slope_start_voltage_idx:end_idx_slope], time[slope_start_voltage_idx:end_idx_slope]) * 1000

        slopes.append(slope.mean())
        mean_voltages.append(trace[min_voltage_index:mean_voltage_end_idx].mean())
        min_voltages.append(trace[:].min())

    slope_df = pd.DataFrame({'slope': slopes})
    slope_df.to_excel(metadata + '_slopes.xlsx')
    mean_voltage_df = pd.DataFrame({'mean_voltage': mean_voltages})
    mean_voltage_df.to_excel(metadata + '_mean_voltages.xlsx')
    min_voltage_df = pd.DataFrame({'min_voltage': min_voltages})
    min_voltage_df.to_excel(metadata + '_min_voltages.xlsx')

    results_df = pd.concat([slope_df, mean_voltage_df, min_voltage_df], axis=1)
    return results_df


#this is the function to extract the spike voltages, the output from this function is required to run the dvdt, threshold and spike width function.
def mean_spike_voltages(path, sweep_start, sweep_end, detection_level, metadata):
    spike_voltage_list = []
    filename_list = []
    os.chdir(path)
    file_names = sorted(os.listdir())  # Sort the file names
    for filename in file_names:
        # check whether file is in the axgx or axgd format
        if filename.endswith(".axgd") or filename.endswith(".axgx"):
            filename_list.append(filename)
            print('Working on ' + filename)
            [traces] = efel.io.load_neo_file(filename, stim_start=0, stim_end=10000)
            table2 = []
            for data in traces[sweep_start:sweep_end]:
                for p in data:
                    times = (p['T']) / 1000
                    voltages = (p['V'])
                    times -= times[0]
                    dt = times[2] - times[1]
                    detection_level = detection_level
                    min_interval = 0.005
                    spike_times = find_spike_times(voltages, dt, detection_level, min_interval)
                    time_before = .025
                    time_after = .015
                    times_rel = list(np.arange(-time_before, time_after, dt))
                    spike_voltages = []
                    for i in range(0, len(spike_times)):
                        if time_before < spike_times[i] < times[-1] - time_after:
                            spike_voltages.append(
                                segment(voltages, dt, spike_times[i] - time_before, spike_times[i] + time_after))
                    spike_voltage_arrays = [np.array(x) for x in spike_voltages]
                    if len(spike_voltage_arrays) > 0:
                        mean_spike_voltages = [np.mean(k) for k in zip(*spike_voltage_arrays)]
                        table2.append(mean_spike_voltages)
            table3 = np.array(table2)
            file_mean_spike_voltages = np.mean(table3, axis=0)
            spike_voltage_list.append(file_mean_spike_voltages)
    filename_list_df = pd.DataFrame(filename_list)
    filename_list_df.columns = ['Filename']
    global_spike_voltage_list_np = np.array(spike_voltage_list)
    global_spike_voltages_df = pd.DataFrame(global_spike_voltage_list_np)
    global_spike_voltages_df = pd.concat([filename_list_df, global_spike_voltages_df], axis=1)
    global_spike_voltages_df.set_index('Filename', inplace=True)
    global_spike_voltages_df.sort_index(inplace=True)
    mean_spike_voltages = global_spike_voltages_df.mean(axis=0)
    mean_spike_voltages_df = pd.DataFrame(mean_spike_voltages)
    mean_spike_voltages_df.columns = ['Voltage (mV) ' + metadata]
    #mean_spike_voltages_df.index = times_rel

    return global_spike_voltages_df, mean_spike_voltages_df


#a helper function to calculate the dVdt of the average spike, which it requries as an input
def calculate_dVdt(df, dt):
    # Convert the DataFrame to numeric, replacing non-numeric values with NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    dVdt_df = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        voltage = df[column].values
        dVdt = (np.gradient(voltage, dt)) / 1000
        dVdt_df[column] = dVdt
    return dVdt_df

#another workhorse function to take the mean traces and ISI from the 'calculate slope and mean voltages' fucntion and calculate the slope, min, and mean voltage of the ISI
def calculate_slope_and_mean_voltage_90ms_window(global_mean_traces, global_mean_isis, save_path, metadata):
    os.chdir(save_path)
    slopes = []
    mean_voltages = []
    min_voltages = []
    for i, trace in enumerate(global_mean_traces):
        if len(trace) == 0:
            print(f"Empty trace found at index {i}")
            continue
        isi = global_mean_isis[i]
        # print(isi)

        # Scale the trace based on the actual length of the ISI
        time = np.linspace(0, isi, len(trace))

        slope_start_time = 10  # Start time of the window (e.g., 30% of ISI)
        slope_end_time = 100  # End time of the window (e.g., 50ms window)

        # Find the indices corresponding to the start and end times of the window
        slope_start_idx = np.argmin(np.abs(time - slope_start_time))
        slope_end_idx = np.argmin(np.abs(time - slope_end_time))
        slope_slope_start_voltage_idx = int(0.3 * len(trace))  # find index of minimum voltage
        end_idx_slope_slope = int(0.7 * len(trace))
        # print('Start Index:', slope_start_idx)
        # print('End Index:', slope_end_idx)

        min_voltage_index = np.argmin(trace)
        # print(min_voltage_index)
        # print(slope_end_idx)
        mean_voltage_end_idx = int(0.98 * len(trace))

        # Calculate the slope using the scaled trace within the window
        slope = np.gradient(trace[slope_slope_start_voltage_idx:end_idx_slope_slope],
                            time[slope_slope_start_voltage_idx:end_idx_slope_slope]) * 1000

        slopes.append(slope.mean())
        mean_voltages.append(trace[slope_start_idx:slope_end_idx].mean())
        min_voltages.append(trace[:].min())

        plt.plot(time, trace)  # Plot the entire trace
        plt.axvline(time[slope_start_idx], color='r', linestyle='--')  # Vertical line at start of window
        plt.axvline(time[slope_end_idx], color='r', linestyle='--')  # Vertical line at end of window

    slope_df = pd.DataFrame({'slope': slopes})
    slope_df.to_excel(metadata + '_slopes_mAHP.xlsx')
    mean_voltage_df = pd.DataFrame({'mean_voltage': mean_voltages})
    mean_voltage_df.to_excel(metadata + '_mean_voltages_mAHP.xlsx')
    min_voltage_df = pd.DataFrame({'min_voltage': min_voltages})
    min_voltage_df.to_excel(metadata + '_min_voltages_mAHP.xlsx')

    results_df = pd.concat([slope_df, mean_voltage_df, min_voltage_df], axis=1)
    return results_df

#function to determine the spike width and action potential threshold, from the average trace, which the function takes as an input. Requires the sampling rate, the dvdt threshold to detect at, the metadata, and the save path
#the spike width is found at half max by calculating the half max voltage value, and then finding the indices corresponding to the time at spike start and time at spike end
def calculate_spike_widths_and_thresholds(df, sampling_rate, dvdt_threshold, metadata, save_path):
    ##########
    #this code is for calculating spike widths and thresholds
    #it requires a dataframe with the voltage traces and the sampling rate
    #it will output a dataframe with the spike widths and thresholds
    ##########
    os.chdir(save_path)
    spike_widths = []
    filenames = []
    threshold = []
    dt = 1 / sampling_rate
    #the first thing we have to do is set the index to be the filename so that we dont have any strings in the df, but only if its there, so let write an if statement for it
    if 'Filename' in df.columns:
        df = df.set_index('Filename')
    else:
        pass
    for index, row in df.iterrows():
        # Extract voltage trace and time values from the row
        voltages = np.array(row.values)
        times = np.arange(0, len(voltages)) / sampling_rate

        # Find the first derivative of the voltage trace
        dvdt = calculate_dVdt(df, dt)

        # Find the start of the action potential (AP_begin_indices)
        start_indices = np.where(dvdt >= dvdt_threshold)[0]

        if len(start_indices) == 0:
            continue
        AP_begin_index = start_indices[0]
        threshold.append(voltages[AP_begin_index])
        # Find the peak of the action potential
        peak_index = np.argmax(voltages)

        # Calculate the half-max voltage value
        half_max_voltage = (voltages[peak_index] + voltages[AP_begin_index]) / 2

        # Find the indices corresponding to the time at spike start (AP_rise_indices) and time at spike end (AP_fall_indices)
        AP_rise_indices = np.where(voltages >= half_max_voltage)[0]
        AP_fall_indices = np.where(voltages <= half_max_voltage)[0]
        AP_rise_index = AP_rise_indices[0]
        AP_fall_index = AP_fall_indices[-1]

        # Calculate the action potential duration at half width
        spike_width = (times[AP_fall_index] - times[AP_rise_index]) * 100  # Convert to ms
        spike_widths.append(spike_width)

        # Extract the filename from the index
        filenames.append(index)

    spike_width_df = pd.DataFrame({'spike_width_half_max': spike_widths}, index=filenames)
    threshold_df = pd.DataFrame({'threshold': threshold}, index=filenames)
    final_df = spike_width_df.join(threshold_df)
    final_df.to_excel(metadata + '_spike_widths_and_thresholds.xlsx')
    return final_df

#a stand alone function that calculates the spontaneous firing frequency and the CV of the ISI
def collect_frequency_and_cv(path1, sample_rate, sweep_start, sweep_end, time_start_s, time_end_s, metadata):
    def one_by_isi(x):
        return 1 / (x / 1000)

    def calc_cv(interspike_intervals):
        return np.std(interspike_intervals) / np.mean(interspike_intervals)

    os.chdir(path1)
    sampling_rate = sample_rate
    time_start = time_start_s * sampling_rate
    time_end = time_end_s * sampling_rate
    global_mean_isi = []
    isi_file_means = []
    freq_file_means = []
    global_mean_cv = []
    filename_list = []
    file_names = sorted(os.listdir())  # Sort the file names
    for filename in file_names:
        if filename.endswith(".axgd") or filename.endswith(".axgx"):
            print('Working on ' + filename)
            filename_list.append(filename)
            [traces] = efel.io.load_neo_file(filename, stim_start=0, stim_end=10000)
            traces = traces[sweep_start:sweep_end]
            if len(traces) > 0:
                isi_list = []
                cv_list = []
                freq_list = []
                for trace in traces:
                    for p in trace:
                        times = (p['T'])
                        times = times[time_start:time_end]  #selected for the first 10 seconds of each sweep
                        voltages = (p['V']).flatten()
                        voltages = voltages[time_start:time_end]  #selected for the first 10 seconds of each sweep
                        times -= times[0]
                        dt = times[2] - times[1]
                        detection_level = -10
                        min_interval = 0.0001
                        spike_times = find_spike_times(voltages, dt, detection_level, min_interval)
                        interspike_intervals = np.diff(spike_times)
                        if len(interspike_intervals) > 0:
                            avg_isi_1 = np.mean(interspike_intervals)
                            avg_freq_1 = one_by_isi(avg_isi_1)
                            cv_1 = calc_cv(interspike_intervals)
                            isi_list.append(avg_isi_1)
                            cv_list.append(cv_1)
                            freq_list.append(avg_freq_1)
                if len(isi_list) > 0:
                    freq_means = np.mean(np.array(freq_list))
                    freq_file_means.append(freq_means)
                    isi_file_mean = np.mean(np.array(isi_list))
                    isi_file_means.append(isi_file_mean)
                    cv_file_mean = np.mean(np.array(cv_list))
                    global_mean_isi.append(isi_file_mean)
                    global_mean_cv.append(cv_file_mean)

    filename_df = pd.DataFrame(filename_list)
    filename_df.rename(columns={0: 'filename'}, inplace=True)
    global_mean_isis = np.array(global_mean_isi)
    global_mean_isi_df = pd.DataFrame(global_mean_isis)

    freq_means_df = pd.DataFrame(freq_file_means)
    freq_means_df.rename(columns={0: 'Frequency'}, inplace=True)
    freq_means_df = pd.concat([filename_df, freq_means_df], axis=1)
    freq_means_df.set_index('filename', inplace=True)
    freq_means_df.sort_index(inplace=True)

    global_mean_frequencies = global_mean_isi_df.applymap(one_by_isi)
    global_mean_frequencies.rename(columns={0: 'Frequency'}, inplace=True)
    global_mean_frequencies = pd.concat([filename_df, global_mean_frequencies], axis=1)
    global_mean_frequencies.set_index('filename', inplace=True)
    global_mean_frequencies.sort_index(inplace=True)

    global_mean_cvs = np.array(global_mean_cv)
    global_mean_cv_df = pd.DataFrame(global_mean_cvs)
    global_mean_cv_df.rename(columns={0: 'CV_ISI'}, inplace=True)
    global_mean_cv_df.set_index(global_mean_frequencies.index, inplace=True)

    result_df = pd.concat([freq_means_df, global_mean_cv_df], axis=1)
    result_df['Metadata'] = metadata
    result_df.to_excel(metadata + '_frequency_and_cv.xlsx')
    return result_df


#function to calculate the AHC at a specific start and end time. Examples are in the example folders
def ahc_analysis_step_time(path, time_start, time_end, metadata):
    sample_rate = 20_000
    start_time = int(time_start * sample_rate)
    end_time = int(time_end * sample_rate)

    def monoExp(x, m, t, b):
        return m * np.exp(-t * x) + b

    def biExp(x, m1, t1, m2, t2, b):
        return m1 * np.exp(-t1 * x) + m2 * np.exp(-t2 * x) + b

    ahc_append = []
    ahc_auc_append = []

    ahc_max_amp_append = []
    auc_append = []
    tau_append = []
    ahc_max_amp_df_append = []
    os.chdir(path)
    file_names = sorted(os.listdir())
    for file_name in file_names:
        tabla = []
        if file_name.endswith(".axgd") or file_name.endswith(".axgx"):
            traces = load_neo_file(file_name)
            for sk_data in traces:
                for data in sk_data:
                    #this is for the first part of the A-current
                    times1 = data['T'][start_time:end_time]  #this is 16.101 to 16.213
                    amps1 = data['A'][start_time:end_time]
                    #plt.plot(times, amps) #inspect
                    baseline = np.mean(data['A'][39200:39800])  #define a baseline period from which to substract from
                    amps1 = amps1 - baseline  #subtract the baseline from the amplitudes of the first pulse

                    amps1_df = pd.DataFrame(amps1)  #we generated all the amps into a dataframe, check
                    flat_times = np.ndarray.flatten(
                        times1)  #we have all of the times into a flattened numpy array, check
                    tabla.append(amps1_df)  #and we appended all of the amps into a dataframe
                amps_concat = pd.concat(tabla,
                                        axis=1)  #this is correct - we put all the amps for a given trace into a dataframe
            #now we need to average those dataframes by row
            averaged_sk_trace = amps_concat.mean(
                axis=1)  #woohooo this is the averaged SK trace in a pd.DF, this is what we want to work with from here on out!
            #print(np.argmax(averaged_sk_trace)) #this is the index of the max value of the averaged trace

            flat_times = flat_times[
                         np.argmax(averaged_sk_trace):]  #this is the times from the max value to the end of the trace
            averaged_sk_trace = averaged_sk_trace[np.argmax(
                averaged_sk_trace):]  #this is the averaged trace from the max value to the end of the trace

            #Block of code of AHC Max Amp !
            ahc_max_amp = pd.DataFrame.max(
                averaged_sk_trace)  #max of the whole ahc, not just sk component, This is the value!!
            #figure out to how put this in a format that can be read and concatenated
            ahc_max_amp_array = np.array(ahc_max_amp, ndmin=2)  #these lines are new
            ahc_max_amp_df = pd.DataFrame(ahc_max_amp_array)
            ahc_max_amp_df['file_name'] = file_name  #adding a column to add the filename

            ahc_append.append(ahc_max_amp_df)  #Here!
            ahc_max_amp_concat_df = pd.concat(ahc_append)  #This is the variable for ahc

            #Block of code for AHC AUC
            averaged_sk_trace_as_np = averaged_sk_trace.to_numpy()
            flattened_average_sk_trace = np.ndarray.flatten(averaged_sk_trace_as_np)
            ahc_auc = np.trapz(flattened_average_sk_trace) / 1000  #this is the auc of the whole ahc
            #bit of code to get the ahc auc into readable condition
            ahc_auc_array = np.array(ahc_auc, ndmin=2)
            ahc_auc_df = pd.DataFrame(ahc_auc_array)  #here we have the auc data in a dataframe
            ahc_auc_df['file_name'] = file_name
            ahc_auc_append.append(ahc_auc_df)
            ahc_auc_concat_df = pd.concat(ahc_auc_append)  #this is the variable for auc
            plt.plot(flat_times, averaged_sk_trace)  #check
            plt.fill_between(flat_times, 0, flattened_average_sk_trace, color='gray', alpha=0.5, label='AUC')

            #Block of code for kinetics
            trace_for_kinetics = flattened_average_sk_trace[:]
            times_rel = flat_times - flat_times[0]
            times_for_kinetics = times_rel[:]
            trace_for_kinetics_pd = pd.DataFrame(trace_for_kinetics)
            #trace_for_kinetics_pd.to_excel("trace_for_kinetics_pd.xlsx")
            times_for_kinetics_pd = pd.DataFrame(times_for_kinetics)
            #times_for_kinetics_pd.to_excel("times_for_kinetics_pd.xlsx")

            #fit the curve for inactivation tau
            # p0 = [500, .001, 50]  #values near what we expect   #here
            # params, cv = scipy.optimize.curve_fit(monoExp, times_for_kinetics, trace_for_kinetics, p0,
            #                                       bounds=(-np.inf, np.inf),
            #                                       maxfev=100000)  #here  #this fits the training curve with an r-squared of 0.97
            # m, t, b = params  #here

            # p0 = [500, .001, 500, .001, 50]  # initial parameter values
            # params, cv = scipy.optimize.curve_fit(biExp, times_for_kinetics, trace_for_kinetics, p0,
            #                                       bounds=(-np.inf, np.inf),
            #                                       maxfev=100000000)  # fitting the curve
            # m1, t1, m2, t2, b = params
            #
            # #plot results
            # plt.plot(times_for_kinetics, trace_for_kinetics, '.', label="data")
            # plt.plot(times_for_kinetics, biExp(times_for_kinetics, m1, t1, m2, t2, b), '--', label="fitted")
            # plt.title("Fitted Biexponential Curve")
            # plt.legend()
            # plt.show()
            #m, t = params
            sampleRate = 20_000  #hz
            # tauSec = (1 / t) / sampleRate
            # print(tauSec)
    #
    #         #determine quality of fit
    #         squaredDiffs = np.square(trace_for_kinetics - monoExp(times_for_kinetics, m, t, b))  #here
    #         squaredDiffsFromMean = np.square(trace_for_kinetics - np.mean(trace_for_kinetics))
    #         rSquared = 1 - np.sum(squaredDiffs) / np.sum(
    #             squaredDiffsFromMean)  #we want these, but they arent super important to display
    #         #print(f"R^2 = {rSquared}")
    #
    # #plot results
    # plt.plot(times_for_kinetics, trace_for_kinetics, '.', label="data")
    # plt.plot(times_for_kinetics, monoExp(times_for_kinetics, m, t, b), '--', label="fitted")  #here
    # plt.show()
    # plt.title("Fitted Expotential Curve")
    #
    #         #inspect the params
    #         #print(f"Y = {m} * e^(-{t} * x) + {b}")   #the equations are important
    #         #print(f"Tau = {tauSec * 1e6} us")    #but the tau is the most important
    #         plt.show()
    #         tau_flat_ms = tauSec * 1e4
    #
    #         #Bit of code to get tau into working order
    #         if 0 <= tauSec*1e4 <= 300:
    #             tau_array = np.array(tauSec * 1e4, ndmin=2)
    #             tau_df = pd.DataFrame(tau_array)
    #             tau_df['file_name'] = file_name
    #             tau_append.append(tau_df)
    #             tau_concat_df = pd.concat(tau_append)   #this is the variable for tau
    #         else:
    #             print(file_name + ' is not within the range of 0-300ms')
    #
    #lets rename columns and export to excel for each of our metrics
    ahc_max_amp_concat_df.rename(columns={0: 'Control AHC Max_Amplitude (pA)'}, inplace=True)
    ahc_max_amp_concat_df.to_excel('ahc_max_amp ' + metadata + '.xlsx', index=False)

    ahc_auc_concat_df.rename(columns={0: 'Control AHC AUC (pA*s)'}, inplace=True)
    ahc_auc_concat_df.to_excel('ahc_auc ' + metadata + '.xlsx', index=False)
    #
    # tau_concat_df.rename(columns = {0:'Control Decay Tau (ms)'}, inplace=True)
    # tau_concat_df.to_excel('ahc_tau ' + metadata + '.xlsx', index=False)
    #
    return display(ahc_max_amp_concat_df), display(ahc_auc_concat_df)  #, display(tau_concat_df)
