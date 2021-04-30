import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime

############################################################
############################################################
def date_check(from_date, to_date):
    
    datediff = datetime.strptime(to_date, '%Y-%m-%d') - datetime.strptime(from_date, '%Y-%m-%d')
    days = int(datediff.days) + 1

    weeks = int(days / 7)
    if days % 7 == 0:
        print(f'Approved date range. Equal number of days of week.\nNumber of days: {days}\nNumber of weeks: {weeks}')
    else: 
        print(f'Warning! Uneven distribution of days of week. Need to normalise values by day of week or adjust date range.\nNumber of days: {days}')
        
        
############################################################
############################################################
def normalise_weeks(from_date, to_date, data):
    
    datediff = datetime.strptime(to_date, '%Y-%m-%d') - datetime.strptime(from_date, '%Y-%m-%d')
    days = int(datediff.days) + 1

    weeks = int(days / 7)
    
    return(data / weeks)


############################################################
############################################################
def transform_weekday(data):
    data['weekday'] = data['weekday'].replace({1: 'Sunday', 
                                               2: 'Monday', 
                                               3: 'Tuesday', 
                                               4: 'Wednesday',
                                               5: 'Thursday',
                                               6: 'Friday',
                                               7: 'Saturday'})

    
############################################################
############################################################    
def detect_outlier(data):
    
    outliers = []
    
    threshold = 3 # Two standard deviations capture ~95% of values, three ~99%. I opt for something in between.
    mean_1 = np.mean(data)
    std_1 = np.std(data)
    
    if std_1 == 0:
        return(pd.DataFrame())
    
    for y in data:
        z_score = (y - mean_1) / std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers # Rewrite so as to register indices rather than values? Risk of overlapping values, removing wrong values.


############################################################
############################################################
def remove_outliers(data, partition):
    
    new_data = data.copy()
    
    wdays = data.weekday.unique()
    hrs = data.hour.unique()
    grouping_list = data[partition].unique()
    
    for w in wdays:
        for h in hrs:
            for g in grouping_list:
                filtered_data = data.loc[(data.weekday==w) &
                                         (data.hour==h) &
                                         (data[partition]==g)]
                ol = detect_outlier(data=filtered_data.n_pageviews)
                new_data = new_data[~new_data.n_pageviews.isin(ol)]
    
    a = len(data)
    b = len(new_data)
    
    print(f"{round(b / a,2)*100}% of original data remains.")
    
    return(new_data)

    
############################################################
############################################################
def standard_pivot(value, data): 
    
    ## ADD A CATCH TRY for having an unfiltered grouping
    df_piv = data.pivot(index="hour", columns="weekday", values=f"{value}")
    
    n_weekdays = len(df_piv.columns)

    if n_weekdays == 7:
        df_piv = df_piv[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']] # Order logically.
    else:
        print(f'Error: Only {n_weekdays} weekdays are represented in the dataset.')

    return(df_piv)


############################################################
############################################################
def comparison_pivot(value, data, comp_data): 
    
    nominal_present_data = standard_pivot(data=data, value=value) 
    nominal_comparison_data = standard_pivot(data=comp_data, value=value) 
    
    comp_piv = ( (nominal_present_data - nominal_comparison_data) / nominal_comparison_data )
    return(comp_piv)


############################################################
############################################################
def pivot_method(value, data, comp_data=pd.DataFrame()):
    
    if comp_data.empty == True:
        standard_piv = standard_pivot(value=value, data=data) 
        return(standard_piv)
    else:
        comparison_piv = comparison_pivot(value=value, data=data, comp_data=comp_data) 
    return(comparison_piv)


############################################################
############################################################
def filter_params(params, data):
    
    df_filtered = data
    
    for i in params:
        
        try:
            df_filtered[i]
        except KeyError: #raised if this particular parameter doesn't exist.
            print(f"Parameter '{i}' does not exist in the data.")
            continue
        
        df_filtered = df_filtered[df_filtered[i] == params[i]]
        
    return(df_filtered)



############################################################
############################################################
def heatmap(params, data, metrics, dates, comp_data=pd.DataFrame()):
    
    metrics_keys = list(metrics.keys())
    metrics_names = list(metrics.values())
    
    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(1,len(metrics), figsize=(20, 10), squeeze=False) # Don't squeeze so we can use indexing even for 1D subplots.
    
    # Filtered the original dataset based on selected filtering parameters
    filtered_data = filter_params(params=params, data=data)
    
    if comp_data.empty == False: # If we have comparison data... 
        filtered_comp_data = filter_params(params=params, data=comp_data) # Then filter that data. 
        NBR_FORMAT = "1.0%" # And change the number format to percentages (for comparison).
        CENTER = 0 # And change the center value to zero to correctly colour the percentages.
    else:
        filtered_comp_data = comp_data
        NBR_FORMAT = "1.0f"
        CENTER = None
    
    for pos in np.arange(len(metrics)): 
       
        try:
            sns.heatmap(pivot_method
                        (
                            data=filtered_data, 
                            value=metrics_keys[pos], 
                            comp_data=filtered_comp_data
                        ), 
                        annot=True, 
                        linewidths=.5, 
                        fmt=f"{NBR_FORMAT}", # Float with no decimals.
                        cbar=False,
                        cmap="RdYlGn",
                        center=CENTER,
                        ax=ax[0, pos]);
        except ValueError: #raised if heatmap is empty is empty.
            pass
        
        ax[0, pos].text(x=0.5, y=1.12, s=metrics_names[pos], fontsize=16, weight='bold', ha='center', va='bottom', transform=ax[0, pos].transAxes)
        ax[0, pos].text(x=0.5, y=1.07, s=f'{params}', fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax[0, pos].transAxes)
        
        # Adapt date range text depending on if we are comparing to a dataset or not
        if comp_data.empty == False:
            ax[0, pos].text(x=0.5, y=1.02, s=f'{dates["from_date"]} - {dates["to_date"]} vs {dates["from_date_comp"]} - {dates["to_date_comp"]}', fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax[0, pos].transAxes)
        else:
            ax[0, pos].text(x=0.5, y=1.02, s=f'{dates["from_date"]} - {dates["to_date"]}', fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax[0, pos].transAxes)


       
