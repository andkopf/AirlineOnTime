import matplotlib
from numpy.lib.shape_base import _put_along_axis_dispatcher
from numpy.ma.extras import average
from numpy.testing._private.utils import tempdir
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from data_handler import load_data
import os
import seaborn as sns
import numpy as np
import scipy.stats as stats
from bioinfokit.analys import stat
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from tqdm import tqdm
import sys


dir_output = sys.argv[1]
if not os.path.exists(dir_output):
    os.makedirs(dir_output)
data_folder = sys.argv[2]
type_delay = sys.argv[3]


# read in data
data, plane_data, carriers, airport = load_data(data_folder)

# filter data
data = data.dropna(subset=[type_delay])


# plot the world with color as average delay from the specific airport
geometry = [Point(xy) for xy in zip(airport.long, airport.lat)]
gdf = GeoDataFrame(airport, geometry=geometry)   

# get average delay for airport
average_delay_airport = list()
for ap in tqdm(airport.iata):
    temp_data = data[data.Dest == ap]
    average_delay_airport.append(temp_data[type_delay].mean())

# create color map
lower = np.nanmin(average_delay_airport)
upper = np.nanmax(average_delay_airport)
colors_map_average_delay = cm.Reds((average_delay_airport-lower)/(upper-lower))


#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
plt.figure()
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color=colors_map_average_delay, markersize=15)
plt.xlim([-175, -50])
plt.ylim([10, 75])
plt.axis('off')
plt.savefig(os.path.join(dir_output, 'world_map_color_average_delay_zoom.pdf'), bbox_inches='tight')
plt.close()


#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
plt.figure()
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color=colors_map_average_delay, markersize=15)
plt.axis('off')
plt.savefig(os.path.join(dir_output, 'world_map_color_average_delay.pdf'), bbox_inches='tight')
plt.close()



# plot world card with JFK and LAX
airport_lax_jfk = pd.concat([airport[airport['iata'] == 'LAX'], airport[airport['iata'] == 'JFK']])
geometry_lax_jfk = [Point(xy) for xy in zip(airport_lax_jfk.long, airport_lax_jfk.lat)]
gdf_lax_jfk = GeoDataFrame(airport_lax_jfk, geometry=geometry_lax_jfk)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
plt.figure()
gdf_lax_jfk.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=30)
plt.xlim([-130, -60])
plt.ylim([10, 55])
plt.axis('off')
plt.savefig(os.path.join(dir_output, 'world_map_lax_jfk.pdf'), bbox_inches='tight')
plt.close()



# filter for specific destinations
data = data[data['Origin'].isin(['JFK', 'LAX'])]
data = data[data['Dest'].isin(['JFK', 'LAX'])]


# plot barplot delay per year
plt.figure()
ax_sns = sns.barplot(data=data, x='Year', y=type_delay)
plt.xticks(rotation=90)
plt.savefig(os.path.join(dir_output, 'barplot_year.pdf'), bbox_inches='tight')
plt.close()

# plot barplot delay per year and month
data = data.sort_values(by=['Month'])
plt.figure(figsize=(15,4))
ax = plt.subplot(111)
ax_sns = sns.barplot(data=data, x='Year', y=type_delay, hue='Month')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=90)
plt.savefig(os.path.join(dir_output, 'barplot_year_month.pdf'), bbox_inches='tight')
plt.close()

# create seasons variable
winter_month = [12, 1 ,2]
spring_month = [3, 4, 5]
summer_month = [6, 7, 8]
autumn_month = [9, 10, 11]
data['Season'] = ['']*data.shape[0]
for mo in range(1, 13):
    if mo in winter_month:
        data['Season'][data['Month'] == mo] = 'Winter'
    elif mo in spring_month:
        data['Season'][data['Month'] == mo] = 'Spring'
    elif mo in summer_month:
        data['Season'][data['Month'] == mo] = 'Summer'
    elif mo in autumn_month:
        data['Season'][data['Month'] == mo] = 'Autumn'

color_seasons = {'Winter' : 'cornflowerblue', 'Spring' : 'mediumseagreen', 'Summer' : 'yellow', 'Autumn' : 'darkgoldenrod'}

# plot barplot delay per year and season
plt.figure(figsize=(10,4))
ax = plt.subplot(111)
ax_sns = sns.barplot(data=data, x='Year', y=type_delay, hue='Season', palette=color_seasons)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=90)
plt.savefig(os.path.join(dir_output, 'barplot_year_seasons.pdf'), bbox_inches='tight')
plt.close()



# plot number of flights per year-month
data = data.sort_values(by=['Year', 'Month'])
number_of_flights = list()
number_of_flights_to_LAX = list()
number_of_flights_to_JFK = list()
year_x_label = list()
month_x_label = list()
for y in data.Year.unique():
    for m in np.sort(data.Month.unique()):
        temp_data = data[data['Year'] == y]
        temp_data = temp_data[temp_data['Month'] == m]
        if temp_data.shape[0] > 0:
            number_of_flights.append(temp_data.shape[0])
            number_of_flights_to_LAX.append(temp_data[temp_data['Dest'] == 'LAX'].shape[0])
            number_of_flights_to_JFK.append(temp_data[temp_data['Dest'] == 'JFK'].shape[0])
            if not y in year_x_label and m == 1:
                year_x_label.append(y)
            else:
                year_x_label.append('')
            month_x_label.append(m)

linewidth = 2
plt.figure(figsize=(10,4))
ax = plt.subplot(111)
x_axis_time = list(range(len(number_of_flights)))
plt.plot(x_axis_time, number_of_flights, label='Total', linewidth=linewidth)
plt.plot(x_axis_time, number_of_flights_to_LAX, label='To LAX', linestyle='dashed', linewidth=linewidth)
plt.plot(x_axis_time, number_of_flights_to_JFK, label='To JFK', linestyle=(0, (1,1)), linewidth=linewidth)
for i, y in enumerate(year_x_label):
    if y != '':
        plt.axvline(x=x_axis_time[i], linestyle='dashed', linewidth = 0.5, color='gray')
plt.xlabel('Time', fontsize=18)
plt.xticks(x_axis_time, year_x_label, rotation=90)
plt.ylabel('Number of flights', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'number_of_flights.pdf'), bbox_inches='tight')
plt.close()





# plot delay per year-month
data = data.sort_values(by=['Year', 'Month'])
avg_arrival_delay = list()
avg_departure_delay = list()
avg_carrier_delay = list()
avg_weather_delay = list()
avg_nas_delay = list()
avg_security_delay = list()
avg_late_aircraft_delay = list()
year_x_label = list()
month_x_label = list()
ind_first_special_delay = 0
for y in data.Year.unique():
    for m in np.sort(data.Month.unique()):
        temp_data = data[data['Year'] == y]
        temp_data = temp_data[temp_data['Month'] == m]
        if temp_data.shape[0] > 0:
            avg_arrival_delay.append(temp_data.ArrDelay.dropna().mean())
            avg_departure_delay.append(temp_data.DepDelay.dropna().mean())
            avg_carrier_delay.append(temp_data.CarrierDelay.dropna().mean())
            avg_weather_delay.append(temp_data.WeatherDelay.dropna().mean())
            avg_nas_delay.append(temp_data.NASDelay.dropna().mean())
            avg_security_delay.append(temp_data.SecurityDelay.dropna().mean())
            avg_late_aircraft_delay.append(temp_data.LateAircraftDelay.dropna().mean())
            if len(temp_data.LateAircraftDelay.dropna()) == 0:
                ind_first_special_delay +=1
            if not y in year_x_label and m == 1:
                year_x_label.append(y)
            else:
                year_x_label.append('')
            month_x_label.append(m)
avg_arrival_delay = np.asarray(avg_arrival_delay)
avg_departure_delay = np.asarray(avg_departure_delay)
avg_carrier_delay = np.asarray(avg_carrier_delay)
avg_weather_delay = np.asarray(avg_weather_delay)
avg_nas_delay = np.asarray(avg_nas_delay)
avg_security_delay = np.asarray(avg_security_delay)
avg_late_aircraft_delay = np.asarray(avg_late_aircraft_delay)


linewidth = 1
plt.figure(figsize=(10,4))
ax = plt.subplot(111)
x_axis_time = np.asarray(list(range(len(avg_arrival_delay))))
plt.plot(x_axis_time, avg_arrival_delay, label='Arrival', linewidth=linewidth)
plt.plot(x_axis_time, avg_departure_delay, label='Departure', linewidth=linewidth)
plt.plot(x_axis_time, avg_carrier_delay, label='Carrier', linewidth=linewidth)
plt.plot(x_axis_time, avg_weather_delay, label='Weather', linewidth=linewidth)
plt.plot(x_axis_time, avg_nas_delay, label='NAS', linewidth=linewidth)
plt.plot(x_axis_time, avg_security_delay, label='Security', linewidth=linewidth)
plt.plot(x_axis_time, avg_late_aircraft_delay, label='Late Aircraft', linewidth=linewidth)
plt.hlines(y=0, xmin=np.min(x_axis_time), xmax=np.max(x_axis_time), linestyles='dashed', linewidth=0.5, color='gray')
for i, y in enumerate(year_x_label):
    if y != '':
        plt.axvline(x=x_axis_time[i], linestyle='dashed', linewidth = 0.5, color='gray')
plt.xlabel('Time', fontsize=18)
plt.xticks(x_axis_time, year_x_label, rotation=90)
plt.ylabel('Average delay [minutes]', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'types-of-delay-over-time.pdf'), bbox_inches='tight')
plt.close()


linewidth = 1
plt.figure(figsize=(10,4))
ax = plt.subplot(111)
x_axis_time = list(range(len(avg_arrival_delay)))
plt.plot(x_axis_time[ind_first_special_delay:], avg_arrival_delay[ind_first_special_delay:], label='Arrival', linewidth=linewidth)
plt.plot(x_axis_time[ind_first_special_delay:], avg_departure_delay[ind_first_special_delay:], label='Departure', linewidth=linewidth)
plt.plot(x_axis_time[ind_first_special_delay:], avg_carrier_delay[ind_first_special_delay:], label='Carrier', linewidth=linewidth)
plt.plot(x_axis_time[ind_first_special_delay:], avg_weather_delay[ind_first_special_delay:], label='Weather', linewidth=linewidth)
plt.plot(x_axis_time[ind_first_special_delay:], avg_nas_delay[ind_first_special_delay:], label='NAS', linewidth=linewidth)
plt.plot(x_axis_time[ind_first_special_delay:], avg_security_delay[ind_first_special_delay:], label='Security', linewidth=linewidth)
plt.plot(x_axis_time[ind_first_special_delay:], avg_late_aircraft_delay[ind_first_special_delay:], label='Late Aircraft', linewidth=linewidth)
plt.hlines(y=0, xmin=np.min(x_axis_time[ind_first_special_delay:]), xmax=np.max(x_axis_time[ind_first_special_delay:]), linestyles='dashed', linewidth=0.5, color='gray')
for i, y in enumerate(year_x_label[ind_first_special_delay:]):
    if y != '':
        plt.axvline(x=x_axis_time[ind_first_special_delay:][i], linestyle='dashed', linewidth = 0.5, color='gray')
plt.xlabel('Time', fontsize=18)
plt.xticks(x_axis_time[ind_first_special_delay:], year_x_label[ind_first_special_delay:], rotation=90)
plt.ylabel('Average delay [minutes]', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'types-of-delay-over-time_zoom.pdf'), bbox_inches='tight')
plt.close()




# plot scatter of month vs. delay at arrival and color for seasons
# plot scatter of number of flights vs. delay at arrival and color for seasons
data = data.sort_values(by=['Year', 'Month'])
average_delay = list()
number_of_flights = list()
month = list()
color_scatter = list()
label_scatter = list()
for y in data.Year.unique():
    for m in np.sort(data.Month.unique()):
        temp_data = data[data['Year'] == y]
        temp_data = temp_data[temp_data['Month'] == m]

        if temp_data.shape[0] > 0:
            number_of_flights.append(temp_data.shape[0])
            average_delay.append(temp_data[type_delay].mean())
            if m in winter_month:
                color_scatter.append(color_seasons['Winter'])
                label_scatter.append('Winter')
            elif m in spring_month:
                color_scatter.append(color_seasons['Spring'])
                label_scatter.append('Spring')
            elif m in summer_month:
                color_scatter.append(color_seasons['Summer'])
                label_scatter.append('Summer')
            elif m in autumn_month:
                color_scatter.append(color_seasons['Autumn'])
                label_scatter.append('Autumn')

            # with added jitter
            month.append(m + np.random.uniform(-0.4, 0.4, 1)[0])
average_delay = np.asarray(average_delay)
number_of_flights = np.asarray(number_of_flights)
month = np.asarray(month)
color_scatter = np.asarray(color_scatter)
label_scatter = np.asarray(label_scatter)

dict_number_flights_vs_delay = {}
dict_number_flights_vs_delay['Average_delay'] = average_delay
dict_number_flights_vs_delay['Number_of_flights'] = number_of_flights
dict_number_flights_vs_delay['Month'] = month
dict_number_flights_vs_delay['Season'] = label_scatter
data_number_flights_vs_delay = pd.DataFrame(dict_number_flights_vs_delay)

plt.figure()
ax = plt.subplot(111)
for g in np.unique(label_scatter):
    ind_g = np.where(label_scatter == g)[0]
    plt.scatter(month[ind_g], average_delay[ind_g], c=color_scatter[ind_g], label=g)
plt.xlabel('Month', fontsize=18)
plt.xticks(np.sort(data.Month.unique()), np.sort(data.Month.unique()))
plt.ylabel('Average delay at arrival', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatter_avg-delay_vs_month.pdf'), bbox_inches='tight')
plt.close



plt.figure()
ax = plt.subplot(111)
for g in np.unique(label_scatter):
    ind_g = np.where(label_scatter == g)[0]
    plt.scatter(number_of_flights[ind_g], average_delay[ind_g], c=color_scatter[ind_g], label=g)
plt.xlabel('Number of flights', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatter_avg-delay_vs_number-of-flights.pdf'), bbox_inches='tight')
plt.close


plt.figure()
ax = plt.subplot(111)
ax_sns = sns.lmplot(data=data_number_flights_vs_delay, x='Number_of_flights', y='Average_delay', hue='Season', palette=color_seasons)
plt.xlabel('Number of flights', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatterlm_avg-delay_vs_number-of-flights.pdf'), bbox_inches='tight')
plt.close


data_number_flights_vs_delay = data_number_flights_vs_delay.sort_values(by='Month')
plt.figure()
ax_sns = sns.boxplot(data=data_number_flights_vs_delay, x='Season', y='Average_delay', palette=color_seasons)
plt.xlabel('Season', fontsize=18)
plt.ylim([-15, 30])
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_avg-delay_seasons.pdf'), bbox_inches='tight')
plt.close

# perform anova for seasons and delay
fvalue_seasons, pvalue_seasons = stats.f_oneway(data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Winter']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Spring']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Summer']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Autumn']['Average_delay'])

res = stat()
res.tukey_hsd(df=data_number_flights_vs_delay, res_var='Average_delay', xfac_var='Season', anova_model='Average_delay ~ C(Season)')
res.tukey_summary




# only to LAX
# plot scatter of month vs. delay at arrival and color for seasons
# plot scatter of number of flights vs. delay at arrival and color for seasons
data_to_LAX = data[data.Dest == 'LAX'].sort_values(by=['Year', 'Month'])
average_delay = list()
number_of_flights = list()
month = list()
color_scatter = list()
label_scatter = list()
for y in data_to_LAX.Year.unique():
    for m in np.sort(data_to_LAX.Month.unique()):
        temp_data = data_to_LAX[data_to_LAX['Year'] == y]
        temp_data = temp_data[temp_data['Month'] == m]

        if temp_data.shape[0] > 0:
            number_of_flights.append(temp_data.shape[0])
            average_delay.append(temp_data[type_delay].mean())
            if m in winter_month:
                color_scatter.append(color_seasons['Winter'])
                label_scatter.append('Winter')
            elif m in spring_month:
                color_scatter.append(color_seasons['Spring'])
                label_scatter.append('Spring')
            elif m in summer_month:
                color_scatter.append(color_seasons['Summer'])
                label_scatter.append('Summer')
            elif m in autumn_month:
                color_scatter.append(color_seasons['Autumn'])
                label_scatter.append('Autumn')

            # with added jitter
            month.append(m + np.random.uniform(-0.4, 0.4, 1)[0])
average_delay = np.asarray(average_delay)
number_of_flights = np.asarray(number_of_flights)
month = np.asarray(month)
color_scatter = np.asarray(color_scatter)
label_scatter = np.asarray(label_scatter)

dict_number_flights_vs_delay = {}
dict_number_flights_vs_delay['Average_delay'] = average_delay
dict_number_flights_vs_delay['Number_of_flights'] = number_of_flights
dict_number_flights_vs_delay['Month'] = month
dict_number_flights_vs_delay['Season'] = label_scatter
data_number_flights_vs_delay = pd.DataFrame(dict_number_flights_vs_delay)

plt.figure()
ax = plt.subplot(111)
for g in np.unique(label_scatter):
    ind_g = np.where(label_scatter == g)[0]
    plt.scatter(month[ind_g], average_delay[ind_g], c=color_scatter[ind_g], label=g)
plt.xlabel('Month', fontsize=18)
plt.xticks(np.sort(data_to_LAX.Month.unique()), np.sort(data_to_LAX.Month.unique()))
plt.ylabel('Average delay at arrival', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatter_avg-delay_vs_month_to_LAX.pdf'), bbox_inches='tight')
plt.close



plt.figure()
ax = plt.subplot(111)
for g in np.unique(label_scatter):
    ind_g = np.where(label_scatter == g)[0]
    plt.scatter(number_of_flights[ind_g], average_delay[ind_g], c=color_scatter[ind_g], label=g)
plt.xlabel('Number of flights', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatter_avg-delay_vs_number-of-flights_to_LAX.pdf'), bbox_inches='tight')
plt.close


plt.figure()
ax = plt.subplot(111)
ax_sns = sns.lmplot(data=data_number_flights_vs_delay, x='Number_of_flights', y='Average_delay', hue='Season', palette=color_seasons)
plt.xlabel('Number of flights', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatterlm_avg-delay_vs_number-of-flights_to_LAX.pdf'), bbox_inches='tight')
plt.close


data_number_flights_vs_delay = data_number_flights_vs_delay.sort_values(by='Month')
plt.figure()
ax_sns = sns.boxplot(data=data_number_flights_vs_delay, x='Season', y='Average_delay', palette=color_seasons)
plt.xlabel('Season', fontsize=18)
plt.ylim([-15, 30])
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_avg-delay_seasons_to_LAX.pdf'), bbox_inches='tight')
plt.close

# perform anova for seasons and delay
fvalue_seasons, pvalue_seasons = stats.f_oneway(data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Winter']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Spring']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Summer']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Autumn']['Average_delay'])

res = stat()
res.tukey_hsd(df=data_number_flights_vs_delay, res_var='Average_delay', xfac_var='Season', anova_model='Average_delay ~ C(Season)')
res.tukey_summary



# only to JFK
# plot scatter of month vs. delay at arrival and color for seasons
# plot scatter of number of flights vs. delay at arrival and color for seasons
data_to_JFK = data[data.Dest == 'JFK'].sort_values(by=['Year', 'Month'])
average_delay = list()
number_of_flights = list()
month = list()
color_scatter = list()
label_scatter = list()
for y in data_to_JFK.Year.unique():
    for m in np.sort(data_to_JFK.Month.unique()):
        temp_data = data_to_JFK[data_to_JFK['Year'] == y]
        temp_data = temp_data[temp_data['Month'] == m]

        if temp_data.shape[0] > 0:
            number_of_flights.append(temp_data.shape[0])
            average_delay.append(temp_data[type_delay].mean())
            if m in winter_month:
                color_scatter.append(color_seasons['Winter'])
                label_scatter.append('Winter')
            elif m in spring_month:
                color_scatter.append(color_seasons['Spring'])
                label_scatter.append('Spring')
            elif m in summer_month:
                color_scatter.append(color_seasons['Summer'])
                label_scatter.append('Summer')
            elif m in autumn_month:
                color_scatter.append(color_seasons['Autumn'])
                label_scatter.append('Autumn')

            # with added jitter
            month.append(m + np.random.uniform(-0.4, 0.4, 1)[0])
average_delay = np.asarray(average_delay)
number_of_flights = np.asarray(number_of_flights)
month = np.asarray(month)
color_scatter = np.asarray(color_scatter)
label_scatter = np.asarray(label_scatter)

dict_number_flights_vs_delay = {}
dict_number_flights_vs_delay['Average_delay'] = average_delay
dict_number_flights_vs_delay['Number_of_flights'] = number_of_flights
dict_number_flights_vs_delay['Month'] = month
dict_number_flights_vs_delay['Season'] = label_scatter
data_number_flights_vs_delay = pd.DataFrame(dict_number_flights_vs_delay)

plt.figure()
ax = plt.subplot(111)
for g in np.unique(label_scatter):
    ind_g = np.where(label_scatter == g)[0]
    plt.scatter(month[ind_g], average_delay[ind_g], c=color_scatter[ind_g], label=g)
plt.xlabel('Month', fontsize=18)
plt.xticks(np.sort(data_to_JFK.Month.unique()), np.sort(data_to_JFK.Month.unique()))
plt.ylabel('Average delay at arrival', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatter_avg-delay_vs_month_to_JFK.pdf'), bbox_inches='tight')
plt.close



plt.figure()
ax = plt.subplot(111)
for g in np.unique(label_scatter):
    ind_g = np.where(label_scatter == g)[0]
    plt.scatter(number_of_flights[ind_g], average_delay[ind_g], c=color_scatter[ind_g], label=g)
plt.xlabel('Number of flights', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatter_avg-delay_vs_number-of-flights_to_JFK.pdf'), bbox_inches='tight')
plt.close


plt.figure()
ax = plt.subplot(111)
ax_sns = sns.lmplot(data=data_number_flights_vs_delay, x='Number_of_flights', y='Average_delay', hue='Season', palette=color_seasons)
plt.xlabel('Number of flights', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'scatterlm_avg-delay_vs_number-of-flights_to_JFK.pdf'), bbox_inches='tight')
plt.close


data_number_flights_vs_delay = data_number_flights_vs_delay.sort_values(by='Month')
plt.figure()
ax_sns = sns.boxplot(data=data_number_flights_vs_delay, x='Season', y='Average_delay', palette=color_seasons)
plt.xlabel('Season', fontsize=18)
plt.ylim([-15, 30])
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_avg-delay_seasons_to_JFK.pdf'), bbox_inches='tight')
plt.close

# perform anova for seasons and delay
fvalue_seasons, pvalue_seasons = stats.f_oneway(data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Winter']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Spring']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Summer']['Average_delay'], data_number_flights_vs_delay[data_number_flights_vs_delay.Season == 'Autumn']['Average_delay'])

res = stat()
res.tukey_hsd(df=data_number_flights_vs_delay, res_var='Average_delay', xfac_var='Season', anova_model='Average_delay ~ C(Season)')
res.tukey_summary



# does the day of the week influence delay 
data = data.sort_values(by='DayOfWeek')
data_filtered = data[data[type_delay] > (data[type_delay].mean() - 3*data[type_delay].std())]
data_filtered = data_filtered[data_filtered[type_delay] < (data[type_delay].mean() + 3*data[type_delay].std())]
plt.figure()
ax_sns = sns.boxplot(data=data_filtered, x='DayOfWeek', y=type_delay)
plt.xlabel('Day of Week', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_day-of-week.pdf'), bbox_inches='tight')
plt.close()


data_filtered_to_LAX = data_filtered[data_filtered['Dest'] == 'LAX']
plt.figure()
ax_sns = sns.boxplot(data=data_filtered_to_LAX, x='DayOfWeek', y=type_delay)
plt.hlines(y=0, xmin=-0.5, xmax=6.5, linestyles='dashed', color='black', linewidth=1)
plt.xlabel('Day of Week', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_day-of-week_to_LAX.pdf'), bbox_inches='tight')
plt.close()

data_filtered_to_JFK = data_filtered[data_filtered['Dest'] == 'JFK']
plt.figure()
ax_sns = sns.boxplot(data=data_filtered_to_JFK, x='DayOfWeek', y=type_delay)
plt.hlines(y=0, xmin=-0.5, xmax=6.5, linestyles='dashed', color='black', linewidth=1)
plt.xlabel('Day of Week', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_day-of-week_to_JFK.pdf'), bbox_inches='tight')
plt.close()




# kde of average delay 
plt.figure()
ax = plt.subplot(111)
ax_sns = sns.kdeplot(data=data_filtered, x=type_delay, hue="Dest")
plt.ylabel('Density', fontsize=18)
plt.xlabel('Average delay', fontsize=18)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # Put a legend to the right of the current axis
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(dir_output, 'kde_average_delay.pdf'), bbox_inches='tight')
plt.close()




# boxplot of delay with respect to carrier
plt.figure()
ax_sns = sns.boxplot(data=data_filtered, x='UniqueCarrier', y=type_delay)
plt.hlines(y=0, xmin=-0.5, xmax=6.5, linestyles='dashed', color='black', linewidth=1)
plt.xlabel('Carrier', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_delay_carrier.pdf'), bbox_inches='tight')
plt.close

# boxplot of delay with respect to carrier
plt.figure()
ax_sns = sns.boxplot(data=data_filtered_to_LAX, x='UniqueCarrier', y=type_delay)
plt.hlines(y=0, xmin=-0.5, xmax=6.5, linestyles='dashed', color='black', linewidth=1)
plt.xlabel('Carrier', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_delay_carrier_to_LAX.pdf'), bbox_inches='tight')
plt.close

# boxplot of delay with respect to carrier
plt.figure()
ax_sns = sns.boxplot(data=data_filtered_to_JFK, x='UniqueCarrier', y=type_delay)
plt.hlines(y=0, xmin=-0.5, xmax=5.5, linestyles='dashed', color='black', linewidth=1)
plt.xlabel('Carrier', fontsize=18)
plt.ylabel('Average delay', fontsize=18)
plt.savefig(os.path.join(dir_output, 'boxplot_delay_carrier_to_JFK.pdf'), bbox_inches='tight')
plt.close

