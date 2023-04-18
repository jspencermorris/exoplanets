#%% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import seaborn as sns
import requests
import tabulate
from scipy import stats

#pd.set_option('display.max_rows', 100)
#pd.set_option('display.width', 500)

#%% Import Data
filename = 'PSCompPars_2023.04.04_17.16.52.csv'
psc = pd.read_csv(filename, comment='#')


#%% Preliminary Analysis
print(psc.head())
print(len(psc))
print(psc.info())


#%% Transform Data

# Check for missing values
print("\nMissing values count:")
print(psc.isnull().sum())

#Check for duplciated rows
print("Number of duplicated rows:", psc.duplicated().sum())

#Filter out all columns except for the following:
cols_to_keep = ['pl_name', 'discoverymethod', 'disc_year', 'pl_controv_flag', \
    'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_dens', 'hostname', 'sy_dist', \
    'st_mass', 'st_spectype', 'st_lum', 'st_teff', 'st_met', 'st_age', \
    'pl_nespec', 'pl_ntranspec']

psc = psc.loc[:, cols_to_keep]
# identify critical nulls
#   drop any rows w/ nan's in these columns:  pl_rade, pl_bmasse
#   NOTE consider dropping any rows w/ nan's in this column:  pl_orbper
psc.isna().sum()
psc = psc.drop( psc[ psc['pl_rade'].isna() ].index)
psc = psc.drop( psc[ psc['pl_bmasse'].isna() ].index)

# calculate pl_dens for rows that are nan
#   ref: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
earth_radius_cm = 6356.752*10**5
earth_mass_g = 5.9722*10**29
psc['pl_dens'].fillna(earth_mass_g*psc['pl_bmasse'] / (earth_radius_cm*psc['pl_rade'])**3, inplace=True )

# identify outliers based on these assumptions:
#   exclude objects with orbiatal periods more than 200 years, \
#       which is well beyond the period of the most-distant planet in the Sol system
#   exclude objects with radii more than 30 times the radius of Earth, as they are brown dwarfs
#   exclude objects with masses more than 3200 times the mass of Earth, as they are by definition brown dwarfs
#   exclude objects with densities greater than lead (11.35 g/cm3)
#   exclude objects orbiting stars with masses more than the theoretical accretion limit (120 solar masses)
#   exclude objects orbiging stars older than the universe (13.8 Gy)
psc.describe()
psc.drop( psc[ psc['pl_orbper'] > 73000].index, inplace=True)
psc.drop( psc[ psc['pl_rade'] > 30].index, inplace=True)
psc.drop( psc[ psc['pl_bmasse'] > 3200 ].index, inplace=True)
psc.drop( psc[ psc['pl_dens'] > 11.35 ].index, inplace=True)
psc.drop( psc[ psc['st_mass'] > 120 ].index, inplace=True)
psc.drop( psc[ psc['st_age'] > 13.8 ].index, inplace=True)
psc.drop( psc[ psc['st_teff'] > 8000 ].index, inplace=True)

# remove controversial objects then drop the controversial column
psc.drop( psc[ psc['pl_controv_flag'] == 1 ].index, inplace=True)
psc.drop(['pl_controv_flag'], axis=1, inplace=True)

#   it looks like the spectral type is mostly null, so many consider dropping that column
#   consider dropping any rows w/ nan's in this column:  pl_orbper

# reset row indices
psc.reset_index(drop=True, inplace=True)


#%% Time and Techniques

psc [ psc['disc_year'] == 1992][['disc_year','hostname','pl_name','discoverymethod']]

method_counts = psc.groupby(psc['discoverymethod']).size().sort_values(ascending=False)
method_stats = pd.DataFrame({
    'Discovery Method': method_counts.index,
    'Count': method_counts.values
}).reset_index(drop=True)

method_stats.index += 1

headers = ["#", "\033[1mDiscovery Method\033[0m", "\033[1mCount\033[0m"]

table = tabulate.tabulate(method_stats, headers=headers, tablefmt='fancy_grid')

title = "\033[1mThree Techniques Account for Over 99% of All Discoveries\033[0m"

print(title)
print(table)


# barchart to compare number of discoveries by technique
plt.style.use('seaborn')
sns.set_style("whitegrid")
fig, axes = plt.subplots(figsize=(8, 6))
discovery_counts = psc['discoverymethod'].value_counts()
discovery_counts.plot(kind='barh', ax=axes)
axes.set_xlabel('# of Discoveries', fontsize=12, fontweight='bold')
axes.set_ylabel('Discovery Technique', fontsize=12, fontweight='bold')
axes.set_title('Three Techniques Account for Over 99% of All Discoveries', fontsize=14, fontweight='bold')
axes.set_xscale('log')
axes.invert_yaxis()
for i, v in enumerate(discovery_counts):
    axes.text(v + 10, i - 0.1, str(v), fontsize=10)
plt.show()

# add new col to categorize by main discovery methods
top_methods = psc.groupby(psc['discoverymethod']).size().sort_values(ascending=False)[0:3].index
new_methods = {method: (method if method in top_methods else 'Others') for method in list(psc['discoverymethod'].unique())}
psc['method2'] = psc['discoverymethod'].map(new_methods)

# Boxplots based on discoverymethod
box_vars = ['disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'sy_dist', 'st_mass', 'st_lum', 'st_teff', 'st_met']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharey='row')

axes = axes.flatten()

for i, var in enumerate(box_vars):
    sns.boxplot(x=var, y='discoverymethod', data=psc, orient='h', ax=axes[i])
    axes[i].set_xlabel(var.capitalize(), fontsize=12, fontweight='bold')
    if i == 0:
        axes[i].set_ylabel('Discovery Method', fontsize=12, fontweight='bold')
    else:
        axes[i].set_ylabel(None)
        
    axes[i].set_title(f"{var.capitalize()} by Discovery Method", fontsize=14, fontweight='bold')
    
fig.subplots_adjust(hspace = 0.3, wspace=0.15)
fig.suptitle('Boxplots of Key Numerical Variables Across Different Discovery Methods', fontsize=16, fontweight='bold', y=1.03)

plt.show()



# Bin years into five-year periods
bins = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
labels = ['1990-1994', '1995-1999', '2000-2004', '2005-2009', '2010-2014', '2015-2019', '2020-2025']
period = pd.cut(psc['disc_year'], bins=bins, labels=labels)

# Compute counts and cumulative counts by period
counts_by_period = psc.groupby(period)['disc_year'].count()
cumulative_counts_by_period = counts_by_period.cumsum()

# Compute percentage of total exoplanets discovered in each period
percent_of_total_by_period = (counts_by_period / cumulative_counts_by_period.iloc[-1] * 100).round(1)

# Combine counts, cumulative counts, and percentage of total into a table
periodic_counts_table = pd.concat([counts_by_period, cumulative_counts_by_period, percent_of_total_by_period], axis=1)
periodic_counts_table.columns = ['Count', 'Cumulative', '% of Total']
periodic_counts_table.index.name = 'Period'

# Format and print table
table = tabulate.tabulate(periodic_counts_table, headers='keys', tablefmt='fancy_grid', floatfmt=".1f")
title = "\033[1mExoplanet Discoveries Accelerated in the 2010s\033[0m"
print(title)
print(table)

# cumulative plot to see how discoveries have trended over time
plt.style.use('seaborn')
sns.set_style("whitegrid")
fig, axes = plt.subplots(figsize=(8, 6))
year_counts = psc.groupby(psc['disc_year']).size().cumsum()
axes.plot(year_counts.index, year_counts.values, marker='o')
axes.set_xlabel('Year of Discovery', fontsize=12, fontweight='bold')
axes.set_ylabel('Cumulative Number of Discoveries', fontsize=12, fontweight='bold')
axes.set_title("Exoplanet Discoveries Accelerated in the 2010s", fontsize=14, fontweight='bold')
axes.grid(color='lightgray', linestyle='--')
plt.show()

# table showing discovery method by year
print(psc.groupby([psc['discoverymethod'], psc['disc_year']]).size().unstack())

##############################################################################
#discovery year
plt.style.use('seaborn')
sns.set_style("whitegrid")
fig, axes = plt.subplots(figsize=(8, 6))
discovery_counts = psc.groupby('disc_year').size()
axes.plot(discovery_counts.index, discovery_counts.values)
axes.set_xlabel('Year of Discovery', fontsize=12, fontweight='bold')
axes.set_ylabel('# of Planets', fontsize=12, fontweight='bold')
axes.set_title('Total Discoveries Over Time', fontsize=14, fontweight='bold')

plt.show()


pass 
##############################################################################

############################################################################
#planent discover methods
discovery_counts = psc.pivot_table(index='disc_year', columns='discoverymethod', values='pl_name', aggfunc='count')
sns.set_style("whitegrid")
# Create a line plot
fig, ax = plt.subplots(figsize=(10,5))
for column in discovery_counts.columns:
    ax.plot(discovery_counts.index, discovery_counts[column], label=column)

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('# Of Planet Discoveries', fontsize=12, fontweight='bold')
ax.set_title('Total Planet Discoveries by Discovery Method', fontsize=14, fontweight='bold')
ax.legend()
plt.show()

pass
############################################################################

#Interactive bar chart of exoplanet discovery & discovery methods over time. 
#Documentation found at https://plotly.com/python/bar-charts/


method_year_counts = psc.groupby(['discoverymethod', 'disc_year']).size().unstack().transpose()
method_year_counts = method_year_counts[method_year_counts.sum().sort_values().index]

fig = go.Figure()

for method in method_year_counts.columns:
    num_planets = int(method_year_counts[method].sum())
    legend_label = f'{method}: {num_planets}'
    fig.add_trace(go.Bar(x=method_year_counts.index, y=method_year_counts[method], name=legend_label))

fig.update_layout(
    title='Exoplanet Discoveries by Observing Method',
    xaxis_title='Year',
    yaxis_title='Number of Exoplanet Discoveries',
    barmode='stack'
)
fig.show()


decade_bins = [1990,2000,2010,2020,2030]
decade_labels = ["1990's","2000's","2010's","2020's"]
psc['disc_decade'] = pd.cut(psc['disc_year'].map(int), bins=decade_bins, labels=decade_labels, right=False, include_lowest=True)

# Group by decade
method_decade_counts = psc.groupby([psc['method2'], psc['disc_decade']]).size().unstack()
headers = ['\033[1mDiscovery Method\033[0m'] + [f'\033[1m{col}\033[0m' for col in method_decade_counts.columns]

table = tabulate.tabulate(method_decade_counts, headers=headers, tablefmt='fancy_grid')

title = "\033[1mExoplanet Discoveries by Observing Method\033[0m"

print(title)
print(table)


# Stacked bar chart of Discovery Method
fig, axes = plt.subplots(figsize = (20,5))

sorted_values = sorted(psc['discoverymethod'].value_counts(), reverse=True)
bottom_vals = np.sum(sorted_values[3:])
bottom_vals_list = [bottom_vals]
top_vals = sorted_values[:3]
top_vals.extend(bottom_vals_list)

axes.barh('Total Discoveries',top_vals[0],label = 'Transit')

others = list(psc.discoverymethod.value_counts().index[:3])
others.extend(['All Other Techniques Combined'])

for i in range(1,len(top_vals)):
    if i ==1:
        axes.barh('Total Discoveries',top_vals[i], left=top_vals[0], label=others[i])
       
    else:
        axes.barh('Total Discoveries',top_vals[i], left=sum(top_vals[:i]), label=others[i])
   
axes.set_xlabel('Total Number of Discoveries')
axes.set_title('Three Techniques account over 99% of all discoveries')
axes.set_yticklabels([])
axes.tick_params(axis='y', which='both', left=False)
axes.legend()
plt.show()

#%% Orbital Period

# add new col to categorize by main discovery methods
top_methods = psc.groupby(psc['discoverymethod']).size().sort_values(ascending=False)[0:3].index
new_methods = {method: (method if method in top_methods else 'Others') for method in list(psc['discoverymethod'].unique())}
psc['method2'] = psc['discoverymethod'].map(new_methods)

# Boxplots based on discoverymethod
box_vars = ['disc_year', 'pl_orbper', 'pl_rade', 'sy_dist', 'st_mass', \
    'st_teff']
fig, axes = plt.subplots(nrows=6, ncols=1, sharey='row',figsize = (10,20))
x_labels = ['Discovery Year',"Planent Orbital Period (Eath Days)",\
            "Planent Radius (Relative to Earth's Raduis)","System Distance \
(Parsecs)","Star Mass (Relative to Sun's Mass)", "Star Temperature \
(Kelvin)"]
k = 0
for i in range(6):
    for j in range(1):
        a = sns.boxplot(x=box_vars[k], y='discoverymethod', data=psc, orient='h', ax=axes[i])
        a.set(ylabel=None)
        a.set(xlabel=x_labels[k])
        k += 1
fig.subplots_adjust(hspace = 0.3, wspace=0.3)
fig.suptitle('Boxplots of key numerical variables across different Discovery Methods')
plt.show()




box_vars = ['disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'sy_dist', 'st_mass', 'st_lum', 'st_teff', 'st_met']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharey='row')

k = 0
for i in range(3):
    for j in range(3):
        a = sns.boxplot(x=box_vars[k], y='discoverymethod', data=psc, orient='h', ax=axes[i,j])
        a.set_xlabel(box_vars[k].capitalize(), fontsize=12, fontweight='bold')
        a.set_ylabel(None)
        a.set_title(f"{box_vars[k].capitalize()} by Discovery Method", fontsize=14, fontweight='bold')
        k += 1

fig.suptitle('Boxplots of Key Numerical Variables Across Different Discovery Methods', fontsize=16, fontweight='bold')
fig.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()




# Orbital Period
orb_period = psc['pl_orbper']
print('Summary Statistics for Orbital Period:')
print(orb_period.describe())
print('Number of Null Periods: ', orb_period.isnull().sum())
print('Ratio of planets w/ Null Periods: {:.2%}'.format(orb_period.isnull().sum() / len(orb_period)))

# Basic histogram - period
fig, ax = plt.subplots()
ax.hist(orb_period, bins=20, color='cornflowerblue', alpha=0.7)
ax.set_xlabel('Orbital Period (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('# of Planets', fontsize=12, fontweight='bold')
ax.set_title('Histogram of Orbital Period', fontsize=14, fontweight='bold')
plt.show()

# Basic histogram - period, zoom1 - past Jupiter
fig, ax = plt.subplots()
ax.hist(orb_period, bins=20, range=(0, 5000), color='mediumaquamarine', alpha=0.7)
ax.set_xlabel('Orbital Period (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('# of Planets', fontsize=12, fontweight='bold')
ax.set_title('Histogram of Orbital Period (Zoom 1: Past Jupiter)', fontsize=14, fontweight='bold')
plt.show()

# Basic histogram - period, zoom2 - past Mars
fig, ax = plt.subplots()
ax.hist(orb_period, bins=20, range=(0, 1000), color='sandybrown', alpha=0.7)
ax.set_xlabel('Orbital Period (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('# of Planets', fontsize=12, fontweight='bold')
ax.set_title('Histogram of Orbital Period (Zoom 2: Past Mars)', fontsize=14, fontweight='bold')
plt.show()

# Basic histogram - period, zoom3 - past Earth
fig, ax = plt.subplots()
ax.hist(orb_period, bins=20, range=(0, 400), color='indianred', alpha=0.7)
ax.set_xlabel('Orbital Period (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('# of Planets', fontsize=12, fontweight='bold')
ax.set_title('Histogram of Orbital Period (Zoom 3: Past Earth)', fontsize=14, fontweight='bold')
plt.show()

# Basic histogram - period, zoom4 - past Mercury
fig, ax = plt.subplots()
ax.hist(orb_period, bins=20, range=(0, 150), color='purple', alpha=0.7)
ax.set_xlabel('Orbital Period (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Vast majority have periods shorter than Mercury', fontsize=14, fontweight='bold')
plt.show()


# Boxplots of period based on method2 and disc_decade
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
sns.boxplot(x='disc_decade', y='pl_orbper', data=psc, orient='v', ax=ax1)
sns.boxplot(x='method2', y='pl_orbper', data=psc, orient='v', ax=ax2)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_title('Orbital Period by Decade', fontsize=14, fontweight='bold')
ax1.set_xlabel('Discovery Decade', fontsize=12, fontweight='bold')
ax1.set_ylabel('Orbital Period (days)', fontsize=12, fontweight='bold')
ax2.set_title('Orbital Period by Discovery Method', fontsize=14, fontweight='bold')
ax2.set_xlabel('Discovery Method', fontsize=12, fontweight='bold')
ax2.set_ylabel('Orbital Period (days)', fontsize=12, fontweight='bold')
plt.show()

# Calculate summary statistics by method2 and disc_decade
method_decade_stats = psc.groupby(['method2', 'disc_decade'])['pl_orbper'].agg(['mean', 'std', 'min', 'max']).reset_index()
method_decade_stats.columns = ['Discovery Method', 'Discovery Decade', 'Mean Orbital Period', 'Standard Deviation', 'Minimum Orbital Period', 'Maximum Orbital Period']

# Format and print table
table = tabulate.tabulate(method_decade_stats.values.tolist(), headers=method_decade_stats.columns, tablefmt='fancy_grid')
print('\033[1mOrbital Period by Discovery Decade and Discovery Method\033[0m')
print(table)


# create new col to estimate mean distance from host star, in AU
#   use kepler's law to calculate the distance:  R_orbit ~ (T^2 * M*)^(1/3)
#   ref: https://www.jpl.nasa.gov/edu/teach/activity/exploring-exoplanets-with-kepler/#:~:text=What%20Kepler%27s%20Third%20Law%20means%20is%20that%20for,T%20is%20the%20planet%27s%20orbital%20period%20in%20years.
earth_kepler_ratio = (365.2**2)**(1/3)
psc['pl_orbr_est'] = (psc['pl_orbper']**2*psc['st_mass'])**(1/3)/earth_kepler_ratio
psc['pl_orbr_est'].describe()

# create new col to categorize by estimated orbital radius
pl_orbr_bins = [0,0.5,1.55,10,40]
pl_orbr_labels = ["Tight","Inner","Middle","Outer"]
psc['pl_orbloc'] = pd.cut(psc['pl_orbr_est'].map(float, na_action='ignore'), bins=pl_orbr_bins, labels=pl_orbr_labels, right=False, include_lowest=True)

orbloc_table = psc.groupby(['pl_orbloc'], as_index=False)['pl_orbloc'].count()
print(orbloc_table)

# histogram of discovery methods

#%% Planet Radius
print(psc['pl_rade'].describe())
psc.groupby('discoverymethod')['pl_rade'].describe()
print(psc.groupby('method2')['pl_rade'].describe())
print(psc.groupby('disc_decade')['pl_rade'].describe())
print(psc.groupby('pl_orbloc')['pl_rade'].describe())
print(psc.groupby(['method2','disc_decade'])['pl_rade'].describe())
print(psc.groupby(['method2','pl_orbloc'])['pl_rade'].describe())

# basic histogram - radius
fig, axes = plt.subplots()
axes.hist(psc['pl_rade'], bins=20)
plt.show()

###############################################################################
# exoplanent radius
fig, ax = plt.subplots(figsize=(10, 5))

# Set x and y labels and title
ax.set_xlabel('Planet Radius (Relative to Earths)', fontsize=12, fontweight='bold')
ax.set_ylabel('# of Planets', fontsize=12, fontweight='bold')
ax.set_title('Two regions of planet radii are predominant', fontsize=14, fontweight='bold')

ax.axvline(1, color='k', linestyle='dashed', linewidth=1)
ax.text(1.1, 190, 'Earth')

ax.axvline(3.9, color='k', linestyle='dashed', linewidth=1)
ax.text(4.05, 150, 'Neptune')

ax.axvline(11.2, color='k', linestyle='dashed', linewidth=1)
ax.text(11.35, 110, 'Jupiter')

# Set number of bins and plot histogram
num_bins = 200
ax.hist(psc['pl_rade'], bins=num_bins, color='teal')

# Show plot
plt.show()


pass
###############################################################################

# density plot - radius
sns.set_style("whitegrid")
sns.set_palette("Dark2")
fig, ax = plt.subplots()
sns.kdeplot(psc['pl_rade'], color='k', fill=True)
ax.set_title('Density of Planet Radii', fontweight='bold')
ax.set_xlabel('Planet Radius (relative to Earth)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
plt.show()

# overlapping histograms - radius
sns.set_palette("Dark2")
plt.hist(psc[psc['discoverymethod']=='Transit']['pl_rade'], label='Transit', bins=20, alpha=0.5)
plt.hist(psc[psc['discoverymethod']=='Radial Velocity']['pl_rade'], label='RV', bins=20, alpha=0.5)
plt.hist(psc[psc['discoverymethod']=='Microlensing']['pl_rade'], label='Microlens', bins=20, alpha=0.5)
plt.legend()
plt.xlabel('Planet Radius (relative to Earth)', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.show()

# overlapping histograms - radius, normalized
sns.set_palette("Dark2")
fig, axes = plt.subplots()
plt.hist(psc[psc['discoverymethod']=='Transit']['pl_rade'], label='Transit', bins=20, alpha=0.5, density=True)
plt.hist(psc[psc['discoverymethod']=='Radial Velocity']['pl_rade'], label='RV', bins=20, alpha=0.5, density=True)
plt.hist(psc[psc['discoverymethod']=='Microlensing']['pl_rade'], label='Microlens', bins=20, alpha=0.5, density=True)
axes.set_xscale('linear')
plt.legend()
plt.xlabel('Planet Radius (relative to Earth)', fontweight='bold')
plt.ylabel('Density', fontweight='bold')
plt.show()

# overlapping histograms - radius, w/ kde
sns.set_style("white")
sns.set_palette("Dark2")
sns.histplot(data=psc, x="pl_rade", kde=True, hue='method2')
plt.xlabel('Planet Radius (relative to Earth)', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.show()

# histogram of - radius, zoom on sizes less than Neptune
sns.set_palette("Dark2")
fig, axes = plt.subplots()
psc['pl_rade'].hist(bins=20, range=(0.5,3.5), density=True)
plt.title('Distribution of Planet Radii', fontweight='bold')
plt.xlabel('Planet Radius (relative to Earth)', fontweight='bold')
plt.ylabel('Density', fontweight='bold')
kde = stats.gaussian_kde(psc['pl_rade'])
xx = np.linspace(0, 3.5, 1000)
axes.plot(xx, kde(xx))
plt.show()

# add new col to categorize by radius
rad_bins = [0, 0.5, 2, 6, 4000]
rad_labels = ["Tiny", "Small", "Medium", "Large"]
psc['rad_cat'] = pd.cut(psc['pl_rade'].map(float), bins=rad_bins, labels=rad_labels, right=False, include_lowest=True)

# Boxplots of radius based on method2 and disc_decade
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
sns.boxplot(data=psc, x='method2', y='pl_rade', ax=ax1)
sns.boxplot(data=psc, x='disc_decade', y='pl_rade', ax=ax2)
ax1.set_title('Planet Radius by Discovery Method', fontsize=14, fontweight='bold')
ax1.set_xlabel('Discovery Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Planet Radius (Earth radii)', fontsize=12, fontweight='bold')
ax2.set_title('Planet Radius by Discovery Decade', fontsize=14, fontweight='bold')
ax2.set_xlabel('Discovery Decade', fontsize=12, fontweight='bold')
ax2.set_ylabel('Planet Radius (Earth radii)', fontsize=12, fontweight='bold')
plt.show()

# Calculate summary statistics by method2 and disc_decade
rad_method_decade_stats = psc.groupby(['method2', 'disc_decade'])['pl_rade'].agg(['mean', 'std', 'min', 'max']).reset_index()
rad_method_decade_stats.columns = ['Discovery Method', 'Discovery Decade', 'Mean Radius', 'Standard Deviation', 'Minimum Radius', 'Maximum Radius']

# Format and print table
table = tabulate.tabulate(rad_method_decade_stats.values.tolist(), headers=rad_method_decade_stats.columns, tablefmt='fancy_grid')
print('\033[1mPlanet Radius by Discovery Method and Discovery Decade\033[0m')
print(table)



#%% Mass
print(psc['pl_bmasse'].describe())
psc.groupby('discoverymethod')['pl_bmasse'].describe()
print(psc.groupby('method2')['pl_bmasse'].describe())
print(psc.groupby('disc_decade')['pl_bmasse'].describe())
print(psc.groupby('pl_orbloc')['pl_bmasse'].describe())
print(psc.groupby(['method2','disc_decade'])['pl_bmasse'].describe())
print(psc.groupby(['method2','pl_orbloc'])['pl_bmasse'].describe())
print(psc.groupby(['pl_orbloc','rad_cat'])['pl_bmasse'].describe())

# basic histogram - mass
fig, axes = plt.subplots()
axes.hist(psc['pl_bmasse'], bins=20)
plt.show()

###############################################################################
# Planets masses
fig, ax = plt.subplots(figsize=(10,5))

ax.set_xlabel('Planet Mass (relative to Earth)', fontweight='bold')
ax.set_ylabel('# of Planets', fontweight='bold')
ax.set_title('Planet Mass', fontweight='bold')

ax.hist(psc.pl_bmasse, bins=200)
plt.show()
pass
###############################################################################

# Histogram - mass, zoom1
fig, axes = plt.subplots()
axes.hist(psc['pl_bmasse'], bins=2000)
axes.set_xlim(0,50)
axes.set_xlabel('Planet Mass (relative to Earth)', fontweight='bold')
axes.set_ylabel('# of Planets', fontweight='bold')
axes.set_title('Planet Mass (Zoomed In)', fontweight='bold')
plt.show()

###############################################################################
ax.hist(psc.pl_bmasse,bins=200)
ax.set_xlabel('Planet Mass (relative to Earth)', fontweight='bold')
ax.set_ylabel('# of Planets', fontweight='bold')
ax.set_title('Planet Mass', fontweight='bold')
plt.show()
pass

# Planet mass zoomed in 
fig, ax = plt.subplots(figsize=(10,5))
earthlike = psc[(psc.pl_bmasse) <=20]
ax.hist(earthlike.pl_bmasse,bins=200)
ax.set_xlabel('Planet Mass (relative to Earth)', fontweight='bold')
ax.set_ylabel('# of Planets', fontweight='bold')
ax.set_title('Planet Mass (Zoomed In)', fontweight='bold')
plt.show()

###############################################################################
# Planets masses
fig, ax = plt.subplots(1,2, squeeze=False, figsize=(15,5))
fig.suptitle('Vast majority of discoveries are lower-mass than Neptune', fontweight='bold')

ax[0,0].set_xlabel('Planet Mass (relative to Earth)', fontweight='bold')
ax[0,0].set_ylabel('# of Planets', fontweight='bold')
ax[0,0].axvline(300, color='k', linestyle='dashed', linewidth=1)
ax[0,0].text(400, 3300, 'Jupiter')
ax[0,0].hist(psc.pl_bmasse, bins=30)

ax[0,1].hist(psc['pl_bmasse'], bins=3000)
ax[0,1].set_xlim(0,25)
ax[0,1].set_xlabel('Planet Mass (relative to Earth)', fontweight='bold')
ax[0,1].set_ylim(0,600)
ax[0,1].axvline(1, color='k', linestyle='dashed', linewidth=1)
ax[0,1].text(1.5, 520, 'Earth')
ax[0,1].axvline(17.1, color='k', linestyle='dashed', linewidth=1)
ax[0,1].text(17.6, 520, 'Neptune')

plt.show()
###############################################################################

# Boxplots of mass based on method2 and disc_decade
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
sns.boxplot(data=psc, x='method2', y='pl_bmasse', ax=ax1)
sns.boxplot(data=psc, x='disc_decade', y='pl_bmasse', ax=ax2)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_title('Planet Mass by Discovery Method', fontsize=14, fontweight='bold')
ax1.set_xlabel('Discovery Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Planet Mass (relative to Earth)', fontsize=12, fontweight='bold')
ax2.set_title('Planet Mass by Discovery Decade', fontsize=14, fontweight='bold')
ax2.set_xlabel('Discovery Decade', fontsize=12, fontweight='bold')
ax2.set_ylabel('Planet Mass (relative to Earth)', fontsize=12, fontweight='bold')
plt.show()

# Calculate summary statistics by method2 and disc_decade
mass_method_decade_stats = psc.groupby(['method2', 'disc_decade'])['pl_bmasse'].agg(['mean', 'std', 'min', 'max']).reset_index()
mass_method_decade_stats.columns = ['Discovery Method', 'Discovery Decade', 'Mean Mass', 'Standard Deviation', 'Minimum Mass', 'Maximum Mass']

# Format and print table
table = tabulate.tabulate(mass_method_decade_stats.values.tolist(), headers=mass_method_decade_stats.columns, tablefmt='fancy_grid')
print('\033[1mPlanet Mass by Discovery Method and Discovery Decade\033[0m')
print(table)

# add line break after table
print('\n')


# basic scatterplot of planet radius vs. mass
# NOTE - there may be an outlier (convert from loglog back to linear to see)
print(psc[['pl_rade','pl_bmasse']].describe())
psc.plot.scatter(x='pl_bmasse', y='pl_rade', \
    xlabel='\nEst. Planet Mass (relative to Earth)\n', \
    ylabel='\nPlanet Radius (relative to Earth)\n', \
    loglog=True)
plt.title('Relationship of Planet Estimated Mass and Radius', fontweight='bold')
plt.xlabel('Est. Planet Mass (relative to Earth)', fontweight='bold')
plt.ylabel('Planet Radius (relative to Earth)', fontweight='bold')
plt.show()


# Scatterplot of planet radius vs. mass, colored by discovery method
# NOTE - there may be an outlier (convert from loglog back to linear to see)
print(psc[['pl_rade','pl_bmasse']].describe())
disc_method_cmap = { method : ( 'orange' if method=='Transit' \
    else 'magenta' if method=='Radial Velocity' \
    else 'green' if method=='Microlensing' \
    else 'black') for method in list(psc['discoverymethod'].unique()) }
disc_method_colors = psc['discoverymethod'].map(disc_method_cmap)
fig, axes = plt.subplots()
axes.scatter(x=psc['pl_bmasse'], y=psc['pl_rade'], c=disc_method_colors, \
    alpha=0.5, edgecolors='none')
axes.set_xlabel('\nEst. Planet Mass (relative to Earth)\n', fontweight='bold')
axes.set_ylabel('\nPlanet Radius (relative to Earth)\n', fontweight='bold')
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_title('\nRelationship of Planet Estimated Mass and Radius\n', fontweight='bold')
legend_colors = [ Line2D([0],[0],color='orange',lw=4,label='Transit'), \
    Line2D([0],[0],color='magenta',lw=4,label='Radial Velocity'), \
    Line2D([0],[0],color='green',lw=4,label='Microlensing'), \
    Line2D([0],[0],color='black',lw=4,label='Others') ]
plt.legend(handles=legend_colors, loc='best')

# add Earth and Jupiter as separate points
earth_mass = 1.0
earth_radius = 1.0
jupiter_mass = 317.8
jupiter_radius = 11.2

axes.scatter([earth_mass], [earth_radius], c='deepskyblue', marker='o', s=150, edgecolors='k', linewidths=1.5, label='Earth')
axes.scatter([jupiter_mass], [jupiter_radius], c='red', marker='o', s=150, edgecolors='k', linewidths=1.5, label='Jupiter')

# add legend for Earth and Jupiter
legend_earth = Line2D([0], [0], marker='o', color='deepskyblue', label='Earth', markersize=10, markerfacecolor='deepskyblue')
legend_jupiter = Line2D([0], [0], marker='o', color='red', label='Jupiter', markersize=10, markerfacecolor='red')
plt.legend(handles=[legend_colors[0], legend_colors[1], legend_colors[2], legend_colors[3], legend_earth, legend_jupiter], loc='best')

plt.show()


#Tables
#m_bins = [0, 0.75, 1.25, 3, 8, 15, 4000]
#m_labels = ["Sub-Terrestrial", "Terrestrial", "Super-Terrestrial", "Sub-Giant", "Giant", "Super-Giant"]
#psc['bmasse_cat'] = pd.cut(psc['pl_bmasse'].map(int), bins=m_bins, labels=m_labels, right=False, include_lowest=True)
# add new col to categorize by mass
#m_bins = [0, 0.75,1.25,3,8,15,4000]
#m_labels = ["Sub-Terrestrial","Terrestrial","Super-Terrestrial","Sub-Giant", "Giant", "Super-Giant"]
#psc['bmasse_cat'] = pd.cut(psc['pl_bmasse'].map(int), bins=m_bins, labels=m_labels, right=False, include_lowest=True)
#print(psc.groupby('bmasse_cat').describe())
#psc.groupby('bmasse_cat')['discoverymethod'].describe()
#print(psc.groupby('discoverymethod')['bmasse_cat'].describe())
#print(psc.groupby('disc_decade')['bmasse_cat'].describe())
#print(psc.groupby('pl_orbloc')['bmasse_cat'].describe())
#print(psc.groupby('bmasse_cat')['discoverymethod'].value_counts().unstack())
#print(psc.groupby(['method2','bmasse_cat'])['bmasse_cat'].describe())
#print(psc.groupby(['method2','bmasse_cat'])['disc_decade'].value_counts().unstack())
#print(psc.groupby(['bmasse_cat','rad_cat'])['discoverymethod'].describe())
#print(psc.groupby(['bmasse_cat','rad_cat'])['discoverymethod'].value_counts().unstack())
#print(psc.groupby(['bmasse_cat'])['rad_cat'].value_counts().unstack())
#print(psc.groupby(['bmasse_cat','rad_cat'])['pl_orbloc'].value_counts().unstack())

from tabulate import tabulate #for some reason I have to call it here again otherwise it errors out.

#Relationship of Planet Estimated Mass and Radius Table
m_bins = [0, 0.75, 2, 10, 80, 400, 4000]
m_labels = ["Sub-Terrestrial", "Terrestrial", "Super-Terrestrial", "Sub-Giant", "Giant", "Super-Giant"]
psc['bmasse_cat'] = pd.cut(psc['pl_bmasse'].map(float), bins=m_bins, labels=m_labels, right=False, include_lowest=True)

mass_table = psc.groupby(['bmasse_cat'])['bmasse_cat'].count()
print(mass_table)

bmasse_cat_stats = psc.groupby('bmasse_cat')['pl_bmasse'].describe()
pl_orbloc_stats = psc.groupby('pl_orbloc')['bmasse_cat'].describe()
table = tabulate(
    bmasse_cat_stats.reset_index(),
    headers=['', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
    showindex='never', tablefmt='fancy_grid')
table += '\n\n'
table += tabulate(
    pl_orbloc_stats.reset_index(),
    headers=['', 'count', 'unique', 'top', 'freq'],
    showindex='never', tablefmt='fancy_grid')

# print the table with title
print('\033[1mRelationship of Planet Estimated Mass and Radius\033[0m')
print(table)


#%% Density

###############################################################################
# planents densities relive to earths
#interesting cut off at 6
fig, ax = plt.subplots(figsize=(10,5))

ax.set_xlabel("Planet Density (Relative to Earth's)", fontweight='bold')
ax.set_ylabel('# of Planets', fontweight='bold')
ax.set_title("Planet Densities", fontweight='bold')

ax.text(5.61, 90, 'Earth')
ax.axvline(5.51, color='k', linestyle='dashed', linewidth=1)

ax.text(1.43, 90, 'Jupiter')
ax.axvline(1.33, color='k', linestyle='dashed', linewidth=1)

ax.hist(psc.pl_dens, bins=200)
plt.show()
pass
###############################################################################


#%% System Distance
print(psc['sy_dist'].describe())
print('Null Distances: ', psc['sy_dist'].isnull().sum())
print('Ratio of planets w/ Null Distances: ', psc['sy_dist'].isnull().sum()/psc['sy_dist'].size)

###############################################################################
# system distance vs density
fig, ax = plt.subplots()
ax.scatter(psc.sy_dist, psc.pl_dens)

# ax.set_xscale('log') 
# ax.set_yscale('log')  
plt.scatter(1, 1, c='red', marker='x', s=200,label ='Earth')

ax.set_xlabel('System Distance', fontweight='bold')
ax.set_ylabel('Planetary Density', fontweight='bold')
ax.set_title('System Distance vs Planetary Density', fontweight='bold')
ax.legend()
plt.show()

###############################################################################

# Interactive scatterplot of exoplanets within fifty lightyears with radius and mass between 80% and 120% of earth.
# Documentation found at https://plotly.com/python/line-and-scatter/
earth_mass_range = (0.75, 1.25)  # 75% to 125% of Earth's mass
earth_radius_range = (0.75, 1.25)  # 75% to 125% of Earth's radius
max_distance = 50

# Convert parsecs to light-years
psc['sy_dist_ly'] = psc['sy_dist'] * 3.262

earth_like_planets = psc[    (psc['pl_bmasse'].between(earth_mass_range[0], earth_mass_range[1])) &
    (psc['pl_rade'].between(earth_radius_range[0], earth_radius_range[1])) &
    (psc['sy_dist_ly'] <= max_distance)
]
marker_size_factor = 10
marker_sizes = earth_like_planets['pl_rade'] * marker_size_factor

fig = go.Figure(data=go.Scatter(
    x=earth_like_planets['pl_bmasse'],
    y=earth_like_planets['pl_rade'],
    mode='markers',
    marker=dict(
        size=marker_sizes,
        color=earth_like_planets['sy_dist_ly'],
        colorscale='viridis_r',
        cmin=earth_like_planets['sy_dist_ly'].min(),
        cmax=earth_like_planets['sy_dist_ly'].max(),
        colorbar=dict(title='Distance (light-years)'),
    ),
    text=earth_like_planets['pl_name'],
    hovertemplate='<b>%{text}</b><br>' +
                  'Mass: %{x:.2f} Earth Masses<br>' +
                  'Radius: %{y:.2f} Earth Radii<br>' +
                  'Distance: %{marker.color:.2f} light-years<br>',
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=[1],
    y=[1],
    mode='markers',
    marker=dict(
        color='lightblue',
        size=marker_sizes.min(),
        symbol='circle',
        line=dict(
            color='black',
            width=1
        )
    ),
    name='Earth'
))

fig.update_layout(
    xaxis_title='Planet Mass (Earth Masses)',
    yaxis_title='Planet Radius (Earth Radii)',
    title='Planets with Mass and Radius Similar to Earth within 50 Light Years (Between 75% and 125%)',
    legend=dict(x=0, y=1)
)
fig.show()

###############################################################################
# planents densities relive to earths
#interesting cut off at 6
fig, ax = plt.subplots(figsize=(10,5))

ax.set_xlabel("Stellar Distance (ly)", fontweight='bold')
ax.set_ylabel('# of Stars', fontweight='bold')
ax.set_title("Stellar Distances", fontweight='bold')
ax.set_xlim(0,10000)

ax.text(600, 150, 'Betelgeuse')
ax.axvline(550, color='k', linestyle='dashed', linewidth=1)
ax.text(2650, 150, 'Deneb')
ax.axvline(2600, color='k', linestyle='dashed', linewidth=1)

ax.hist(psc.sy_dist_ly, bins=500)
plt.show()
pass
###############################################################################


#%% Star Mass
print(psc['st_mass'].describe())
print('Null Mass: ', psc['st_mass'].isnull().sum())
print('Ratio of planets w/ Null Mass: ', psc['st_mass'].isnull().sum()/psc['st_mass'].size)

###############################################################################
# Stars mass appears to be nearly evenly distributed around the suns Mass
fig, ax = plt.subplots(figsize=(10,5))

ax.set_xlabel('Star Mass (Relative to The Sun)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Stars', fontsize=12, fontweight='bold')
ax.set_title("Star's Mass", fontsize=14, fontweight='bold')

ax.hist(psc.st_mass,bins=200)
plt.show()
pass
###############################################################################


#%% Star Spectral Type
print(psc['st_spectype'].describe())
print('Null Spectral Type: ', psc['st_spectype'].isnull().sum())
print('Ratio of planets w/ Null Spectral Type: ', psc['st_spectype'].isnull().sum()/psc['st_spectype'].size)


#%% Star Luminosity
print(psc['st_lum'].describe())
print('Null Luminosity: ', psc['st_lum'].isnull().sum())
print('Ratio of planets w/ Null Luminosity: ', psc['st_lum'].isnull().sum()/psc['st_lum'].size)

###############################################################################
# Stars mass vs Luminosity
fig, ax = plt.subplots(figsize=(10,5))
colors = psc.st_mass
sns.regplot(psc.st_mass,psc.st_lum)

ax.set_xlabel('Stars Mass (Reletive to Sun)')
ax.set_ylabel('unit: relative to Sun’s luminosity log10(Solar)')
ax.set_title('Stars Mass vs Luminosity')
plt.show()
###############################################################################

###############################################################################
#Stars mass vs luminosity
fig, ax = plt.subplots(figsize=(10,5))
colors = psc.st_age
scatter = ax.scatter(psc.st_mass,psc.st_lum, c = colors, s=100, edgecolors='black')
 
ax.set_xlabel('Star Mass (Reletive to Sun)')
ax.set_ylabel('Luminosity Relative to Sun’s luminosity log10(Solar)')
ax.set_title("Star's Mass vs Luminosity")

cbar = plt.colorbar(scatter)
cbar.set_label('Star Mass (reletive to Sun)')
plt.show()
###############################################################################


#%% Star Effective Temperature
print(psc['st_teff'].describe())
print('Null Effective Temp: ', psc['st_teff'].isnull().sum())
print('Ratio of planets w/ Null Effective Temp: ', psc['st_teff'].isnull().sum()/psc['st_teff'].size)


#%% Star Metallicity
print(psc['st_met'].describe())
print('Null Metallicity: ', psc['st_met'].isnull().sum())
print('Ratio of planets w/ Null Metallicity: ', psc['st_met'].isnull().sum()/psc['st_met'].size)


#%% Star Age
print(psc['st_age'].describe())
print('Null Age: ', psc['st_age'].isnull().sum())
print('Ratio of planets w/ Null Age: ', psc['st_age'].isnull().sum()/psc['st_age'].size)

# Scatterplot of star's metallicity and age and planet's density
print('METALICITY\n',psc['st_met'].describe())
print('AGE\n',psc['st_age'].describe())
print('DENSITY\n',psc['pl_dens'].describe())
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(psc['st_met'], psc['st_age'], psc['pl_dens'])
ax.set_xlabel('Stellar Metallicity (dex)')
ax.set_ylabel('Stellar Age (Gy)')
ax.set_zlabel('Planet Density (g/cm3)')
plt.show()


#%% System

print(psc.groupby(['bmasse_cat'])['rad_cat'].value_counts().unstack())

# barchart to show the frequency of the # of planets per star
psc['hostname'].value_counts().describe()
psc['hostname'].value_counts().value_counts()

# plot the frequency of planets in a system
ax = psc['hostname'].value_counts().value_counts().plot(kind='bar')
ax.set_title('Few detected multi-planetary systems', fontweight='bold')
ax.set_xlabel('# of Planets within a System', fontweight='bold')
ax.set_ylabel('# of Detected Planets', fontweight='bold')
ax.set_yscale('log')
plt.show()

# plot to compare the 10 most-populated systems w/ Sol system
systems_1 = psc.groupby('hostname').filter(lambda row: (row.hostname.size > 5))
systems_1_hosts = systems_1['hostname'].unique()
sns.relplot(data=systems_1, x='pl_orbr_est', y='hostname', size='pl_bmasse')
plt.show()

# plot to compare the systems w/ most Earth-like planets w/ Sol system
systems_2_hosts = earth_like_planets['hostname'].unique()
systems_2 = psc[psc['hostname'].isin(systems_2_hosts)].copy()
systems_2['is_earth_like'] = systems_2['pl_name'].isin(earth_like_planets['pl_name'])
g = sns.relplot(data=systems_2, x='pl_orbr_est', y='hostname', size='pl_bmasse', hue='is_earth_like', palette=['teal', 'red'])
g.set_axis_labels('Estimated Orbital Period (days)', 'Host Star Name')
plt.subplots_adjust(top=0.9)
plt.xlabel('Total Planet Orbital Estimate (AU)', fontweight='bold')
plt.ylabel('Host Star', fontweight='bold')
plt.title('Orbital Estimates of Earth-Like Planets Around Their Star', fontweight='bold')

###########################################################################################

systems_2['st_teff'].describe()
systems_2['st_spectype'].unique()

###########################################################################################

# plot to compare the systems w/ most Earth-like planets w/ Sol system (Includes SOL up to and including Earth)
systems_2_hosts = earth_like_planets['hostname'].unique()
systems_2 = psc[psc['hostname'].isin(systems_2_hosts)].copy()

solar_system = pd.DataFrame({
    'pl_name': ['Mercury', 'Venus', 'Earth'],
    'hostname': ['Sun', 'Sun', 'Sun'],
    'pl_orbr_est': [0.387, 0.723, 1.0],
    'pl_rade': [0.383, 0.949, 1.0],
    'pl_bmasse': [0.0553, 0.815, 1.0],
    'pl_dens': [5.427, 5.243, 5.515]
})

# concatenate the two data frames
systems_2 = pd.concat([systems_2, solar_system])

systems_2['is_earth_like'] = systems_2['pl_name'].isin(earth_like_planets['pl_name'])

sns.relplot(data=systems_2, x='pl_orbr_est', y='hostname', size='pl_bmasse', hue='is_earth_like', palette=['teal', 'red'], height=5, aspect=3)
plt.xlabel('Total Planet Orbital Estimate (AU)', fontweight='bold')
plt.ylabel('Host Star', fontweight='bold')
plt.title('Orbital Estimates of Earth-Like Planets Around Their Star + Sol (Mercury, Venus and Earth)', fontweight='bold')
plt.annotate('Earth', xy=(0, 4), xytext=(1.03, 5.9), ha='center', fontsize=12, arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.show()


#%% Planetary Spectroscopy
# add new col to summarize spectroscopic measurements
psc['pl_spec'] = psc['pl_nespec']+psc['pl_ntranspec']
print(psc['pl_spec'].describe())
psc_spec = psc[psc['pl_spec']>0]
print('Planets w/ spectroscopy Measurements: ', psc_spec['pl_spec'].size)

# histogram of the number of spectroscopy measurements per planet
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlabel("# of Acquired Spectra)", fontweight='bold')
ax.set_ylabel('# of Planets', fontweight='bold')
ax.set_title("Most compositional measurements are based on few spectra", fontweight='bold')
ax.hist(psc_spec['pl_spec'], bins=20)
plt.show()
pass

# add new col to categorize by # of spectroscopy measurements
spec_bins = [0,20,200,2000]
spec_labels = ["Few","Several","Many"]
psc['spec_cat'] = pd.cut(psc['pl_spec'].map(int, na_action='ignore'), bins=spec_bins, labels=spec_labels, right=False, include_lowest=True)
psc_spec = psc[psc['pl_spec']>0]

# counts of spec measurements by categories
print(psc_spec['pl_spec'].describe())
print(psc_spec.groupby('discoverymethod')['spec_cat'].value_counts().unstack())
print(psc_spec.groupby('disc_decade')['spec_cat'].value_counts().unstack())
print(psc_spec.groupby('pl_orbloc')['spec_cat'].value_counts().unstack())
print(psc_spec.groupby('rad_cat')['spec_cat'].value_counts().unstack())
print(psc_spec.groupby('bmasse_cat')['spec_cat'].value_counts().unstack())
print(psc_spec.groupby(['bmasse_cat','rad_cat'])['spec_cat'].value_counts().unstack())


#%% new 


