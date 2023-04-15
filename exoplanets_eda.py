#%% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import seaborn as sns
import requests
from tabulate import tabulate
import tabulate


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

# remove controversial objects then drop the controversial column
psc.drop( psc[ psc['pl_controv_flag'] == 1 ].index, inplace=True)
psc.drop(['pl_controv_flag'], axis=1, inplace=True)

#   it looks like the spectral type is mostly null, so many consider dropping that column
#   consider dropping any rows w/ nan's in this column:  pl_orbper

# reset row indices
psc.reset_index(drop=True, inplace=True)

# Descriptive statistics
method_counts = psc.groupby(psc['discoverymethod']).size().sort_values(ascending=False)
method_stats = pd.DataFrame({
    'Discovery Method': method_counts.index,
    'Count': method_counts.values
}).reset_index(drop=True)

method_stats.index += 1

headers = ["#", "\033[1mDiscovery Method\033[0m", "\033[1mCount\033[0m"]

table = tabulate.tabulate(method_stats, headers=headers, tablefmt='fancy_grid')

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



# Summary statistics for discovery year
disc_year_stats = psc['disc_year'].describe().to_frame().reset_index().rename(
    columns={'index': '\033[1mStatistic\033[0m', 'disc_year': '\033[1mValue\033[0m'}
)

# Group by year
disc_year_counts = psc.groupby(psc['disc_year']).size().reset_index().rename(
    columns={'disc_year': '\033[1mYear\033[0m', 0: '\033[1mCount\033[0m'}
)

# Format table
table1 = tabulate.tabulate(disc_year_stats, headers='keys', tablefmt='fancy_grid', showindex=False)
table2 = tabulate.tabulate(disc_year_counts, headers='keys', tablefmt='fancy_grid', showindex=False)

# Print tables
print('\n' + table1)
print('\n' + table2)




# cumulative plot to see how discoveries have trended over time
plt.style.use('seaborn')
sns.set_style("whitegrid")
fig, axes = plt.subplots(figsize=(8, 6))
year_counts = psc.groupby(psc['disc_year']).size().cumsum()
axes.plot(year_counts.index, year_counts.values, marker='o')
axes.set_xlabel('Year of Discovery', fontsize=12, fontweight='bold')
axes.set_ylabel('Cumulative Number of Discoveries', fontsize=12, fontweight='bold')
axes.set_title("Exoplanet Discoveries Accelerated in the 2010's", fontsize=14, fontweight='bold')
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
    title='Exoplanet Discoveries by Observing Method Over Time',
    xaxis_title='Year',
    yaxis_title='Number of Exoplanet Discoveries',
    barmode='stack'
)
fig.show()


# table showing discovery method by decade
decade_bins = [1990,2000,2010,2020,2030]
decade_labels = ["1990's","2000's","2010's","2020's"]
psc['disc_decade'] = pd.cut(psc['disc_year'].map(int), bins=decade_bins, labels=decade_labels, right=False, include_lowest=True)
# Group by decade
method_decade_counts = psc.groupby([psc['method2'], psc['disc_decade']]).size().unstack()
headers = ['\033[1mDiscovery Method\033[0m'] + [f'\033[1m{col}\033[0m' for col in method_decade_counts.columns]
table = tabulate.tabulate(method_decade_counts, headers=headers, tablefmt='fancy_grid')

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
print('% of planets w/ Null Periods: {:.2%}'.format(orb_period.isnull().sum() / len(orb_period)))

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
ax.set_title('Histogram of Exoplanet Orbital Periods (Past Mercury)', fontsize=14, fontweight='bold')
plt.show()


# Boxplots of period based on method2 and disc_decade
print(psc.groupby('disc_decade')['pl_orbper'].describe())
print(psc.groupby('method2')['pl_orbper'].describe())
print(psc['pl_orbper'].isnull().groupby(psc['method2']).sum())
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
sns.boxplot(x='disc_decade', y='pl_orbper', data=psc, orient='v', ax=ax1)
sns.boxplot(x='method2', y='pl_orbper', data=psc, orient='v', ax=ax2)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_title('Period Boxplots by Decade', fontsize=14, fontweight='bold')
ax1.set_xlabel('Discovery Decade', fontsize=12, fontweight='bold')
ax1.set_ylabel('Orbital Period (days)', fontsize=12, fontweight='bold')
ax2.set_title('Period Boxplots by Discovery Method', fontsize=14, fontweight='bold')
ax2.set_xlabel('Discovery Method', fontsize=12, fontweight='bold')
ax2.set_ylabel('Orbital Period (days)', fontsize=12, fontweight='bold')

plt.show()


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
#planent radious
fig, ax = plt.subplots(figsize=(10, 5))

# Set x and y labels and title
ax.set_xlabel('Planet Radius (Relative to Earths)', fontsize=12, fontweight='bold')
ax.set_ylabel('# of Planets', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Planet Radii', fontsize=14, fontweight='bold')

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

# boxplots - radius vs. method
psc['pl_rade'].groupby(psc['method2']).describe()
sns.boxplot(data=psc, x='method2', y='pl_rade')
plt.show()

# boxplots - radius vs. decade
psc['pl_rade'].groupby(psc['disc_decade']).describe()
sns.boxplot(data=psc, x='disc_decade', y='pl_rade')
plt.show()
psc[ (psc['disc_year']<=1999) & (psc['pl_rade']<5) ]  # these 3 planets are tiny, detected by 'pulsar timing' (very special/unique), and were the first-ever detections!

# add new col to categorize by radius
rad_bins = [0, 0.75,1.25,3,8,15,4000]
rad_labels = ["Sub-Terrestrial","Terrestrial","Super-Terrestrial","Sub-Giant", "Giant", "Super-Giant"]
psc['rad_cat'] = pd.cut(psc['pl_rade'].map(int), bins=rad_bins, labels=rad_labels, right=False, include_lowest=True)
print(psc.groupby('rad_cat').describe())
psc.groupby('rad_cat')['discoverymethod'].describe()
print(psc.groupby('discoverymethod')['rad_cat'].describe())
print(psc.groupby('disc_decade')['rad_cat'].describe())
print(psc.groupby('pl_orbloc')['rad_cat'].describe())
psc.groupby('rad_cat')['discoverymethod'].describe()
print(psc.groupby('rad_cat')['discoverymethod'].value_counts().unstack())
print(psc.groupby(['method2','rad_cat'])['rad_cat'].describe())
print(psc.groupby(['method2','rad_cat'])['disc_decade'].value_counts().unstack())


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


# boxplots - mass vs. method
fig, axes = plt.subplots(figsize=(10, 5))
sns.boxplot(data=psc, x='method2', y='pl_bmasse')
axes.set_yscale('log')
axes.set_xlabel('Discovery Method')
axes.set_ylabel('Planet Mass (relative to Earth)')
axes.set_title('Planet Mass vs. Discovery Method', fontweight='bold')
plt.show()

# boxplots - mass vs. decade
fig, axes = plt.subplots(figsize=(10, 5))
sns.boxplot(data=psc, x='disc_decade', y='pl_bmasse')
axes.set_yscale('log')
axes.set_xlabel('Discovery Decade')
axes.set_ylabel('Planet Mass (relative to Earth)')
axes.set_title('Planet Mass vs. Discovery Decade', fontweight='bold')
plt.show()


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
plt.show()

# add new col to categorize by mass
m_bins = [0, 0.75,1.25,3,8,15,4000]
m_labels = ["Sub-Terrestrial","Terrestrial","Super-Terrestrial","Sub-Giant", "Giant", "Super-Giant"]
psc['bmasse_cat'] = pd.cut(psc['pl_bmasse'].map(int), bins=m_bins, labels=m_labels, right=False, include_lowest=True)
print(psc.groupby('bmasse_cat').describe())
psc.groupby('bmasse_cat')['discoverymethod'].describe()
print(psc.groupby('discoverymethod')['bmasse_cat'].describe())
print(psc.groupby('disc_decade')['bmasse_cat'].describe())
print(psc.groupby('pl_orbloc')['bmasse_cat'].describe())
print(psc.groupby('bmasse_cat')['discoverymethod'].value_counts().unstack())
print(psc.groupby(['method2','bmasse_cat'])['bmasse_cat'].describe())
print(psc.groupby(['method2','bmasse_cat'])['disc_decade'].value_counts().unstack())
print(psc.groupby(['bmasse_cat','rad_cat'])['discoverymethod'].describe())
print(psc.groupby(['bmasse_cat','rad_cat'])['discoverymethod'].value_counts().unstack())
print(psc.groupby(['bmasse_cat'])['rad_cat'].value_counts().unstack())
print(psc.groupby(['bmasse_cat','rad_cat'])['pl_orbloc'].value_counts().unstack())




#%% Density

###############################################################################
# planents densities relive to earths
#interesting cut off at 6
fig, ax = plt.subplots(figsize=(10,5))

ax.set_xlabel("Planet Density (Relative to Earth's)", fontweight='bold')
ax.set_ylabel('# of Planets', fontweight='bold')
ax.set_title("Planet Densities", fontweight='bold')

ax.hist(psc.pl_dens, bins=200)
plt.show()
pass
###############################################################################


#%% System Distance
print(psc['sy_dist'].describe())
print('Null Distances: ', psc['sy_dist'].isnull().sum())
print('% of planets w/ Null Distances: ', psc['sy_dist'].isnull().sum()/psc['sy_dist'].size)

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
    title='Planets with Mass and Radius Similar to Earth within 50 Light Years (Between 80% and 120%)',
    legend=dict(x=0, y=1)
)
fig.show()


#%% Star Mass
print(psc['st_mass'].describe())
print('Null Mass: ', psc['st_mass'].isnull().sum())
print('% of planets w/ Null Mass: ', psc['st_mass'].isnull().sum()/psc['st_mass'].size)

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
print('% of planets w/ Null Spectral Type: ', psc['st_spectype'].isnull().sum()/psc['st_spectype'].size)


#%% Star Luminosity
print(psc['st_lum'].describe())
print('Null Luminosity: ', psc['st_lum'].isnull().sum())
print('% of planets w/ Null Luminosity: ', psc['st_lum'].isnull().sum()/psc['st_lum'].size)

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
print('% of planets w/ Null Effective Temp: ', psc['st_teff'].isnull().sum()/psc['st_teff'].size)


#%% Star Metallicity
print(psc['st_met'].describe())
print('Null Metallicity: ', psc['st_met'].isnull().sum())
print('% of planets w/ Null Metallicity: ', psc['st_met'].isnull().sum()/psc['st_met'].size)


#%% Star Age
print(psc['st_age'].describe())
print('Null Age: ', psc['st_age'].isnull().sum())
print('% of planets w/ Null Age: ', psc['st_age'].isnull().sum()/psc['st_age'].size)

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

# barchart to show the frequency of the # of planets per star
psc['hostname'].value_counts().describe()
psc['hostname'].value_counts().value_counts()

# plot the frequency of planets in a system
ax = psc['hostname'].value_counts().value_counts().plot(kind='bar')
ax.set_yscale('log')
plt.show()

# plot to compare the 10 most-populated systems w/ Sol system
systems_1 = psc.groupby('hostname').filter(lambda row: (row.hostname.size > 5))
systems_1_hosts = systems_1['hostname'].unique()
sns.relplot(data=systems_1, x='pl_orbr_est', y='hostname', size='pl_bmasse')
plt.show()

# plot to compare the systems w/ most Earth-like planets w/ Sol system
systems_2_hosts = earth_like_planets['hostname'].unique()
systems_2 = psc [ psc['hostname'].isin(systems_2_hosts)]
sns.relplot(data=systems_2, x='pl_orbr_est', y='hostname', size='pl_bmasse')
plt.show()


#%% Planetary Spectroscopy
# add new col to summarize spectroscopic measurements
psc['pl_spec'] = psc['pl_nespec']+psc['pl_ntranspec']
print(psc['pl_spec'].describe())
psc_spec = psc[psc['pl_spec']>0]
print('Planets w/ spectroscopy Measurements: ', psc_spec['pl_spec'].size)

# histogram of the number of spectroscopy measurements per planet
plt.hist(psc_spec['pl_spec'], bins=20)
plt.show()

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


