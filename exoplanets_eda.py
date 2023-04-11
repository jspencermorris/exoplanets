import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import requests

filename = 'PSCompPars_2023.04.04_17.16.52.csv'
psc = pd.read_csv(filename, comment='#')

# Data transformations
    # set index as planet name
    # filter data:  remove outliers? exclude controversials?  exclude incompletes or nulls?
        # document transformations and assumptions


# Preliminary EDA
print(psc)
print(psc.index)
print(psc.columns)
print(psc.describe)

# Check for missing values
print("\nMissing values count:")
print(psc.isnull().sum())

#Check for duplciated rows
print("Number of duplicated rows:", psc.duplicated().sum())

#Filter out all columns except for the following:
cols_to_keep = ['pl_name', 'discoverymethod', 'disc_year', 'pl_controv_flag', \
    'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_dens', 'hostname', 'sy_dist', \
    'st_mass', 'st_spectype', 'st_lum', 'st_teff', 'st_met', 'st_age']

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
#   exclude objects with orbiatal periods more than 1000 years, \
#       which is well beyond the period of the most-distant dwarf planets in the Sol system
#   exclude objects with radii more than 30 times the radius of Earth, as they are brown dwarfs
#   exclude objects with masses more than 3200 times the mass of Earth, as they are by definition brown dwarfs
#   exclude objects with densities greater than lead (11.35 g/cm3)
#   exclude objects orbiting stars with masses more than the theoretical accretion limit (120 solar masses)
#   exclude objects orbiging stars older than the universe (13.8 Gy)
psc.describe()
psc.drop( psc[ psc['pl_orbper'] > 1000].index, inplace=True)
psc.drop( psc[ psc['pl_rade'] > 30].index, inplace=True)
psc.drop( psc[ psc['pl_bmasse'] > 3200 ].index, inplace=True)
psc.drop( psc[ psc['pl_dens'] > 11.35 ].index, inplace=True)
psc.drop( psc[ psc['st_mass'] > 120 ].index, inplace=True)
psc.drop( psc[ psc['st_age'] > 13.8 ].index, inplace=True)

# remove controversial objects
psc.drop( psc[ psc['pl_controv_flag'] == 1 ].index, inplace=True)

#   it looks like the spectral type is mostly null, so many consider dropping that column
#   consider dropping any rows w/ nan's in this column:  pl_orbper

# reset row indices
psc.reset_index(drop=True, inplace=True)

# Hisgoram of planet radii (relative to Earth's radius)
print(psc['pl_rade'].describe())
psc['pl_rade'].plot.hist(bins=100)
plt.title('distribution of Planet Radii')
plt.xlabel('Planet Radius (relative to Earth)')
plt.ylabel('Counts')
plt.show()

# Density plot of planet radii (relative to Earth's radius)
sns.distplot(psc['pl_rade'], bins=100, color='k')
plt.title('density of Planet Radii')
plt.xlabel('Planet Radius (relative to Earth)')
plt.ylabel('Counts')
plt.show()

# Scatterplot of planet radius vs. mass
# NOTE - there may be an outlier (convert from loglog back to linear to see)
print(psc[['pl_rade','pl_bmasse']].describe())
psc.plot.scatter(x='pl_bmasse', y='pl_rade', \
    xlabel='Est. Planet Mass (relative to Earth)', \
    ylabel='Planet Radius (relative to Earth)', \
    loglog=True)
plt.title('Relationship of Planet Estimated Mass and Radius ')
plt.show()

# Boxplots based on discoverymethod
box_vars = ['disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'sy_dist', 'st_mass', 'st_lum', \
    'st_teff', 'st_met']
fig, axes = plt.subplots(nrows=3, ncols=3, sharey='row')
k = 0
for i in range(3):
    for j in range(3):
        a = sns.boxplot(x=box_vars[k], y='discoverymethod', data=psc, orient='h', ax=axes[i,j])
        a.set(ylabel=None)
        k += 1
fig.subplots_adjust(hspace = 0.3, wspace=0)
fig.suptitle('Boxplots of key numerical variables across different Discovery Methods')
plt.show()

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

# Hisgoram of planet radii (relative to Earth's radius)
print(psc['pl_rade'].describe())
psc['pl_rade'].hist(bins=100)
plt.title('distribution of Planet Radii')
plt.xlabel('Planet Radius (relative to Earth)')
plt.ylabel('Counts')
plt.show()

# Scatterplot of planet radius vs. mass
# NOTE - there may be an outlier (convert from loglog back to linear to see)
print(psc[['pl_rade','pl_bmasse']].describe())
psc.plot.scatter(x='pl_bmasse', y='pl_rade', \
    xlabel='Est. Planet Mass (relative to Earth)', \
    ylabel='Planet Radius (relative to Earth)', \
    loglog=True)
plt.title('Relationship of Planet Estimated Mass and Radius ')
plt.show()

# Scatterplot of star's age vs. metallicity
print('METALICITY\n',psc['st_met'].describe)

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

# Interactive scatterplot of exoplanets within fifty lightyears with radius and mass between 80% and 120% of earth.
#Documentation found at https://plotly.com/python/line-and-scatter/
earth_mass_range = (0.8, 1.2)  # 80% to 120% of Earth's mass
earth_radius_range = (0.8, 1.2)  # 80% to 120% of Earth's radius
max_distance = 15.3301  # parsecs

earth_like_planets = psc[    (psc['pl_bmasse'].between(earth_mass_range[0], earth_mass_range[1])) &
    (psc['pl_rade'].between(earth_radius_range[0], earth_radius_range[1])) &
    (psc['sy_dist'] <= max_distance)
]
marker_size_factor = 10
marker_sizes = earth_like_planets['pl_rade'] * marker_size_factor

fig = go.Figure(data=go.Scatter(
    x=earth_like_planets['pl_bmasse'],
    y=earth_like_planets['pl_rade'],
    mode='markers',
    marker=dict(
        size=marker_sizes,
        color=earth_like_planets['sy_dist'],
        colorscale='viridis_r',
        cmin=earth_like_planets['sy_dist'].min(),
        cmax=earth_like_planets['sy_dist'].max(),
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
    title='Planets with Mass and Radius Similar to Earth within 50 Light Years',
    legend=dict(x=0, y=1)
)
fig.show()