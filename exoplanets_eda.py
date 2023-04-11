import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
max_distance = 15.5  # light-years

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