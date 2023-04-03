import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests

# Query data and store as a dataframe -- \
    # ref: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
columns = ['pl_name', 'discoverymethod', 'disc_year', 'pl_controv_flag', \
    'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_dens', 'hostname', 'sy_dist', \
    'st_mass', 'st_spectype', 'st_lum', 'st_teff', 'st_met', 'st_age']
url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query='+'select+'\
    +','.join(columns)+'+from+pscomppars'+'&format=json'
data = requests.get(url).text
psc = pd.read_json(data)

# Data transformations
    # set index as planet name
    # filter data:  remove outliers? exclude controversials?  exclude incompletes or nulls?
        # document transformations and assumptions

# Preliminary EDA
print(psc)
print(psc.index)
print(psc.columns)
print(psc.describe)

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

#Interactive bar chart of exoplanet discovery & discovery methods over time. Instructons found at https://plotly.com/python/bar-charts/

method_year_counts = psc.groupby(['discoverymethod', 'disc_year']).size().unstack().transpose()
# Sort the columns of method_year_counts by their values
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



