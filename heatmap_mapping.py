import folium
from folium.plugins import HeatMap
import pandas as pd

# Load GPS data from a CSV file
gps_data = pd.read_csv('latitude_longitude.csv')

# Extract coordinates
heatmap_data = gps_data[['latitude', 'longitude']].values.tolist()

# Create a Folium map
m = folium.Map(location=[20.0, 78.0], zoom_start=5)

# Add heatmap
HeatMap(heatmap_data).add_to(m)

# Save the map as an HTML file
m.save('plastic_waste_heatmap.html')
