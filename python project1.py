import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df=pd.read_csv(r"C:\Users\mohit\Downloads\Crash_Reporting_-_Drivers_Data.csv")



# Handle missing values
df.fillna({
    "route_type": "Unknown",
    "road_name": "Unknown",
    "collision_type": "Unknown",
    "weather": "Unknown",
    "surface_condition": "Unknown",
    "light": "Unknown",
    "traffic_control": "Unknown",
    "driver_substance_abuse": "None",
    "injury_severity": "Unknown",
    "vehicle_make": "Unknown",
    "vehicle_model": "Unknown",
}, inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)


# Standardize column names (convert to lowercase and replace spaces with underscores)
df.columns = df.columns.str.lower().str.replace(" ", "_")


# Identify numerical columns
numeric_columns = ["speed_limit", "vehicle_year", "latitude", "longitude"]


# Fill missing numerical values with median
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())


# Display dataset information and first few rows
print(df.info())
print(df.head())

#-----------------------------------------------------------------------------------------------------------------------------------------------------

# Exploratory Data Analysis (EDA)
print("Summary Statistics:")
print(df.describe())

#----------------------------------------------------------------------------------------------------------------------------------------------------

# Accident Severity Distribution
df["injury_severity"] = df["injury_severity"].str.upper()
severity_counts = df["injury_severity"].value_counts()
print("Accident Severity Distribution:")
print(severity_counts)

#----------------------------------------------------------------------------------------------------------------------------------------------------

# Speed Limit vs. Accident Severity
severity_speed = df.groupby("injury_severity")["speed_limit"].agg(["mean", "median"])
print("\nSpeed Limit Analysis: Speed Limit vs. Accident Severity")
print(severity_speed)

#---------------------------------------------------------------------------------------------------------------------------------------------------

# Driver Substance Abuse Analysis
df["driver_substance_abuse"] = df["driver_substance_abuse"].str.upper()
df["driver_substance_abuse"] = df["driver_substance_abuse"].replace({
    "UNKNOWN, UNKNOWN": "UNKNOWN",
    
    "UNKNOWN, SUSPECT OF DRUG USE": "UNKNOWN"
})
substance_counts = df["driver_substance_abuse"].value_counts()
print("\nSubstance Abuse and Accidents:")
print(substance_counts)

#-------------------------------------------------------------------------------------- -----------------------------------------------------

# Weather Condition & Accidents
df["weather"] = df["weather"].str.upper()
weather_counts = df["weather"].value_counts()
print("\nAccidents by Weather Conditions:")
print(weather_counts)
#-------------------------------------------------------------------------------------- -----------------------------------------------------

# Driver Distraction Impact
df["driver_distracted_by"] = df["driver_distracted_by"].str.upper()
distraction_counts = df["driver_distracted_by"].value_counts()
print("\nDriver Distraction Analysis:")
print(distraction_counts)

#----------------------------------------------------------------------------------------------------------------------------

# Visualize Injury Severity Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="injury_severity", order=severity_counts.index, palette="viridis")
plt.title("Distribution of Injury Severity")
plt.xlabel("Injury Severity")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------------------------

# Speed Limit vs Severity Boxplot

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="injury_severity", y="speed_limit", palette="coolwarm")
plt.title("Speed Limit Distribution by Injury Severity")
plt.xticks(rotation=45)
plt.show()

#------------------------------------------------------------------------------------------------------------


# Pie chart of ACRS Report Type
type_counts = df['acrs_report_type'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Crash Type Distribution')
plt.axis('equal')
plt.show()


#------------------------------------------------------------------------------------------------------------

# Histplot for Speed Limit Distribution
df['speed_limit'] = pd.to_numeric(df['speed_limit'], errors='coerce')

plt.figure(figsize=(10, 6))
sns.histplot(df['speed_limit'].dropna(), bins=15, kde=True)
plt.title('Speed Limit Distribution in Crashes')
plt.xlabel('Speed Limit (mph)')
plt.ylabel('Count')
plt.show()

#------------------------------------------------------------------------------------------------------------


#lineplot for yearlt crashes
df['crash_date/time'] = pd.to_datetime(df['crash_date/time'], errors='coerce')
yearly_crashes = df['crash_date/time'].dt.to_period('M').value_counts().sort_index()

# Plot
plt.figure(figsize=(12, 6))
yearly_crashes.plot(kind='line', marker='o', color='teal')
plt.title('Number of Crashes Per year')
plt.xlabel('Year')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------------------------

# Map Visualization of Crashes for each type of injury
import folium
import webbrowser

df["injury_severity"] = df["injury_severity"].str.strip().str.lower()

# Map injury severity to colors
injury_colors = {
    "fatal injury": "green",  # Fatal injury will be green
    "no apparent injury": "blue",  # No apparent injury will be blue
    "possible injury": "Purple",  # Possible injury will be orange
    "suspected minor injury": "black",  # Suspected minor injury will be yellow
    "suspected serious injury": "red"  # Suspected serious injury will be red
}

# Assign colors based on injury severity
df["color"] = df["injury_severity"].map(injury_colors).fillna("blue")  # Default to blue for unknown types

# Center the map using median location
map_center = [df["latitude"].median(), df["longitude"].median()]
m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

# Add markers with the appropriate colors
for _, row in df.sample(100).iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color=row["color"],  # Use the 'color' column for the color of the circle
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

m.save("crash_map.html")
webbrowser.open("crash_map.html")

#----------------------------------------------------------------------------------------------------------------

#Countplot for Crash Count by top 5  Weather Conditionin which most accident happened
df.columns = df.columns.str.upper()

df["WEATHER"] = df["WEATHER"].str.strip().str.title()

top_5_weather = df["WEATHER"].value_counts().nlargest(5).index


plt.figure(figsize=(10, 5))
sns.countplot(data=df[df["WEATHER"].isin(top_5_weather)],
              x="WEATHER",
              order=top_5_weather,
              palette="inferno")
plt.title("Top 5 Weather Conditions in Crashes")
plt.xticks(rotation=45)
plt.xlabel("Weather")
plt.ylabel("Crash Count")
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------


#Crash Count by top 4 Light Conditions in which most crash occured
df["LIGHT"] = df["LIGHT"].str.strip().str.title()

# Fix known issues like 'DAylight'
df["LIGHT"] = df["LIGHT"].replace({
    "Aylight": "Daylight",
    "Dark - Lighted": "Dark - Lighted",
    "Dark - Not Lighted": "Dark - Not Lighted",
    "Dark - Unknown Lighting": "Dark - Unknown",
    "Dark -- Unknown Lighting": "Dark - Unknown",
    "Dark Lights On": "Dark - Lighted",
    "Dark No Lights": "Dark - Not Lighted"
})


df["LIGHT"] = df["LIGHT"].str.strip().str.title()

top_4_lights = df["LIGHT"].value_counts().nlargest(4).index

plt.figure(figsize=(8, 4))
sns.countplot(data=df[df["LIGHT"].isin(top_4_lights)],
              x="LIGHT",
              order=top_4_lights,
              palette="rocket")
plt.title("Top 4 Lighting Conditions in Crashes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(df.columns)


#-----------------------------------------------------------------------------------------------------------
#countplot for top 7 route types where max  crashes occurred
top7 = df["ROUTE_TYPE"].value_counts().nlargest(7).index

plt.figure(figsize=(8, 5))
sns.countplot(data=df[df["ROUTE_TYPE"].isin(top7)], 
              x="ROUTE_TYPE", 
              order=top7, 
              palette="Spectral")
plt.title("Crashes by Route Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------------------------------