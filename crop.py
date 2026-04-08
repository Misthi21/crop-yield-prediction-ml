import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Styling
sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.facecolor": "#fdfefe",
    "axes.facecolor": "#fdfefe",
    "axes.edgecolor": "#dddddd"
})

# Load dataset
df = pd.read_csv(r"C:\Users\MISTHI\Downloads\misthikatoch65_17754861315134325.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns
df = df.rename(columns={
    "Yield (UOM:Kg/Ha(KilogramperHectare)), Scaling Factor:1": "yield(hec)",
    "Area (UOM:Ha(Hectare)), Scaling Factor:1000": "Area(hec)",
    "Crop Name": "Crop"
})

# Drop unnecessary columns
df = df.drop(columns=[
    "Country",
    "Additional Info",
    "Area Percentage To All India (%) (UOM:%(Percentage)), Scaling Factor:1"
], errors='ignore')

# Fix Year column
df["Year"] = df["Year"].astype(str).str.extract(r'(\d{4})')
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

# Remove missing values
df = df.dropna()

# Remove outliers
df = df[df["yield(hec)"] < df["yield(hec)"].quantile(0.97)]

# Encode categorical variables
le = LabelEncoder()
df["State"] = le.fit_transform(df["State"])
df["Crop"] = le.fit_transform(df["Crop"])

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="flare")
plt.title("Feature Correlation Heatmap", weight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# Features & target
X = df[["State", "Crop", "Year", "Area(hec)"]]
y = df["yield(hec)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluation
print("Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("R2:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("R2:", r2_score(y_test, y_pred_rf))


# -----------------------------
# GRAPH 1: Scatter
# -----------------------------
plt.figure(figsize=(6,4))

plt.scatter(y_test, y_pred_rf, color="#ff8fab", edgecolor="white", s=70)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='grey', linestyle='--')

plt.title("Actual vs Predicted", weight="bold")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.text(
    0.02, 0.98,
    "• Good fit near line\n• Shows prediction quality\n• Some deviation\n• RF performs well",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.85)
)

plt.grid(True, linestyle='--', color='grey', alpha=0.6)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


# -----------------------------
# GRAPH 2: Bar
# -----------------------------
plt.figure(figsize=(8,4))

top_crops = df.groupby("Crop")["yield(hec)"].mean().sort_values(ascending=False).head(10)

sns.barplot(
    x=top_crops.index,
    y=top_crops.values,
    hue=top_crops.index,
    palette="pastel",
    legend=False
)

plt.title("Top Crops", weight="bold")
plt.xlabel("Crop")
plt.ylabel("Yield")

plt.text(
    0.65, 0.85,
    "• Crop comparison\n• Top performers\n• Useful insight\n• Clear variation",
    transform=plt.gca().transAxes,
    fontsize=9,
    bbox=dict(facecolor='white', alpha=0.85)
)

plt.xticks(rotation=40)
plt.grid(axis='y', linestyle='--', color='grey', alpha=0.6)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


# -----------------------------
# GRAPH 3: Line
# -----------------------------
plt.figure(figsize=(7,4))

yearly = df.groupby("Year")["yield(hec)"].mean().reset_index()

sns.lineplot(x="Year", y="yield(hec)", data=yearly,
             marker="o", linewidth=3, color="#a2d2ff")

peak = yearly.loc[yearly["yield(hec)"].idxmax()]
plt.scatter(peak["Year"], peak["yield(hec)"], color="red", s=80)

plt.title("Yearly Trend", weight="bold")
plt.xlabel("Year")
plt.ylabel("Yield")

plt.text(
    0.02, 0.95,
    "• Trend over time\n• Peak highlighted\n• Fluctuations visible\n• Useful insight",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.85)
)

plt.grid(True, linestyle='--', color='grey', alpha=0.6)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


# -----------------------------
# GRAPH 4: Pie
# -----------------------------
plt.figure(figsize=(6,6))

dist = df["Crop"].value_counts().head(5)

plt.pie(
    dist,
    labels=dist.index,
    autopct='%1.1f%%',
    colors=["#ffafcc","#cdb4db","#a2d2ff","#ffc8dd","#bde0fe"],
    startangle=140,
    wedgeprops={'edgecolor': 'white'}
)

plt.title("Crop Distribution", weight="bold")

plt.text(
    0, -1.3,
    "• Shows proportions\n• Top crops dominate\n• Easy comparison\n• Diversity insight",
    ha='center',
    fontsize=9,
    bbox=dict(facecolor='white', alpha=0.85)
)

plt.subplots_adjust(top=0.88, bottom=0.15)
plt.show()