# EXP 3 - Delhi Air Quality Analysis

### Name:R Anirudh
### Reg No.: 212223230016

**Aim**


To compare air quality parameters in Delhi across different stations and analyze the relationship between pollutants (e.g., PM2.5 and NO₂) using scatter plots and correlation analysis.


**Procedure / Algorithm**

1)Load the dataset using pandas.

2)Preprocess the data:

3)Convert the date column (period.datetimeFrom.utc) to datetime format.

4)Drop missing or invalid values.

5)Pivot the dataset so each pollutant (parameter) becomes a separate column.

6)Plot scatter plot between PM2.5 and NO₂ to study their relationship.

7)Plot correlation heatmap between all pollutants to identify relationships.

8)Interpret the results — identify which pollutants are correlated and which stations are most polluted.


**Program**

# Q1. Load the dataset and perform preprocessing
```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("del-sirifort-cpcb-2024-25.csv")

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# Select required columns
columns = [
    'PM2.5 (µg/m³)',
    'PM10 (µg/m³)',
    'NO2 (µg/m³)',
    'Ozone (µg/m³)'
]

data = df[columns].copy()

# Convert selected columns into numeric format
for col in columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Check missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Handle missing values separately for each variable
clean_data = data.copy()

print("\nAvailable Records After Preprocessing:")
for col in columns:
    print(col, ":", clean_data[col].count())

print("\nDataset Information:")
print(clean_data.info())

print("\nPreprocessed Dataset:")
print(clean_data.head())
```
# Q2. Distribution analysis using histograms
```python
plt.figure(figsize=(14, 10))

for i, col in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    sns.histplot(
        clean_data[col].dropna(),
        bins=30,
        kde=True
    )

    plt.title("Distribution of " + col)
    plt.xlabel("Concentration (µg/m³)")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
```
# Q3. Calculate numerical summary
```python
summary = pd.DataFrame(index=columns)

for col in columns:

    x = clean_data[col].dropna()

    summary.loc[col, 'Mean'] = x.mean()

    summary.loc[col, 'Median'] = x.median()

    summary.loc[col, 'Mode'] = x.mode().iloc[0]

    summary.loc[col, 'Minimum'] = x.min()

    summary.loc[col, 'Q1'] = x.quantile(0.25)

    summary.loc[col, 'Q3'] = x.quantile(0.75)

    summary.loc[col, 'Maximum'] = x.max()

print("NUMERICAL SUMMARY")
print(summary.round(2))
```
# Q4. Calculate measures of spread
```python
spread = pd.DataFrame(index=columns)

for col in columns:

    x = clean_data[col].dropna()

    mean = x.mean()

    # Range
    spread.loc[col, 'Range'] = x.max() - x.min()

    # Variance
    spread.loc[col, 'Variance'] = x.var()

    # Standard Deviation
    spread.loc[col, 'Standard Deviation'] = x.std()

    # Coefficient of Variation
    spread.loc[col, 'Coefficient of Variation (%)'] = (
        (x.std() / mean) * 100 if mean != 0 else np.nan
    )

print("MEASURES OF SPREAD")
print(spread.round(2))
```
# Q5. Construct box plots and detect outliers using IQR
```python
#Box plot visualization
plt.figure(figsize=(12, 7))

sns.boxplot(data=clean_data)

plt.title("Box Plot of Delhi Air Quality Pollutants")
plt.xlabel("Air Pollutants")
plt.ylabel("Concentration (µg/m³)")

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```
#Outlier detection using IQR
```python
for col in columns:

    x = clean_data[col].dropna()

    # Calculate quartiles
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = x[
        (x < lower_limit) | (x > upper_limit)
    ]

    print("\nPollutant:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Number of Outliers:", len(outliers))

    print("First 10 Outliers:")
    print(outliers.head(10).values)
```
# Q6. Perform Min-Max scaling
```python
# Create an empty DataFrame
minmax_data = pd.DataFrame(index=clean_data.index)

for col in columns:

    x = clean_data[col]

    min_value = x.min()
    max_value = x.max()

    # Apply Min-Max scaling
    if max_value != min_value:
        minmax_data[col] = (
            (x - min_value) / (max_value - min_value)
        )
    else:
        minmax_data[col] = 0.0

print("MIN-MAX SCALED DATA")
print(minmax_data.head(10))

# Display minimum and maximum after scaling
print("\nScaled Data Range:")
print(minmax_data.agg(['min', 'max']).round(4))

# Save complete scaled dataset
minmax_data.to_csv("minmax_scaled_data.csv", index=False)
```
# Q7. Perform Z-score standardization using StandardScaler
```python
# Create an empty DataFrame
standardized_data = pd.DataFrame(index=clean_data.index)

# Apply StandardScaler separately to each pollutant
for col in columns:

    valid = clean_data[col].notna()

    scaler = StandardScaler()

    standardized_data.loc[valid, col] = (
        scaler.fit_transform(
            clean_data.loc[valid, [col]]
        ).flatten()
    )

print("Z-SCORE STANDARDIZED DATA")
print(standardized_data.head(10))

# Verify mean and standard deviation
print("\nMean After Standardization:")
print(standardized_data.mean().round(4))

print("\nStandard Deviation After Standardization:")
print(standardized_data.std(ddof=0).round(4))

# Save complete standardized dataset
standardized_data.to_csv(
    "zscore_standardized_data.csv",
    index=False
)
```
# Q8. Calculate Gini coefficient
```python
# Function to calculate Gini coefficient
def gini_coefficient(values):

    # Convert values into NumPy array
    x = np.array(values, dtype=float)

    # Remove missing values
    x = x[~np.isnan(x)]

    # Gini requires non-negative values
    if np.any(x < 0):
        raise ValueError("Gini coefficient requires non-negative values")

    # Handle empty or zero-valued data
    if len(x) == 0:
        return np.nan

    if np.sum(x) == 0:
        return 0

    # Sort values in ascending order
    x = np.sort(x)

    n = len(x)

    # Calculate Gini coefficient
    index = np.arange(1, n + 1)

    gini = (
        2 * np.sum(index * x) / (n * np.sum(x))
    ) - ((n + 1) / n)

    return gini


# Calculate Gini for all four pollutants
print("GINI COEFFICIENT OF AIR POLLUTANTS")

for col in columns:

    gini = gini_coefficient(clean_data[col])

    print(col, ":", round(gini, 4))
```
```python
# Save numerical summary
summary.to_csv("numerical_summary.csv")

# Save measures of spread
spread.to_csv("measures_of_spread.csv")

print("All analysis results saved successfully!")
```
**Output**
# Q1. Load the dataset and perform preprocessing

<img width="872" height="936" alt="image" src="https://github.com/user-attachments/assets/a392e64a-bdbd-4bdf-99c1-14c340c41be6" />

# Q2. Distribution analysis using histograms

<img width="1195" height="858" alt="image" src="https://github.com/user-attachments/assets/ad0c2f2e-3015-4587-be4e-20a2b9f6e496" />

# Q3. Calculate numerical summary

<img width="707" height="172" alt="image" src="https://github.com/user-attachments/assets/5ebdbd60-8be2-4f08-a650-bdbfbc753e4b" />

# Q4. Calculate measures of spread

<img width="565" height="292" alt="image" src="https://github.com/user-attachments/assets/9be0c05c-2396-4a78-a03c-fbc7edda66e9" />

# Q5. Construct box plots and detect outliers using IQR

<img width="1122" height="648" alt="image" src="https://github.com/user-attachments/assets/a5044283-2952-4dc7-b22b-ca35537ff160" />

<img width="672" height="706" alt="image" src="https://github.com/user-attachments/assets/354fa4ef-01bf-4383-a7ce-956454e98d5e" />

# Q6. Perform Min-Max scaling

<img width="537" height="308" alt="image" src="https://github.com/user-attachments/assets/fcd17d50-aa03-42ee-b3a1-90441b27b219" />

# Q7. Perform Z-score standardization using StandardScaler

<img width="752" height="666" alt="image" src="https://github.com/user-attachments/assets/727dcffe-93ea-4987-b4e1-eb34bbfaff0e" />

# Q8. Calculate Gini coefficient

<img width="417" height="148" alt="image" src="https://github.com/user-attachments/assets/d104540f-ab8a-498a-9545-ed90991a24f4" />

<img width="568" height="263" alt="image" src="https://github.com/user-attachments/assets/cea4a1ac-99c3-45b3-b83e-26f226c6fe4c" />

**Interpretation**

1)PM2.5 and NO₂ show a strong positive correlation, suggesting that both pollutants increase together, likely due to vehicle and industrial emissions.


**Result**

The dataset was successfully loaded and processed to extract pollutant-wise and station-wise air quality data for Delhi.


