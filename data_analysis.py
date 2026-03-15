import pandas as pd
import numpy as np
from scipy import stats
import os

# File path (use raw string for Windows)
file_path = r'D:\New worksheet\Final worksheet.xlsx'

# Step 1: Load and inspect
print("Loading data...")
df = pd.read_excel(file_path)
print("Shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

# Step 2: Clean column names FIRST (before using them)
df.columns = df.columns.str.strip()
# Fix the regex issue - use raw string prefix
df.columns = df.columns.str.replace(r'[%\[\]]', '', regex=True)  
df.columns = df.columns.str.replace('[(%)]', '', regex=True)  # Remove parentheses too
df.columns = [col.replace('  ', ' ').strip() for col in df.columns]

print("\nCleaned columns:")
print(df.columns.tolist())

# Step 3: Rename for consistency (use EXACT cleaned names)
df = df.rename(columns={
    'Sl No': 'Sl_No',
    '12th Examination Name': '12th_Name',
    '12th Examination Board/Council Name': '12th_Board',
    '12th Mark of Overall Percentage all subjects': '12th_Pct',
    '10th Mark of Overall Percentage All subjects': '10th_Pct',
    '10th Board/Council Name': '10th_Board',
    '10th Medium of Studies': '10th_Medium'
})

print("\nRenamed columns:")
print(df.columns.tolist())

# 🔥 NEW: REPLACE ALL ZEROS WITH NaN (ADD THESE 3 LINES) 🔥
print("🧹 Replacing ALL zeros with NaN across entire dataset...")
zero_count_before = (df == 0).sum().sum()
df = df.replace(0, np.nan)  # MAGIC LINE - ALL 0s → NaN everywhere!
print(f"✅ Replaced {zero_count_before} zeros with NaN")


# Step 4: Convert percentages to numeric
pct_cols = ['12th_Pct', '10th_Pct']
for col in pct_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
        print(f"Converted {col} to numeric")

# Step 5: Mark columns (handle missing ones safely)
mark_cols = ['Total Lang1', 'Total Lang2', 'Total Math', 'Total PHY', 
             'Total CHE', 'Total BIO/other', 'Total CS/IT']
mark_cols = [col for col in mark_cols if col in df.columns]

# Fill missing with median
for col in mark_cols + pct_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {col} with median: {median_val:.1f}")

# Categorical columns (fill with mode)
cat_cols = ['12th_Name', '12th_Board', '10th_Board', '10th_Medium']
for col in cat_cols:
    if col in df.columns and df[col].notna().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Filled {col} with mode: {mode_val}")

# Step 6: Cap outliers (0-100)
for col in mark_cols + pct_cols:
    if col in df.columns:
        df[col] = df[col].clip(0, 100)

# Step 7: Feature engineering
stem_cols = ['Total Math', 'Total PHY', 'Total CHE', 'Total CS/IT']
stem_cols = [col for col in stem_cols if col in df.columns]
if len(stem_cols) > 0:
    df['STEM_Total'] = df[stem_cols].sum(axis=1, skipna=True)
    df['STEM_Avg'] = df['STEM_Total'] / len(stem_cols)
else:
    df['STEM_Avg'] = df['12th_Pct']

df['Overall_Avg'] = df[['12th_Pct', '10th_Pct']].mean(axis=1, skipna=True)
df['Aptitude'] = pd.cut(df['STEM_Avg'], bins=[0, 60, 80, 100], 
                       labels=['Low', 'Medium', 'High'])

print("\nNew features created: STEM_Avg, Overall_Avg, Aptitude")
print("\nFinal shape:", df.shape)
print("\nSample data:")
print(df[['Sl_No', 'STEM_Avg', 'Overall_Avg', 'Aptitude', '12th_Pct']].head())

# Save cleaned data
output_path = r'D:\New worksheet\cleaned_career_data.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")
