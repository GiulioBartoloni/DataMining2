check the data and the code extensively, 

here are the columns that i considered to be the best to keep, but i may have dropped too much, consider missing data, importance etc to maybe add something more

print(df_clean.columns)

cols = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-Height', 'Physical-Weight', 'FGC_FitnessScore', 'BIA-BIA_FFMI', 'BIA-BIA_FFM', 'BIA-BIA_BMI',  'PreInt_EduHx-computerinternet_hoursday', 'sii',]
print(len(cols))
Show more
Read module_0.ipynb
Read MEMORY.md
Read module_0.ipynb
Globpattern: "CMI/**/*"
Found 7 files
Read data_dictionary.csv, column_bounds.md
Now let me check the actual data to understand missing rates and available columns.

Bash
IN
python3 -c "
import pandas as pd
df = pd.read_csv('CMI/processed_data/cmi_preprocessed.csv')
print('Shape:', df.shape)
print()
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
summary = pd.DataFrame({'missing': missing, 'missing_pct': missing_pct}).sort_values('missing_pct')
print(summary.to_string())
"

OUT
Shape: (8243, 27)

                                        missing  missing_pct
Basic_Demos-Age                               0          0.0
Basic_Demos-Sex                               0          0.0
CGAS-CGAS_Score                               0          0.0
Physical-BMI                                  0          0.0
Physical-Height                               0          0.0
Physical-Weight                               0          0.0
Physical-Waist_Circumference                  0          0.0
Physical-Diastolic_BP                         0          0.0
Physical-HeartRate                            0          0.0
Physical-Systolic_BP                          0          0.0
Fitness_Endurance-Max_Stage                   0          0.0
FGC-FGC_CU                                    0          0.0
FGC-FGC_GSND                                  0          0.0
FGC-FGC_GSD                                   0          0.0
FGC-FGC_PU                                    0          0.0
FGC-FGC_SRL_Zone                              0          0.0
FGC-FGC_SRR_Zone                              0          0.0
FGC-FGC_TL_Zone                               0          0.0
BIA-BIA_BMC                                   0          0.0
BIA-BIA_FFM                                   0          0.0
BIA-BIA_FFMI                                  0          0.0
BIA-BIA_FMI                                   0          0.0
BIA-BIA_Fat                                   0          0.0
BIA-BIA_Frame_num                             0          0.0
SDS-SDS_Total_T                               0          0.0
PreInt_EduHx-computerinternet_hoursday        0          0.0
sii                                           0          0.0
Bash
IN
python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('CMI/processed_data/cmi_preprocessed.csv')
print('All columns:', list(df.columns))
print()
# Check if FGC_FitnessScore exists
print('FGC_FitnessScore in df:', 'FGC_FitnessScore' in df.columns)
print()
# Check correlation with sii
corr = df.select_dtypes(include=[np.number]).corr()['sii'].sort_values(key=abs, ascending=False)
print('Correlations with sii:')
print(corr.to_string())
"

OUT
All columns: ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP', 'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone', 'BIA-BIA_BMC', 'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday', 'sii']

FGC_FitnessScore in df: False

Correlations with sii:
sii                                       1.000000
BIA-BIA_FFM                               0.230210
Physical-Height                           0.229336
Physical-Weight                           0.221803
Basic_Demos-Age                           0.212355
PreInt_EduHx-computerinternet_hoursday    0.207190
FGC-FGC_CU                                0.182489
BIA-BIA_Fat                               0.175783
SDS-SDS_Total_T                           0.165060
BIA-BIA_FFMI                              0.154920
Physical-Waist_Circumference              0.146654
BIA-BIA_FMI                               0.136105
FGC-FGC_PU                                0.135758
Physical-BMI                              0.120665
BIA-BIA_BMC                               0.112674
Physical-Systolic_BP                      0.109404
BIA-BIA_Frame_num                         0.086164
FGC-FGC_GSD                               0.084268
Basic_Demos-Sex                          -0.083843
FGC-FGC_GSND                              0.080276
FGC-FGC_SRL_Zone                         -0.077698
FGC-FGC_SRR_Zone                         -0.065370
FGC-FGC_TL_Zone                          -0.054205
Physical-Diastolic_BP                     0.045582
Physical-HeartRate                       -0.034734
CGAS-CGAS_Score                          -0.033418
Fitness_Endurance-Max_Stage              -0.010610
Bash
IN
python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('CMI/processed_data/cmi_preprocessed.csv')

# Check multicollinearity among selected numeric features
selected = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 
            'Physical-Height', 'Physical-Weight', 'BIA-BIA_FFMI', 'BIA-BIA_FFM',
            'PreInt_EduHx-computerinternet_hoursday']

# Also look at what's NOT selected but has decent correlation
candidates = ['FGC-FGC_CU', 'BIA-BIA_Fat', 'SDS-SDS_Total_T', 
              'Physical-Waist_Circumference', 'BIA-BIA_FMI', 'FGC-FGC_PU',
              'Physical-BMI', 'BIA-BIA_BMC', 'Physical-Systolic_BP']

print('=== Correlations among selected features (to detect redundancy) ===')
corr_matrix = df[selected].corr()
print(corr_matrix.round(2).to_string())
print()
print('=== Correlations of candidates with selected features ===')
for cand in candidates:
    corr_with_selected = df[selected].corrwith(df[cand])
    sii_corr = df['sii'].corr(df[cand])
    print(f'{cand} (sii_corr={sii_corr:.3f}):')
    print('  ', corr_with_selected.round(2).to_dict())
"

OUT
=== Correlations among selected features (to detect redundancy) ===
                                        Basic_Demos-Age  Basic_Demos-Sex  CGAS-CGAS_Score  Physical-Height  Physical-Weight  BIA-BIA_FFMI  BIA-BIA_FFM  PreInt_EduHx-computerinternet_hoursday
Basic_Demos-Age                                    1.00             0.01             0.01             0.72             0.68          0.46         0.68                                    0.40
Basic_Demos-Sex                                    0.01             1.00             0.02            -0.02            -0.03         -0.09        -0.08                                   -0.01
CGAS-CGAS_Score                                    0.01             0.02             1.00             0.01            -0.01         -0.00         0.01                                   -0.01
Physical-Height                                    0.72            -0.02             0.01             1.00             0.75          0.53         0.77                                    0.35
Physical-Weight                                    0.68            -0.03            -0.01             0.75             1.00          0.58         0.76                                    0.37
BIA-BIA_FFMI                                       0.46            -0.09            -0.00             0.53             0.58          1.00         0.69                                    0.27
BIA-BIA_FFM                                        0.68            -0.08             0.01             0.77             0.76          0.69         1.00                                    0.36
PreInt_EduHx-computerinternet_hoursday             0.40            -0.01            -0.01             0.35             0.37          0.27         0.36                                    1.00

=== Correlations of candidates with selected features ===
FGC-FGC_CU (sii_corr=0.182):
   {'Basic_Demos-Age': 0.46, 'Basic_Demos-Sex': -0.01, 'CGAS-CGAS_Score': 0.03, 'Physical-Height': 0.47, 'Physical-Weight': 0.43, 'BIA-BIA_FFMI': 0.26, 'BIA-BIA_FFM': 0.47, 'PreInt_EduHx-computerinternet_hoursday': 0.2}
BIA-BIA_Fat (sii_corr=0.176):
   {'Basic_Demos-Age': 0.56, 'Basic_Demos-Sex': 0.0, 'CGAS-CGAS_Score': 0.0, 'Physical-Height': 0.62, 'Physical-Weight': 0.7, 'BIA-BIA_FFMI': 0.62, 'BIA-BIA_FFM': 0.75, 'PreInt_EduHx-computerinternet_hoursday': 0.31}
SDS-SDS_Total_T (sii_corr=0.165):
   {'Basic_Demos-Age': 0.06, 'Basic_Demos-Sex': -0.03, 'CGAS-CGAS_Score': -0.07, 'Physical-Height': 0.07, 'Physical-Weight': 0.09, 'BIA-BIA_FFMI': 0.12, 'BIA-BIA_FFM': 0.07, 'PreInt_EduHx-computerinternet_hoursday': 0.07}
Physical-Waist_Circumference (sii_corr=0.147):
   {'Basic_Demos-Age': 0.53, 'Basic_Demos-Sex': -0.04, 'CGAS-CGAS_Score': 0.01, 'Physical-Height': 0.62, 'Physical-Weight': 0.65, 'BIA-BIA_FFMI': 0.53, 'BIA-BIA_FFM': 0.63, 'PreInt_EduHx-computerinternet_hoursday': 0.3}
BIA-BIA_FMI (sii_corr=0.136):
   {'Basic_Demos-Age': 0.45, 'Basic_Demos-Sex': 0.01, 'CGAS-CGAS_Score': -0.0, 'Physical-Height': 0.52, 'Physical-Weight': 0.6, 'BIA-BIA_FFMI': 0.63, 'BIA-BIA_FFM': 0.63, 'PreInt_EduHx-computerinternet_hoursday': 0.26}
FGC-FGC_PU (sii_corr=0.136):
   {'Basic_Demos-Age': 0.31, 'Basic_Demos-Sex': -0.11, 'CGAS-CGAS_Score': 0.0, 'Physical-Height': 0.32, 'Physical-Weight': 0.26, 'BIA-BIA_FFMI': 0.19, 'BIA-BIA_FFM': 0.31, 'PreInt_EduHx-computerinternet_hoursday': 0.13}
Physical-BMI (sii_corr=0.121):
   {'Basic_Demos-Age': 0.43, 'Basic_Demos-Sex': -0.0, 'CGAS-CGAS_Score': -0.0, 'Physical-Height': 0.45, 'Physical-Weight': 0.67, 'BIA-BIA_FFMI': 0.57, 'BIA-BIA_FFM': 0.56, 'PreInt_EduHx-computerinternet_hoursday': 0.26}
BIA-BIA_BMC (sii_corr=0.113):
   {'Basic_Demos-Age': 0.37, 'Basic_Demos-Sex': -0.06, 'CGAS-CGAS_Score': 0.01, 'Physical-Height': 0.46, 'Physical-Weight': 0.41, 'BIA-BIA_FFMI': 0.42, 'BIA-BIA_FFM': 0.52, 'PreInt_EduHx-computerinternet_hoursday': 0.19}
Physical-Systolic_BP (sii_corr=0.109):
   {'Basic_Demos-Age': 0.29, 'Basic_Demos-Sex': -0.03, 'CGAS-CGAS_Score': -0.04, 'Physical-Height': 0.31, 'Physical-Weight': 0.36, 'BIA-BIA_FFMI': 0.27, 'BIA-BIA_FFM': 0.33, 'PreInt_EduHx-computerinternet_hoursday': 0.17}
Bash
IN
python3 -c "
import json
with open('CMI/src/module_0.ipynb') as f:
    nb = json.load(f)

# Print all cell sources to find FGC_FitnessScore and BIA-BIA_BMI
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'FGC_FitnessScore' in src or 'BIA-BIA_BMI' in src or 'cols' in src:
        print(f'=== Cell {i} ({cell[\"cell_type\"]}) ===')
        print(src[:2000])
        print()
"

OUT
=== Cell 7 (code) ===
# Histograms of key continuous features (non-PCIAT, non-Season)
cont_cols = [
    "Basic_Demos-Age", "CGAS-CGAS_Score",
    "Physical-BMI", "Physical-Height", "Physical-Weight",
    "Physical-Waist_Circumference", "Physical-Diastolic_BP",
    "Physical-HeartRate", "Physical-Systolic_BP",
    "Fitness_Endurance-Max_Stage", "Fitness_Endurance-Time_Mins",
    "Fitness_Endurance-Time_Sec",
    "FGC-FGC_CU", "FGC-FGC_GSND", "FGC-FGC_GSD",
    "FGC-FGC_PU", "FGC-FGC_SRL", "FGC-FGC_SRR", "FGC-FGC_TL",
    "BIA-BIA_BMI", "BIA-BIA_BMR", "BIA-BIA_Fat", "BIA-BIA_FFM",
    "BIA-BIA_FMI", "BIA-BIA_FFMI", "BIA-BIA_SMM", "BIA-BIA_TBW",
    "BIA-BIA_DEE", "BIA-BIA_BMC", "BIA-BIA_ECW", "BIA-BIA_ICW",
    "BIA-BIA_LDM", "BIA-BIA_LST",
    "PAQ_A-PAQ_A_Total", "PAQ_C-PAQ_C_Total",
    "SDS-SDS_Total_Raw", "SDS-SDS_Total_T",
]

n = len(cont_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(cont_cols):
    df[col].dropna().hist(bins=40, ax=axes[i], edgecolor="k", alpha=0.7)
    axes[i].set_title(col.split("-")[-1], fontsize=9)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distributions of continuous features", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

=== Cell 9 (code) ===
# Bar plots for categorical / ordinal features
cat_cols = [
    "Basic_Demos-Sex",
    "FGC-FGC_CU_Zone", "FGC-FGC_PU_Zone", "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR_Zone", "FGC-FGC_TL_Zone",
    "FGC-FGC_GSND_Zone", "FGC-FGC_GSD_Zone",
    "BIA-BIA_Activity_Level_num", "BIA-BIA_Frame_num",
    "PreInt_EduHx-computerinternet_hoursday",
]

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    df[col].dropna().astype(int).value_counts().sort_index().plot.bar(
        ax=axes[i], edgecolor="k", alpha=0.8)
    axes[i].set_title(col.split("-")[-1], fontsize=9)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Categorical / ordinal feature distributions", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

=== Cell 11 (code) ===
# Box plots to spot outliers
bia_cols = [c for c in df.columns if c.startswith("BIA-") and "Season" not in c]
fig, axes = plt.subplots(3, 6, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(bia_cols):
    df[col].dropna().plot.box(ax=axes[i], vert=True)
    axes[i].set_title(col.replace("BIA-BIA_", ""), fontsize=9)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("BIA features — box plots (note extreme outliers)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

=== Cell 14 (code) ===
# Correlation with sii — only non-PCIAT, non-Season features
numeric = df.select_dtypes("number").drop(columns=["id"])
pciat_cols = [c for c in numeric.columns if "PCIAT" in c]
corr_sii = (numeric.drop(columns=pciat_cols)
            .corrwith(numeric["sii"])
            .drop("sii")
            .sort_values(key=abs, ascending=False))

fig, ax = plt.subplots(figsize=(10, 8))
corr_sii.plot.barh(ax=ax, color=corr_sii.apply(lambda x: "steelblue" if x >= 0 else "salmon"))
ax.set_title("Correlation with sii (non-PCIAT features)")
ax.set_xlabel("Pearson r")
ax.axvline(0, color="k", lw=0.5)
plt.tight_layout()
plt.show()

print("Top 10 absolute correlates:")
print(corr_sii.head(10).to_string())

=== Cell 15 (code) ===
# Correlation heatmap among feature groups (Physical + BIA + FGC + Demographics)
keep_for_heatmap = [
    "Basic_Demos-Age", "Basic_Demos-Sex",
    "Physical-BMI", "Physical-Height", "Physical-Weight",
    "Physical-Waist_Circumference", "Physical-Diastolic_BP",
    "Physical-HeartRate", "Physical-Systolic_BP",
    "BIA-BIA_BMI", "BIA-BIA_Fat", "BIA-BIA_FFM", "BIA-BIA_FMI",
    "BIA-BIA_FFMI", "BIA-BIA_SMM", "BIA-BIA_TBW", "BIA-BIA_BMR",
    "BIA-BIA_DEE",
    "SDS-SDS_Total_T", "PreInt_EduHx-computerinternet_hoursday",
    "sii",
]

corr_mat = df[keep_for_heatmap].corr()
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.3, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation heatmap — Physical, BIA, Sleep, Demographics, Target")
plt.tight_layout()
plt.show()

=== Cell 21 (code) ===
# --- 5a. Drop columns: id, all PCIAT, all Season columns ---
season_cols = [c for c in df.columns if "Season" in c]
pciat_cols_all = [c for c in df.columns if "PCIAT" in c]

print(f"Dropping {len(season_cols)} Season columns: {season_cols}")
print(f"Dropping {len(pciat_cols_all)} PCIAT columns")
print("Dropping: id")

df_clean = df.drop(columns=["id"] + season_cols + pciat_cols_all)
print(f"\nShape after drops: {df_clean.shape}")

=== Cell 26 (markdown) ===
The seven FGC Zone columns classify fitness outcomes per exercise:
- Five columns (`CU`, `PU`, `SRL`, `SRR`, `TL`) use a **binary** encoding: 0 = below Healthy Fitness Zone, 1 = in Healthy Fitness Zone.
- Two columns (`GSND`, `GSD`) use a **ternary** encoding: 1 = Weak, 2 = Normal, 3 = Strong.

The ternary columns are recoded to the same binary scale (1 → 0; 2 or 3 → 1), then all seven are summed into `FGC_FitnessScore` (range 0–7): higher = more fitness zones met.

=== Cell 27 (code) ===
fgc_zone_cols = [
    "FGC-FGC_CU_Zone",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD_Zone",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL_Zone",
]

# Recode ternary columns (1=Weak, 2=Normal, 3=Strong) -> binary (0 / 1)
ternary_cols = ["FGC-FGC_GSND_Zone", "FGC-FGC_GSD_Zone"]
for col in ternary_cols:
    df_clean[col] = df_clean[col].map({1.0: 0, 2.0: 1, 3.0: 1})

# Fitness score = count of zones met (0-7)
df_clean["FGC_FitnessScore"] = df_clean[fgc_zone_cols].sum(axis=1, min_count=1)

print(f"FGC_FitnessScore - non-null: {df_clean['FGC_FitnessScore'].notna().sum()}, "
      f"null: {df_clean['FGC_FitnessScore'].isna().sum()}")
df_clean["FGC_FitnessScore"].value_counts().sort_index()

=== Cell 35 (code) ===
print(df_clean.columns)

cols = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-Height', 'Physical-Weight', 'FGC_FitnessScore', 'BIA-BIA_FFMI', 'BIA-BIA_FFM', 'BIA-BIA_BMI',  'PreInt_EduHx-computerinternet_hoursday', 'sii',]
print(len(cols))

=== Cell 37 (code) ===
# --- 9b. Inter-feature correlation heatmap (BIA block) ---
bia_cols_check = [c for c in df_clean.columns if c.startswith("BIA-") and "Season" not in c]
corr_bia = df_clean[bia_cols_check].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_bia, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.3, ax=ax, annot_kws={"size": 8})
ax.set_title("BIA inter-feature correlation (pre-imputation)")
plt.tight_layout()
plt.show()

# Print pairs with |r| > 0.80
print("Feature pairs with |r| > 0.80:")
for i in range(len(corr_bia.columns)):
    for j in range(i+1, len(corr_bia.columns)):
        r = corr_bia.iloc[i, j]
        if abs(r) > 0.80:
            print(f"  {r:+.3f}  {corr_bia.columns[i]}  <->  {corr_bia.columns[j]}")

=== Cell 38 (code) ===
# --- 9c. Drop redundant / low-signal variables ---

# A. Multicollinear BIA lean-mass cluster (keep FFM only)
bia_drop = [
    "BIA-BIA_LST",   # r=0.907 with FFM
    "BIA-BIA_BMR",   # r=0.916 with FFM
    "BIA-BIA_TBW",   # r=0.883 with FFM (= ECW + ICW)
    "BIA-BIA_ECW",   # r=0.859 with FFM
    "BIA-BIA_ICW",   # r=0.848 with FFM
    "BIA-BIA_SMM",   # r=0.838 with FFM
    "BIA-BIA_DEE",   # derived from BMR
    "BIA-BIA_LDM",   # r=0.823 with BMR
]

# B. Duplicate metrics
dup_drop = [
    "BIA-BIA_BMI",       # same as Physical-BMI
    "SDS-SDS_Total_Raw", # T-score is the standardized form
]

# C. FGC: drop whichever of raw/zone has lower sii signal
fgc_drop = [
    "FGC-FGC_CU_Zone",   # |r|=0.005 vs raw 0.200
    "FGC-FGC_PU_Zone",   # |r|=0.049 vs raw 0.148
    "FGC-FGC_GSND_Zone", # |r|=0.049 vs raw 0.100, 84% one class
    "FGC-FGC_GSD_Zone",  # |r|=0.052 vs raw 0.111, 85% one class
    "FGC-FGC_SRL",       # |r|=0.018 vs zone 0.076
    "FGC-FGC_SRR",       # |r|=0.029 vs zone 0.060
    "FGC-FGC_TL",        # |r|=0.044 vs zone 0.055
]

# D. Near-zero signal
zero_drop = [
    "Fitness_Endurance_TotalTime",       # |r|=0.003
    "PAQ_Combined",                       # |r|=0.008
    "BIA-BIA_Activity_Level_num",         # |r|=0.028, 56% one class
]

all_drop = bia_drop + dup_drop + fgc_drop + zero_drop
print(f"Dropping {len(all_drop)} variables:")
for group, cols in [("BIA cluster", bia_drop), ("Duplicates", dup_drop),
                     ("FGC raw/zone", fgc_drop), ("Near-zero signal", zero_drop)]:
    print(f"\n  {group} ({len(cols)}):")
    for c in cols:
        print(f"    - {c}")

df_clean = df_clean.drop(columns=all_drop)
print(f"\nShape after variable selection: {df_clean.shape}")
print(f"Remaining features: {[c for c in df_clean.columns if c != 'sii']}")

=== Cell 43 (code) ===
import miceforest as mf

# Separate target before imputation
sii = df_clean["sii"].copy()
df_to_impute = df_clean.drop(columns=["sii"]).copy()

# Cast categorical columns to pd.Categorical
# (updated after variable selection — dropped zones/activity level)
categorical_cols = [
    # Binary (0/1)
    "Basic_Demos-Sex",
    "FGC-FGC_SRL_Zone", "FGC-FGC_SRR_Zone", "FGC-FGC_TL_Zone",
    # Ordinal
    "BIA-BIA_Frame_num",                       # 1/2/3
    "PreInt_EduHx-computerinternet_hoursday",   # 0–3
]

for col in categorical_cols:
    df_to_impute[col] = df_to_impute[col].astype("category")

print(f"Shape to impute: {df_to_impute.shape}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Continuous columns: {df_to_impute.shape[1] - len(categorical_cols)}")
print(f"\nMissing values before imputation: {df_to_impute.isnull().sum().sum()}")

=== Cell 44 (code) ===
# Create imputation kernel and run MICE
# PMM (mean_match_candidates > 0) can crash with "data must be finite" on categorical
# columns where LightGBM produces p=0 or p=1, because logodds(p) → ±inf which breaks
# KDTree. Fix: use 'shap' mean match strategy for categoricals — avoids the logodds
# transformation entirely. Continuous columns keep the default 'normal' PMM strategy.
NUM_DATASETS = 3

mm_strategy = {col: "shap" for col in categorical_cols}

kernel = mf.ImputationKernel(
    data=df_to_impute,
    num_datasets=NUM_DATASETS,
    random_state=42,
    mean_match_candidates=5,
    mean_match_strategy=mm_strategy,
)

kernel.mice(iterations=5, verbose=True)

=== Cell 45 (code) ===
# Pool across all imputed datasets:
# - Continuous columns: average across datasets
# - Categorical columns: majority vote across datasets
all_datasets = [kernel.complete_data(dataset=i) for i in range(NUM_DATASETS)]

continuous_cols = [c for c in df_to_impute.columns if c not in categorical_cols]

df_imputed = all_datasets[0].copy()

# Average continuous columns across all datasets
for col in continuous_cols:
    df_imputed[col] = np.mean([ds[col].values for ds in all_datasets], axis=0)

# Majority vote for categorical columns across all datasets
for col in categorical_cols:
    stacked = pd.DataFrame({i: ds[col] for i, ds in enumerate(all_datasets)})
    df_imputed[col] = stacked.mode(axis=1)[0]

# Re-attach the target
df_imputed["sii"] = sii.values

# Verify: no missing values remain
print(f"Missing values after imputation: {df_imputed.isnull().sum().sum()}")
print(f"Shape: {df_imputed.shape}")
print()

# Sanity check: categorical columns still have valid discrete values
print("Categorical columns — unique values after imputation:")
for col in categorical_cols:
    vals = sorted(df_imputed[col].dropna().unique())
    print(f"  {col.split('-')[-1]}: {vals}")

=== Cell 46 (code) ===
# Compare distributions before vs after imputation for ALL columns
check_cols = [c for c in df_imputed.columns if c != "sii"]

ncols = 4
nrows = (len(check_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(check_cols):
    axes[i].hist(df_clean[col].dropna(), bins=40, alpha=0.5, label="Before", density=True, edgecolor="k")
    axes[i].hist(df_imputed[col].dropna(), bins=40, alpha=0.5, label="After", density=True, edgecolor="k")
    axes[i].set_title(col.split("-")[-1], fontsize=9)
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribution comparison: before vs after MICE imputation", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

=== Cell 47 (code) ===
# Export the imputed dataset
# Convert categoricals back to int for export
for col in categorical_cols:
    df_imputed[col] = df_imputed[col].astype(float).astype(int)

df_imputed.to_csv("../processed_data/cmi_imputed.csv", index=False)
print(f"Saved imputed dataset to processed_data/cmi_imputed.csv")
print(f"Final shape: {df_imputed.shape}")
df_imputed.describe().T

=== Cell 49 (code) ===
# Identify columns to log-scale: continuous, skew > 1, min >= 0, not CGAS
categorical_cols_final = [
    "Basic_Demos-Sex", "Basic_Demos-Age",
    "FGC-FGC_SRL_Zone", "FGC-FGC_SRR_Zone", "FGC-FGC_TL_Zone",
    "BIA-BIA_Frame_num",
    "PreInt_EduHx-computerinternet_hoursday", "sii",
]
skip_cols = categorical_cols_final + ["CGAS-CGAS_Score"]

continuous_cols = [c for c in df_imputed.columns if c not in skip_cols]

# Select skewed columns with no negative values
log_candidates = []
for col in continuous_cols:
    sk = df_imputed[col].skew()
    mn = df_imputed[col].min()
    if abs(sk) > 1 and mn >= 0:
        log_candidates.append((col, sk))

log_candidates.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"Columns selected for log1p transform ({len(log_candidates)}):\n")
print(f"{'Column':<45s}  {'Skew (before)':>13s}")
print("-" * 62)
for col, sk in log_candidates:
    print(f"{col:<45s}  {sk:>+13.2f}")

=== Cell 50 (code) ===
# Apply log1p and compare skewness before vs after
log_cols = [col for col, _ in log_candidates]

skew_before = df_imputed[log_cols].skew()
df_imputed[log_cols] = np.log1p(df_imputed[log_cols])
skew_after = df_imputed[log_cols].skew()

comparison = pd.DataFrame({
    "skew_before": skew_before,
    "skew_after": skew_after,
    "reduction": (skew_before.abs() - skew_after.abs()).round(2),
}).sort_values("skew_before", key=abs, ascending=False)

print("Skewness before vs after log1p transform:\n")
print(comparison.to_string())

=== Cell 51 (code) ===
# Visual comparison: distributions before vs after log1p
ncols = 4
nrows = (len(log_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(log_cols):
    # Reconstruct original via expm1 for overlay
    original = np.expm1(df_imputed[col])
    axes[i].hist(df_imputed[col], bins=40, alpha=0.5, label="After log-scaling", density=True, edgecolor="k")
    axes[i].set_title(col.split("-")[-1], fontsize=9)
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distributions after log1p scaling", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
Bash
IN
python3 -c "
import pandas as pd
df = pd.read_csv('CMI/processed_data/cmi_imputed.csv')
print('Imputed shape:', df.shape)
print('Columns:', list(df.columns))
print()
corr = df.corr()['sii'].sort_values(key=abs, ascending=False)
print('Correlations with sii in imputed data:')
print(corr.to_string())
"

OUT
Imputed shape: (8243, 27)
Columns: ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP', 'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone', 'BIA-BIA_BMC', 'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday', 'sii']

Correlations with sii in imputed data:
sii                                       1.000000
Physical-Height                           0.229336
BIA-BIA_FFM                               0.226062
Physical-Weight                           0.220146
Basic_Demos-Age                           0.212355
PreInt_EduHx-computerinternet_hoursday    0.207190
FGC-FGC_CU                                0.195839
SDS-SDS_Total_T                           0.171398
BIA-BIA_Fat                               0.168099
BIA-BIA_FFMI                              0.153031
FGC-FGC_PU                                0.142917
Physical-Waist_Circumference              0.137939
BIA-BIA_FMI                               0.129505
Physical-BMI                              0.122683
Physical-Systolic_BP                      0.111950
BIA-BIA_BMC                               0.111576
FGC-FGC_GSD                               0.107534
FGC-FGC_GSND                              0.100470
BIA-BIA_Frame_num                         0.086164
Basic_Demos-Sex                          -0.083843
FGC-FGC_SRL_Zone                         -0.077698
FGC-FGC_SRR_Zone                         -0.065370
FGC-FGC_TL_Zone                          -0.054205
Physical-Diastolic_BP                     0.050491
Physical-HeartRate                       -0.034734
CGAS-CGAS_Score                          -0.033418
Fitness_Endurance-Max_Stage              -0.016303
Bash
IN
python3 -c "
import json
with open('CMI/src/module_0.ipynb') as f:
    nb = json.load(f)

# Look at cells around 35
for i, cell in enumerate(nb['cells']):
    if i >= 33 and i <= 42:
        src = ''.join(cell['source'])
        print(f'=== Cell {i} ({cell[\"cell_type\"]}) ===')
        print(src[:1500])
        print()
"

OUT
=== Cell 33 (code) ===
# Convert imperial → metric
df_clean["Physical-Height"] = df_clean["Physical-Height"] * 2.54          # in → cm
df_clean["Physical-Weight"] = df_clean["Physical-Weight"] * 0.453592      # lbs → kg
df_clean["Physical-Waist_Circumference"] = df_clean["Physical-Waist_Circumference"] * 2.54  # in → cm

print("Converted to metric:")
print(f"  Height: {df_clean['Physical-Height'].describe()[['mean','min','max']].to_string()}")
print(f"  Weight: {df_clean['Physical-Weight'].describe()[['mean','min','max']].to_string()}")
print(f"  Waist:  {df_clean['Physical-Waist_Circumference'].describe()[['mean','min','max']].to_string()}")

=== Cell 34 (markdown) ===
---
## Step 9. Variable Selection (before imputation)

We remove redundant and low-signal variables **before** imputation so that MICE
operates on a cleaner, lower-dimensional feature set.

Selection criteria (all computed on the cleaned, pre-imputation data):

### A. Multicollinear BIA lean-mass cluster

Eight BIA variables form a single highly correlated block (all pairwise |r| > 0.80):
`FFM`, `LST`, `BMR`, `TBW`, `ECW`, `ICW`, `SMM`, `LDM`, `DEE`.

They all measure overlapping aspects of lean body mass / body water / metabolism.
We keep **`BIA_FFM`** (highest sii correlation in the cluster, r = 0.241) and drop the rest.

| Drop | r with FFM | r with sii | Why redundant |
|---|---|---|---|
| `BIA_LST` | 0.907 | 0.241 | FFM ≈ LST + BMC |
| `BIA_BMR` | 0.916 | 0.225 | Metabolic derivative of lean mass |
| `BIA_TBW` | 0.883 | 0.227 | Water component = ECW + ICW |
| `BIA_ECW` | 0.859 | 0.206 | Subset of TBW |
| `BIA_ICW` | 0.848 | 0.235 | Subset of TBW |
| `BIA_SMM` | 0.838 | 0.218 | Skeletal subset of FFM |
| `BIA_DEE` | via BMR | 0.217 | DEE = BMR × activity multiplier |
| `BIA_LDM` | via BMR 0.823 | 0.200 | Dry component of lean mass |

### B. Duplicate metrics

| Drop | Keep | Reason |
|---|---|---|
| `BIA_BMI` (r=0.159) | `Physical-BMI` (r=0.126) | Same metric from different instrument |
| `SDS_Total_Raw` (r=0.196) | `SDS_Total_T` (r=0.210) | T-score is the standardized form (r=0.808 between them) |

### C. FGC: keep whichever of raw / zone has higher sii signal

| Test | 

=== Cell 35 (code) ===
print(df_clean.columns)

cols = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-Height', 'Physical-Weight', 'FGC_FitnessScore', 'BIA-BIA_FFMI', 'BIA-BIA_FFM', 'BIA-BIA_BMI',  'PreInt_EduHx-computerinternet_hoursday', 'sii',]
print(len(cols))

=== Cell 36 (code) ===
# --- 9a. Correlation analysis (pre-imputation, pairwise complete obs) ---
numeric_pre = df_clean.select_dtypes("number")
corr_sii_pre = (numeric_pre.corrwith(numeric_pre["sii"])
                .drop("sii")
                .sort_values(key=abs, ascending=False))

fig, ax = plt.subplots(figsize=(10, 10))
corr_sii_pre.plot.barh(ax=ax,
    color=corr_sii_pre.apply(lambda x: "steelblue" if x >= 0 else "salmon"))
ax.set_title("Correlation with sii (pre-imputation, pairwise complete)")
ax.set_xlabel("Pearson r")
ax.axvline(0, color="k", lw=0.5)
plt.tight_layout()
plt.show()

print("Top 15 absolute correlates with sii:")
print(corr_sii_pre.head(15).round(3).to_string())

=== Cell 37 (code) ===
# --- 9b. Inter-feature correlation heatmap (BIA block) ---
bia_cols_check = [c for c in df_clean.columns if c.startswith("BIA-") and "Season" not in c]
corr_bia = df_clean[bia_cols_check].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_bia, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.3, ax=ax, annot_kws={"size": 8})
ax.set_title("BIA inter-feature correlation (pre-imputation)")
plt.tight_layout()
plt.show()

# Print pairs with |r| > 0.80
print("Feature pairs with |r| > 0.80:")
for i in range(len(corr_bia.columns)):
    for j in range(i+1, len(corr_bia.columns)):
        r = corr_bia.iloc[i, j]
        if abs(r) > 0.80:
            print(f"  {r:+.3f}  {corr_bia.columns[i]}  <->  {corr_bia.columns[j]}")

=== Cell 38 (code) ===
# --- 9c. Drop redundant / low-signal variables ---

# A. Multicollinear BIA lean-mass cluster (keep FFM only)
bia_drop = [
    "BIA-BIA_LST",   # r=0.907 with FFM
    "BIA-BIA_BMR",   # r=0.916 with FFM
    "BIA-BIA_TBW",   # r=0.883 with FFM (= ECW + ICW)
    "BIA-BIA_ECW",   # r=0.859 with FFM
    "BIA-BIA_ICW",   # r=0.848 with FFM
    "BIA-BIA_SMM",   # r=0.838 with FFM
    "BIA-BIA_DEE",   # derived from BMR
    "BIA-BIA_LDM",   # r=0.823 with BMR
]

# B. Duplicate metrics
dup_drop = [
    "BIA-BIA_BMI",       # same as Physical-BMI
    "SDS-SDS_Total_Raw", # T-score is the standardized form
]

# C. FGC: drop whichever of raw/zone has lower sii signal
fgc_drop = [
    "FGC-FGC_CU_Zone",   # |r|=0.005 vs raw 0.200
    "FGC-FGC_PU_Zone",   # |r|=0.049 vs raw 0.148
    "FGC-FGC_GSND_Zone", # |r|=0.049 vs raw 0.100, 84% one class
    "FGC-FGC_GSD_Zone",  # |r|=0.052 vs raw 0.111, 85% one class
    "FGC-FGC_SRL",       # |r|=0.018 vs zone 0.076
    "FGC-FGC_SRR",       # |r|=0.029 vs zone 0.060
    "FGC-FGC_TL",        # |r|=0.044 vs zone 0.055
]

# D. Near-zero signal
zero_drop = [
    "Fitness_Endurance_TotalTime",       # |r|=0.003
    "PAQ_Combined",                       # |r|=0.008
    "BIA-BIA_Activity_Level_num",         # |r|=0.028, 56% one class
]

all_drop = bia_drop + dup_drop + fgc_drop + zero_drop
print(f"Dropping {len(all_drop)} variables:")
for group, cols in [("BIA cluster", bia_drop), ("Duplicates", dup_drop),
                     ("FGC raw/zone", fgc_drop)

=== Cell 39 (markdown) ===
---
## Step 10. Deterministic Imputation

Before statistical imputation, we compute missing values from known physical formulas where 2 of 3 variables are present:

- BMI = Weight_kg / (Height_m)²  — recover any one from the other two

=== Cell 40 (code) ===
def deterministic_impute(df):
    """Fill missing values using known physical formulas.
    Only fills where exactly 1 of the group is missing and the others are present."""
    filled = {}

    # --- BMI / Height / Weight (metric: BMI = kg / m²) ---
    h = df["Physical-Height"]       # cm
    w = df["Physical-Weight"]       # kg
    b = df["Physical-BMI"]          # kg/m²
    h_m = h / 100                   # cm → m

    # BMI missing, H + W present
    mask = b.isna() & h.notna() & w.notna() & (h_m > 0)
    filled["Physical-BMI"] = mask.sum()
    df.loc[mask, "Physical-BMI"] = w[mask] / (h_m[mask] ** 2)

    # Weight missing, H + BMI present
    mask = w.isna() & h.notna() & b.notna() & (h_m > 0)
    filled["Physical-Weight"] = mask.sum()
    df.loc[mask, "Physical-Weight"] = b[mask] * (h_m[mask] ** 2)

    # Height missing, W + BMI present
    mask = h.isna() & w.notna() & b.notna() & (b > 0)
    filled["Physical-Height"] = mask.sum()
    df.loc[mask, "Physical-Height"] = np.sqrt(w[mask] / b[mask]) * 100  # m → cm

    return filled

filled = deterministic_impute(df_clean)
total = sum(filled.values())
print(f"Deterministic imputation — {total} values recovered:")
for col, n in filled.items():
    if n > 0:
        print(f"  {col}: {n} values computed from formula")

=== Cell 41 (code) ===
# Updated missing-value summary after deterministic imputation
print(f"Shape: {df_clean.shape}")
print()

miss_final = df_clean.isnull().sum()
miss_final_pct = (miss_final / len(df_clean) * 100).round(1)
summary = (pd.DataFrame({"missing": miss_final, "pct%": miss_final_pct})
           .sort_values("pct%", ascending=False))
print("Missing values per column after deterministic imputation:")
summary

=== Cell 42 (markdown) ===
---
## Step 11. Statistical Imputation (MICE - miceforest)

We use miceforest, a specific implementation for MICE (Multiple Imputation by Chained Equations with LightGBM) that uses LightGBM to fill remaining missing values.

The target `sii` is excluded from imputation — it has no missing values and should not influence feature imputation.
Here's the full analysis. Several issues and opportunities:

Bugs in current cols
1. BIA-BIA_BMI doesn't exist in the imputed dataset. It was explicitly dropped in cell 38 as a duplicate of Physical-BMI. Using this column will cause a KeyError downstream.

2. FGC_FitnessScore doesn't exist in the imputed dataset. It's computed in cell 27 from zone columns, but cmi_imputed.csv doesn't include it — only the raw FGC columns (FGC-FGC_CU, FGC-FGC_PU, FGC-FGC_GSD, FGC-FGC_GSND) and three zone columns survived. You'd need to recompute it post-imputation, or use the raw columns instead.

Correlations with sii (on imputed data, sorted)
Column	r with sii	In your cols?
Physical-Height	0.229	✓
BIA-BIA_FFM	0.226	✓
Physical-Weight	0.220	✓
Basic_Demos-Age	0.212	✓
PreInt_EduHx-computerinternet_hoursday	0.207	✓
FGC-FGC_CU	0.196	✗ missing
SDS-SDS_Total_T	0.171	✗ missing
BIA-BIA_Fat	0.168	✗ missing
BIA-BIA_FFMI	0.153	✓
FGC-FGC_PU	0.143	✗
Physical-Waist_Circumference	0.138	✗
CGAS-CGAS_Score	-0.033	✓ very weak
What to fix/add
Drop from cols:

BIA-BIA_BMI — doesn't exist post-imputation
FGC_FitnessScore — doesn't exist post-imputation
CGAS-CGAS_Score — |r|=0.033, essentially noise
Most valuable missing columns:

SDS-SDS_Total_T — sleep disturbance score, nearly independent from physical vars (r≈0.07 with FFM/Height/Weight), directly relevant to internet addiction, r=0.171 with sii
FGC-FGC_CU (r=0.196) — curl-ups, the best single FGC measure; replaces the dropped FGC_FitnessScore
BIA-BIA_Fat (r=0.168) — body fat %, complements FFM (captures fat dimension vs lean mass)
Note on redundancy: BIA-BIA_FFMI and BIA-BIA_FFM correlate at r=0.69, so one is partially redundant. But FFMI normalizes for height, so it adds some independent signal.

Suggested replacement

cols = [
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Physical-Height',
    'Physical-Weight',
    'BIA-BIA_FFM',
    'BIA-BIA_Fat',       # replaces BIA-BIA_BMI (fat dimension)
    'FGC-FGC_CU',        # replaces FGC_FitnessScore (best FGC measure, exists in imputed)
    'SDS-SDS_Total_T',   # new: sleep disturbance, most independent signal
    'PreInt_EduHx-computerinternet_hoursday',
    'sii',
]
This goes from 11 cols (2 broken) to 10 cols (all valid), adds the two most informative missing features (SDS-SDS_Total_T, FGC-FGC_CU), and drops the near-zero-signal CGAS-CGAS_Score. If you want to keep BIA-BIA_FFMI as well, it's defensible (normalized lean mass index), just be aware it overlaps significantly with BIA-BIA_FFM.

problem sleep has lots of missing
Bash
IN
python3 -c "
import pandas as pd
df = pd.read_csv('CMI/base_data/cmi_internet.csv')
print('Raw shape:', df.shape)
sleep_cols = [c for c in df.columns if 'SDS' in c]
fgc_cols = [c for c in df.columns if 'FGC' in c or 'Fitness' in c]
print()
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(1)
target_cols = ['SDS-SDS_Total_T', 'SDS-SDS_Total_Raw'] + fgc_cols + ['sii']
for c in target_cols:
    if c in df.columns:
        print(f'{c}: {miss_pct[c]}% missing ({miss[c]} rows)')
print()
# Show all columns with >30% missing
high_miss = miss_pct[miss_pct > 30].sort_values(ascending=False)
print('Columns with >30% missing:')
print(high_miss.to_string())
"

OUT
Raw shape: (8460, 82)

SDS-SDS_Total_T: 42.0% missing (3557 rows)
SDS-SDS_Total_Raw: 42.0% missing (3557 rows)
Fitness_Endurance-Season: 29.0% missing (2456 rows)
Fitness_Endurance-Max_Stage: 35.2% missing (2980 rows)
Fitness_Endurance-Time_Mins: 35.2% missing (2982 rows)
Fitness_Endurance-Time_Sec: 35.2% missing (2982 rows)
FGC-Season: 6.7% missing (569 rows)
FGC-FGC_CU: 17.9% missing (1517 rows)
FGC-FGC_CU_Zone: 18.4% missing (1554 rows)
FGC-FGC_GSND: 31.6% missing (2673 rows)
FGC-FGC_GSND_Zone: 31.7% missing (2684 rows)
FGC-FGC_GSD: 31.6% missing (2673 rows)
FGC-FGC_GSD_Zone: 31.7% missing (2684 rows)
FGC-FGC_PU: 18.1% missing (1528 rows)
FGC-FGC_PU_Zone: 18.5% missing (1565 rows)
FGC-FGC_SRL: 18.1% missing (1534 rows)
FGC-FGC_SRL_Zone: 18.5% missing (1569 rows)
FGC-FGC_SRR: 18.1% missing (1532 rows)
FGC-FGC_SRR_Zone: 18.5% missing (1567 rows)
FGC-FGC_TL: 17.9% missing (1515 rows)
FGC-FGC_TL_Zone: 18.3% missing (1552 rows)
sii: 0.0% missing (0 rows)

Columns with >30% missing:
PCIAT-PCIAT_06                  67.9
PCIAT-PCIAT_07                  67.9
PCIAT-Season                    67.9
PCIAT-PCIAT_17                  67.9
PCIAT-PCIAT_18                  67.9
PCIAT-PCIAT_19                  67.9
PCIAT-PCIAT_05                  67.9
PCIAT-PCIAT_04                  67.9
PCIAT-PCIAT_03                  67.9
PCIAT-PCIAT_02                  67.9
PCIAT-PCIAT_01                  67.9
PCIAT-PCIAT_08                  67.9
PCIAT-PCIAT_15                  67.9
PCIAT-PCIAT_14                  67.9
PCIAT-PCIAT_13                  67.9
PCIAT-PCIAT_12                  67.9
PCIAT-PCIAT_11                  67.9
PCIAT-PCIAT_10                  67.9
PCIAT-PCIAT_09                  67.9
PCIAT-PCIAT_Total               67.9
PCIAT-PCIAT_16                  67.9
PCIAT-PCIAT_20                  67.9
SDS-SDS_Total_Raw               42.0
SDS-SDS_Total_T                 42.0
SDS-Season                      42.0
PAQ_A-PAQ_A_Total               38.2
PAQ_A-Season                    38.2
Fitness_Endurance-Max_Stage     35.2
Fitness_Endurance-Time_Mins     35.2
Fitness_Endurance-Time_Sec      35.2
Physical-Waist_Circumference    33.5
FGC-FGC_GSND_Zone               31.7
FGC-FGC_GSD_Zone                31.7
FGC-FGC_GSD                     31.6
FGC-FGC_GSND                    31.6
Bash
IN
python3 -c "
import pandas as pd
df = pd.read_csv('CMI/base_data/cmi_internet.csv')
miss_pct = (df.isnull().sum() / len(df) * 100).round(1)

check = ['Basic_Demos-Age','Basic_Demos-Sex','CGAS-CGAS_Score',
         'Physical-Height','Physical-Weight','Physical-BMI','Physical-Waist_Circumference',
         'BIA-BIA_FFMI','BIA-BIA_FFM','BIA-BIA_Fat','BIA-BIA_FMI','BIA-BIA_BMI',
         'FGC-FGC_CU','FGC-FGC_PU','FGC-FGC_GSD','FGC-FGC_GSND',
         'SDS-SDS_Total_T','PreInt_EduHx-computerinternet_hoursday','sii']

for c in check:
    if c in miss_pct.index:
        print(f'{miss_pct[c]:5.1f}%  {c}')
"

OUT
  0.0%  Basic_Demos-Age
  0.0%  Basic_Demos-Sex
 16.9%  CGAS-CGAS_Score
 10.2%  Physical-Height
  9.7%  Physical-Weight
 10.3%  Physical-BMI
 33.5%  Physical-Waist_Circumference
 21.6%  BIA-BIA_FFMI
 21.6%  BIA-BIA_FFM
 21.6%  BIA-BIA_Fat
 21.6%  BIA-BIA_FMI
 21.6%  BIA-BIA_BMI
 17.9%  FGC-FGC_CU
 18.1%  FGC-FGC_PU
 31.6%  FGC-FGC_GSD
 31.6%  FGC-FGC_GSND
 42.0%  SDS-SDS_Total_T
  7.2%  PreInt_EduHx-computerinternet_hoursday
  0.0%  sii
You're right to flag this. Here's the full picture ranked by missing rate:

Column	Missing (raw)	r with sii	Notes
Basic_Demos-Age	0%	0.21	
Basic_Demos-Sex	0%	-0.08	
sii	0%	—	target
PreInt_EduHx-computerinternet_hoursday	7.2%	0.21	
Physical-Height	10.2%	0.23	
Physical-Weight	9.7%	0.22	
CGAS-CGAS_Score	16.9%	-0.033	high missing + near-zero signal
FGC-FGC_CU	17.9%	0.196	
FGC-FGC_PU	18.1%	0.143	
BIA-BIA_FFM	21.6%	0.226	
BIA-BIA_FFMI	21.6%	0.153	
BIA-BIA_Fat	21.6%	0.168	
FGC-FGC_GSD/GSND	31.6%	0.10	
Physical-Waist_Circumference	33.5%	0.138	too much missing
SDS-SDS_Total_T	42.0%	0.171	too risky to impute
So SDS is out — 42% imputed would be mostly synthetic noise. Same logic applies to Physical-Waist_Circumference (33.5%) and FGC_GSD/GSND (31.6%).

Revised recommendation:


cols = [
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Physical-Height',
    'Physical-Weight',
    'BIA-BIA_FFM',
    'BIA-BIA_Fat',       # 21.6% — same block as FFM, acceptable
    'FGC-FGC_CU',        # 17.9% — best FGC measure, acceptable
    'PreInt_EduHx-computerinternet_hoursday',
    'sii',
]
vs. your original:

Drops: BIA-BIA_BMI (broken), FGC_FitnessScore (broken), CGAS-CGAS_Score (16.9% missing + r=-0.033), BIA-BIA_FFMI (redundant with FFM at r=0.69)
Adds: BIA-BIA_Fat (r=0.168, independent fat dimension), FGC-FGC_CU (r=0.196, best fitness predictor available)
The BIA block at 21.6% is borderline but all those columns are missing together (same instrument), so MICE can use the others to impute — that's more reliable than 42% random missingness like SDS.

im trying to readd the firnessscore and bia bmi instead of the physical. overview of columns and what to add
Bash
IN
python3 -c "
import pandas as pd
import numpy as np

raw = pd.read_csv('CMI/base_data/cmi_internet.csv')
n = len(raw)

# Simulate df_clean at cell 35: drop id, Season, PCIAT; add FGC_FitnessScore
drop = ['id'] + [c for c in raw.columns if 'Season' in c or 'PCIAT' in c]
df = raw.drop(columns=drop)

# Cell 27: recode ternary zone cols, compute FGC_FitnessScore
ternary = ['FGC-FGC_GSND_Zone','FGC-FGC_GSD_Zone']
for col in ternary:
    df[col] = df[col].map({1.0:0, 2.0:1, 3.0:1})
zone_cols = ['FGC-FGC_CU_Zone','FGC-FGC_GSND_Zone','FGC-FGC_GSD_Zone',
             'FGC-FGC_PU_Zone','FGC-FGC_SRL_Zone','FGC-FGC_SRR_Zone','FGC-FGC_TL_Zone']
df['FGC_FitnessScore'] = df[zone_cols].sum(axis=1, min_count=1)

miss_pct = (df.isnull().sum() / n * 100).round(1)
corr = df.select_dtypes('number').corrwith(df['sii']).round(3)

summary = pd.DataFrame({'miss%': miss_pct, 'r_sii': corr}).dropna(subset=['r_sii'])
summary['abs_r'] = summary['r_sii'].abs()
summary = summary.sort_values('abs_r', ascending=False)
print(summary[['miss%','r_sii']].to_string())
"

OUT
                                        miss%  r_sii
sii                                       0.0  1.000
Physical-Height                          10.2  0.236
PreInt_EduHx-computerinternet_hoursday    7.2  0.218
Physical-Weight                           9.7  0.216
SDS-SDS_Total_T                          42.0  0.214
Basic_Demos-Age                           0.0  0.211
SDS-SDS_Total_Raw                        42.0  0.205
FGC-FGC_CU                               17.9  0.198
BIA-BIA_BMI                              21.6  0.152
FGC-FGC_PU                               18.1  0.146
Physical-BMI                             10.3  0.124
FGC-FGC_GSD                              31.6  0.111
Physical-Systolic_BP                     11.0  0.110
Physical-Waist_Circumference             33.5  0.109
BIA-BIA_Frame_num                        21.6  0.100
FGC-FGC_GSD_Zone                         31.7 -0.098
FGC-FGC_GSND                             31.6  0.098
FGC-FGC_GSND_Zone                        31.7 -0.095
Basic_Demos-Sex                           0.0 -0.084
FGC-FGC_SRL_Zone                         18.5 -0.075
BIA-BIA_FMI                              21.6  0.061
FGC-FGC_SRR_Zone                         18.5 -0.061
FGC-FGC_TL_Zone                          18.3 -0.060
BIA-BIA_ICW                              21.6  0.054
Physical-Diastolic_BP                    11.0  0.053
FGC-FGC_PU_Zone                          18.5  0.051
BIA-BIA_FFMI                             21.6  0.048
BIA-BIA_FFM                              21.6  0.044
FGC-FGC_TL                               17.9  0.042
Fitness_Endurance-Max_Stage              35.2 -0.031
Physical-HeartRate                       10.9 -0.031
FGC-FGC_SRR                              18.1 -0.030
BIA-BIA_DEE                              21.6  0.029
BIA-BIA_LST                              21.6  0.026
BIA-BIA_Activity_Level_num               21.6  0.026
CGAS-CGAS_Score                          16.9 -0.025
Fitness_Endurance-Time_Mins              35.2  0.020
BIA-BIA_TBW                              21.6  0.020
FGC-FGC_SRL                              18.1 -0.020
PAQ_A-PAQ_A_Total                        38.2  0.019
BIA-BIA_SMM                              21.6  0.019
BIA-BIA_ECW                              21.6  0.019
FGC_FitnessScore                          0.2 -0.017
BIA-BIA_Fat                              21.6  0.014
BIA-BIA_BMR                              21.6  0.014
BIA-BIA_BMC                              21.6 -0.014
BIA-BIA_LDM                              21.6  0.013
Fitness_Endurance-Time_Sec               35.2 -0.013
FGC-FGC_CU_Zone                          18.4  0.007
PAQ_C-PAQ_C_Total                        24.5 -0.002
Important finding — the correlations in raw (pre-preprocessing) vs imputed data differ a lot. Key columns the user cares about:

Column	miss%	r (raw)	r (imputed)	Notes
FGC_FitnessScore	0.2%	-0.017	n/a	Near-zero signal
BIA-BIA_BMI	21.6%	0.152	n/a (dropped)	Slightly better than Physical-BMI
Physical-BMI	10.3%	0.124	0.122	
FGC-FGC_CU	17.9%	0.198	0.196	
FGC_FitnessScore is essentially useless — r=-0.017 against sii. The zone columns it sums are all weak and negative (kids in the healthy zone correlate slightly negatively with internet addiction, but the effect is tiny and mixed). The raw counts like FGC-FGC_CU are far stronger.

BIA-BIA_BMI over Physical-BMI: slightly better correlation (0.152 vs 0.124), same metric from a different instrument. Fine to keep — just remove it from the dup_drop list in cell 38.

Here's a clean overview of all available columns grouped by category, with missing rate and signal:


DEMOGRAPHICS         miss%   r_sii
Basic_Demos-Age        0%    0.21   ★★★
Basic_Demos-Sex        0%   -0.08   ★

PHYSICAL             miss%   r_sii
Physical-Height       10%    0.24   ★★★
Physical-Weight       10%    0.22   ★★★
Physical-BMI          10%    0.12   ★★  (or BIA-BIA_BMI at 21.6%, r=0.15)
Physical-Waist_Circ   34%    0.11   ★   (too much missing)
Physical-Systolic_BP  11%    0.11   ★★
Physical-Diastolic_BP 11%    0.05   ─
Physical-HeartRate    11%   -0.03   ─

BIA                  miss%   r_sii
BIA-BIA_FFM          21.6%   0.04*  (0.23 imputed)  ★★★
BIA-BIA_FFMI         21.6%   0.05*  (0.15 imputed)  ★★
BIA-BIA_Fat          21.6%   0.01*  (0.17 imputed)  ★★
BIA-BIA_BMI          21.6%   0.15   ★★
BIA-BIA_FMI          21.6%   0.06*                  ★
(rest of BIA: near-zero, highly redundant with FFM)

FGC                  miss%   r_sii
FGC-FGC_CU           17.9%   0.20   ★★★
FGC-FGC_PU           18.1%   0.15   ★★
FGC_FitnessScore      0.2%  -0.02   ✗ near-zero
FGC-FGC_GSD/GSND     31.6%   0.10   ★  (too much missing)
Zone cols             18-32%  -0.06 to -0.10  ─

SLEEP                miss%   r_sii
SDS-SDS_Total_T       42%    0.21   ✗ too much missing

INTERNET             miss%   r_sii
computerinternet_hrs   7%    0.22   ★★★

CGAS                 miss%   r_sii
CGAS-CGAS_Score       17%   -0.03   ✗ near-zero
*BIA raw correlations look lower due to outliers — preprocessing cleans them up significantly

Suggested cols:


cols = [
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Physical-Height',
    'Physical-Weight',
    'BIA-BIA_BMI',       # keep from cell 38 drop list; better than Physical-BMI
    'BIA-BIA_FFM',
    'FGC-FGC_CU',        # instead of FGC_FitnessScore (r=0.20 vs -0.02)
    'PreInt_EduHx-computerinternet_hoursday',
    'sii',
]
To make BIA-BIA_BMI work you just need to remove it from dup_drop in cell 38. FGC_FitnessScore I'd drop entirely — the zone-based composite just isn't predictive.

i understand, but aren't we dropping too much? complex models will be used, such as forests, ensemble, etc
Good point. For tree-based models and ensembles, the usual reasons to drop features (multicollinearity, curse of dimensionality) largely don't apply. Trees split on one feature at a time, handle correlated features fine, and built-in feature importance will downweight noise naturally.

The real exclusion criteria for these models are:

Near-zero signal — won't get split on anyway, just noise
Too much missing (>35%) — imputed values are mostly synthetic, potentially misleading
By those criteria on the imputed data (27 cols):

Bash
IN
python3 -c "
import pandas as pd
df = pd.read_csv('CMI/processed_data/cmi_imputed.csv')
corr = df.corr()['sii'].drop('sii').sort_values(key=abs, ascending=False)
raw = pd.read_csv('CMI/base_data/cmi_internet.csv')
miss = (raw.isnull().sum() / len(raw) * 100).round(1)
for col in corr.index:
    m = miss.get(col, 0)
    flag = '  << drop (near-zero)' if abs(corr[col]) < 0.05 else ''
    flag = flag or ('  << borderline missing' if m > 35 else '')
    print(f'{m:5.1f}% miss  r={corr[col]:+.3f}  {col}{flag}')
"

OUT
 10.2% miss  r=+0.229  Physical-Height
 21.6% miss  r=+0.226  BIA-BIA_FFM
  9.7% miss  r=+0.220  Physical-Weight
  0.0% miss  r=+0.212  Basic_Demos-Age
  7.2% miss  r=+0.207  PreInt_EduHx-computerinternet_hoursday
 17.9% miss  r=+0.196  FGC-FGC_CU
 42.0% miss  r=+0.171  SDS-SDS_Total_T  << borderline missing
 21.6% miss  r=+0.168  BIA-BIA_Fat
 21.6% miss  r=+0.153  BIA-BIA_FFMI
 18.1% miss  r=+0.143  FGC-FGC_PU
 33.5% miss  r=+0.138  Physical-Waist_Circumference
 21.6% miss  r=+0.130  BIA-BIA_FMI
 10.3% miss  r=+0.123  Physical-BMI
 11.0% miss  r=+0.112  Physical-Systolic_BP
 21.6% miss  r=+0.112  BIA-BIA_BMC
 31.6% miss  r=+0.108  FGC-FGC_GSD
 31.6% miss  r=+0.100  FGC-FGC_GSND
 21.6% miss  r=+0.086  BIA-BIA_Frame_num
  0.0% miss  r=-0.084  Basic_Demos-Sex
 18.5% miss  r=-0.078  FGC-FGC_SRL_Zone
 18.5% miss  r=-0.065  FGC-FGC_SRR_Zone
 18.3% miss  r=-0.054  FGC-FGC_TL_Zone
 11.0% miss  r=+0.050  Physical-Diastolic_BP
 10.9% miss  r=-0.035  Physical-HeartRate  << drop (near-zero)
 16.9% miss  r=-0.033  CGAS-CGAS_Score  << drop (near-zero)
 35.2% miss  r=-0.016  Fitness_Endurance-Max_Stage  << drop (near-zero)
For ensemble/forest models, the recommendation is: keep almost everything from the imputed data, only hard-dropping the 3 near-zero columns:


cols = [c for c in df_imputed.columns if c not in [
    'Physical-HeartRate',       # r=-0.035
    'CGAS-CGAS_Score',          # r=-0.033
    'Fitness_Endurance-Max_Stage', # r=-0.016, also 35% missing
]]
# → 24 features + sii
On SDS (42% missing): it's a judgment call. The linear correlation is 0.171, which is real signal — sleep disruption and internet addiction are genuinely linked. If you include it, the model is partly learning from imputed values. If you exclude it, you lose that signal entirely. For a forest, I'd lean toward keeping it and letting the model weight it.

On FGC_FitnessScore: still not worth adding — it has near-zero correlation even in clean data. The individual FGC-FGC_CU and FGC-FGC_PU already in the imputed data capture the same info better.

On BIA-BIA_BMI: to add it back, remove it from dup_drop in cell 38 and add it to the MICE categorical/continuous handling in cell 43. Since Physical-BMI is already there, having both is fine for a forest — it'll just use whichever split is cleaner.

do not consider the imputed datset, just the base, what my code does and my considerations. data imputed was a prev  version
Bash
IN
python3 -c "
import json
with open('CMI/src/module_0.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells'][:25]):
    src = ''.join(cell['source'])
    if src.strip():
        print(f'=== Cell {i} ({cell[\"cell_type\"]}) ===')
        print(src[:800])
        print()
"

OUT
=== Cell 0 (markdown) ===
# Module 0

## Dataset: Child Mind Institute — Problematic Internet Use

Explore and analyse the dataset to understand its structure and key characteristics (describe
them); Conduct data pre-processing (i.e., encoding categorical variables, feature scaling,
missing values imputation); Create new variables if needed.

=== Cell 1 (code) ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_columns", 90)

df = pd.read_csv("../base_data/cmi_internet.csv")

# General info: dtypes and non-null counts
df.info(verbose=True, show_counts=True)
df

=== Cell 2 (code) ===
# Quick descriptive statistics for all numeric columns
df.describe().T

=== Cell 3 (markdown) ===
---
## Step 1. Missing values Overview


We start by going over missing values in order to get an idea of what we're working with

=== Cell 4 (code) ===
# Missing values table (only columns with missingness)
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(1)
miss_df = (pd.DataFrame({"missing": miss, "pct": miss_pct})
             .query("missing > 0")
             .sort_values("pct", ascending=False))
miss_df

=== Cell 5 (markdown) ===
77 of the columns present missing values, especially those that are in the PCIAT instrument.

This group of variables will be addressed first after data visualization, then moving on to the rest.

=== Cell 6 (markdown) ===
---
## Step 2. Data visualization

We explore each feature's distribution to spot outliers and wrong data.

=== Cell 7 (code) ===
# Histograms of key continuous features (non-PCIAT, non-Season)
cont_cols = [
    "Basic_Demos-Age", "CGAS-CGAS_Score",
    "Physical-BMI", "Physical-Height", "Physical-Weight",
    "Physical-Waist_Circumference", "Physical-Diastolic_BP",
    "Physical-HeartRate", "Physical-Systolic_BP",
    "Fitness_Endurance-Max_Stage", "Fitness_Endurance-Time_Mins",
    "Fitness_Endurance-Time_Sec",
    "FGC-FGC_CU", "FGC-FGC_GSND", "FGC-FGC_GSD",
    "FGC-FGC_PU", "FGC-FGC_SRL", "FGC-FGC_SRR", "FGC-FGC_TL",
    "BIA-BIA_BMI", "BIA-BIA_BMR", "BIA-BIA_Fat", "BIA-BIA_FFM",
    "BIA-BIA_FMI", "BIA-BIA_FFMI", "BIA-BIA_SMM", "BIA-BIA_TBW",
    "BIA-BIA_DEE", "BIA-BIA_BMC", "BIA-BIA_ECW", "BIA-BIA_ICW",
    "BIA-BIA_LDM", "BIA-BIA_LST",
    "PAQ_A-PAQ_A_Total", "PAQ_C-PAQ_C_Total",
    "SDS-SDS_Total_Raw", "S

=== Cell 8 (markdown) ===
Observations on continuous features:
- CGAS_Score has a spike at 999, clearly an error, as scale is 1–100.
- Physical-BMI and Physical-Weight have values at exactly 0 which is biologically impossible, likely missing data encoded as zero.
- BIA features (BMR, FFM, DEE, ECW, etc.) show extreme right-tail outliers orders of magnitude above the median, possible measurement errors.
- BIA-BIA_Fat has negative values which is physically impossible for body fat percentage.

=== Cell 9 (code) ===
# Bar plots for categorical / ordinal features
cat_cols = [
    "Basic_Demos-Sex",
    "FGC-FGC_CU_Zone", "FGC-FGC_PU_Zone", "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR_Zone", "FGC-FGC_TL_Zone",
    "FGC-FGC_GSND_Zone", "FGC-FGC_GSD_Zone",
    "BIA-BIA_Activity_Level_num", "BIA-BIA_Frame_num",
    "PreInt_EduHx-computerinternet_hoursday",
]

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    df[col].dropna().astype(int).value_counts().sort_index().plot.bar(
        ax=axes[i], edgecolor="k", alpha=0.8)
    axes[i].set_title(col.split("-")[-1], fontsize=9)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Categorical / ordinal feature distributions", fontsize=13, y=1.01)
p

=== Cell 10 (markdown) ===
Observations — categorical/ordinal features:
- Sex: 60% male, 40% female — moderate imbalance.
- Fitness zones (CU, PU, SRL, SRR, TL): most are binary (0 = Needs Improvement, 1 = Healthy). Push-up zone is heavily skewed toward 0 (most children need improvement), while Trunk Lift zone is heavily skewed toward 1 (most in healthy zone).
- Grip Strength zones (GSND, GSD): 3-level ordinal (Weak/Normal/Strong), concentrated around Normal (2).
- Activity Level (BIA): 5-level ordinal, peaks at levels 2–3 (Light/Moderate).
- Internet hours/day: ordinal 0–3, majority at 0 (< 1h/day).

=== Cell 11 (code) ===
# Box plots to spot outliers
bia_cols = [c for c in df.columns if c.startswith("BIA-") and "Season" not in c]
fig, axes = plt.subplots(3, 6, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(bia_cols):
    df[col].dropna().plot.box(ax=axes[i], vert=True)
    axes[i].set_title(col.replace("BIA-BIA_", ""), fontsize=9)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("BIA features — box plots (note extreme outliers)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

=== Cell 12 (markdown) ===
**Observations — BIA outliers:**

The box plots confirm that several BIA variables have extreme outliers far beyond the whiskers (e.g. BMR up to ~83,000 vs a median ~1,200; FFM up to ~8,800 vs a median ~30). These are almost certainly recording errors and will need to be addressed during cleaning.

=== Cell 13 (markdown) ===
---
## Step 3. Linear Correlation

Now we look at how features relate to each other and to the target `sii`.

=== Cell 14 (code) ===
# Correlation with sii — only non-PCIAT, non-Season features
numeric = df.select_dtypes("number").drop(columns=["id"])
pciat_cols = [c for c in numeric.columns if "PCIAT" in c]
corr_sii = (numeric.drop(columns=pciat_cols)
            .corrwith(numeric["sii"])
            .drop("sii")
            .sort_values(key=abs, ascending=False))

fig, ax = plt.subplots(figsize=(10, 8))
corr_sii.plot.barh(ax=ax, color=corr_sii.apply(lambda x: "steelblue" if x >= 0 else "salmon"))
ax.set_title("Correlation with sii (non-PCIAT features)")
ax.set_xlabel("Pearson r")
ax.axvline(0, color="k", lw=0.5)
plt.tight_layout()
plt.show()

print("Top 10 absolute correlates:")
print(corr_sii.head(10).to_string())

=== Cell 15 (code) ===
# Correlation heatmap among feature groups (Physical + BIA + FGC + Demographics)
keep_for_heatmap = [
    "Basic_Demos-Age", "Basic_Demos-Sex",
    "Physical-BMI", "Physical-Height", "Physical-Weight",
    "Physical-Waist_Circumference", "Physical-Diastolic_BP",
    "Physical-HeartRate", "Physical-Systolic_BP",
    "BIA-BIA_BMI", "BIA-BIA_Fat", "BIA-BIA_FFM", "BIA-BIA_FMI",
    "BIA-BIA_FFMI", "BIA-BIA_SMM", "BIA-BIA_TBW", "BIA-BIA_BMR",
    "BIA-BIA_DEE",
    "SDS-SDS_Total_T", "PreInt_EduHx-computerinternet_hoursday",
    "sii",
]

corr_mat = df[keep_for_heatmap].corr()
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.3, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation heatmap — Phy

=== Cell 16 (markdown) ===
Observations — correlations:
- The strongest non-PCIAT correlates with `sii` are modest: Height (0.24), Internet hours/day (0.22), Weight (0.22), Sleep T-score (0.21), Age (0.21).
- BIA and Physical features are highly correlated with each other (e.g. FFM ↔ TBW ↔ SMM ↔ LST, Physical-BMI ↔ BIA-BMI). This multicollinearity should be kept in mind during modelling.
- Sleep disturbance (SDS_Total_T) positively correlates with `sii`, more sleep problems associate with more internet use problems.

=== Cell 17 (markdown) ===
---
## Step 4. Target Variable Analysis

Having explored the features, we now examine the target variable `sii` (Severity Impairment Index) and its relationship with the PCIAT instrument.

=== Cell 18 (code) ===
sii_labels = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Class distribution
counts = df["sii"].value_counts().sort_index()
counts.index = counts.index.map(lambda x: f"{int(x)} – {sii_labels[int(x)]}")
counts.plot.bar(ax=axes[0], color=sns.color_palette("coolwarm", 4), edgecolor="k")
axes[0].set_title("sii class distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts):
    axes[0].text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

# sii vs PCIAT_Total — show derivation
mask = df["PCIAT-PCIAT_Total"].notna()
axes[1].scatter(df.loc[mask, "PCIAT-PCIAT_Total"], df.loc[mask, "sii"],
                alpha=0.3, s=10, c="steelblue")
for thr in [30, 49, 79]:
    axes[1].axvline(thr, color="red", ls="--"

=== Cell 19 (markdown) ===
Key finding: `sii` is derived directly from `PCIAT_Total`.

The scatter plot reveals a deterministic step-function relationship. According to the data dictionary, `sii` is the Severity Impairment Index computed by binning `PCIAT_Total` with thresholds from the official PCIAT scoring guide:

| PCIAT Total | sii | Label |
|---|---|---|
| 0–30 | 0 | None |
| 31–49 | 1 | Mild |
| 50–79 | 2 | Moderate |
| 80–100 | 3 | Severe |

Implication for modelling: All 22 PCIAT columns (20 items + Total + Season) must be dropped from the feature set. Using them would constitute data leakage.

Target imbalance: The target is heavily skewed — 68.9% of participants fall in class 0 (None) and only 1.0% in class 3 (Severe). This will need to be addressed during modelling (e.g. class weights, oversampling).

=== Cell 20 (markdown) ===
---
## Step 5. Data Cleaning

Based on the previous observations, we now address: erroneous values, column drops (PCIAT leakage, Season low-signal), and extreme BIA outliers.

=== Cell 21 (code) ===
# --- 5a. Drop columns: id, all PCIAT, all Season columns ---
season_cols = [c for c in df.columns if "Season" in c]
pciat_cols_all = [c for c in df.columns if "PCIAT" in c]

print(f"Dropping {len(season_cols)} Season columns: {season_cols}")
print(f"Dropping {len(pciat_cols_all)} PCIAT columns")
print("Dropping: id")

df_clean = df.drop(columns=["id"] + season_cols + pciat_cols_all)
print(f"\nShape after drops: {df_clean.shape}")

=== Cell 22 (code) ===
# --- 5b. Fix erroneous / impossible values → NaN ---

fixes = {}

# CGAS = 999 (scale is 1–100)
mask = df_clean["CGAS-CGAS_Score"] > 100
fixes["CGAS-CGAS_Score == 999"] = mask.sum()
df_clean.loc[mask, "CGAS-CGAS_Score"] = np.nan

# Physical-BMI = 0
mask = df_clean["Physical-BMI"] <= 0
fixes["Physical-BMI == 0"] = mask.sum()
df_clean.loc[mask, "Physical-BMI"] = np.nan

# Physical-Weight = 0
mask = df_clean["Physical-Weight"] <= 0
fixes["Physical-Weight == 0"] = mask.sum()
df_clean.loc[mask, "Physical-Weight"] = np.nan

# Diastolic_BP = 0
mask = df_clean["Physical-Diastolic_BP"] <= 0
fixes["Physical-Diastolic_BP == 0"] = mask.sum()
df_clean.loc[mask, "Physical-Diastolic_BP"] = np.nan

# Systolic_BP = 0
mask = df_clean["Physical-Systolic_BP"] <= 0
fixes["Physical-Systolic_BP == 0"] = mask.su

=== Cell 23 (code) ===
# --- 5c. Handle BIA outliers ---
# BIA features have extreme values that are clearly measurement/recording errors
# (e.g. BMR = 83,000 vs median ~1,200; FFM = 8,800 vs median ~30).
#
# Strategy:
#   1. REMOVE rows with truly extreme BIA values (beyond 10×IQR) — these are
#      impossible measurements, not real biological variation.
#   2. CAP remaining moderate outliers at the 1st/99th percentile (winsorization)
#      to reduce their influence without losing data.

bia_continuous = [c for c in df_clean.columns
                  if c.startswith("BIA-") and c not in
                  ["BIA-BIA_Activity_Level_num", "BIA-BIA_Frame_num"]]

# Step 1: identify rows with extreme BIA values (10×IQR)
extreme_mask = pd.Series(False, index=df_clean.index)
for col in bia_continuous:
    q1 = df_clea

=== Cell 24 (markdown) ===
---
## Step 6. Encoding Categorical Variables

All categoricals are already numerically encoded:
- `Basic_Demos-Sex` (0/1) — binary, keep as-is
- `FGC-*_Zone` (0/1) — binary, keep as-is  
- `FGC-FGC_GSND_Zone`, `FGC-FGC_GSD_Zone` (1/2/3) — ordinal (Weak < Normal < Strong), keep numeric
- `BIA-BIA_Activity_Level_num` (1–5) — ordinal, keep numeric
- `BIA-BIA_Frame_num` (1/2/3) — ordinal, keep numeric
- `PreInt_EduHx-computerinternet_hoursday` (0–3) — ordinal, keep numeric

No encoding transformations needed, all variables are already in a suitable numeric format.
Bash
IN
python3 -c "
import json
with open('CMI/src/module_0.ipynb') as f:
    nb = json.load(f)
for i in [22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
    cell = nb['cells'][i]
    src = ''.join(cell['source'])
    if src.strip():
        print(f'=== Cell {i} ({cell[\"cell_type\"]}) ===')
        print(src[:1200])
        print()
"

OUT
=== Cell 22 (code) ===
# --- 5b. Fix erroneous / impossible values → NaN ---

fixes = {}

# CGAS = 999 (scale is 1–100)
mask = df_clean["CGAS-CGAS_Score"] > 100
fixes["CGAS-CGAS_Score == 999"] = mask.sum()
df_clean.loc[mask, "CGAS-CGAS_Score"] = np.nan

# Physical-BMI = 0
mask = df_clean["Physical-BMI"] <= 0
fixes["Physical-BMI == 0"] = mask.sum()
df_clean.loc[mask, "Physical-BMI"] = np.nan

# Physical-Weight = 0
mask = df_clean["Physical-Weight"] <= 0
fixes["Physical-Weight == 0"] = mask.sum()
df_clean.loc[mask, "Physical-Weight"] = np.nan

# Diastolic_BP = 0
mask = df_clean["Physical-Diastolic_BP"] <= 0
fixes["Physical-Diastolic_BP == 0"] = mask.sum()
df_clean.loc[mask, "Physical-Diastolic_BP"] = np.nan

# Systolic_BP = 0
mask = df_clean["Physical-Systolic_BP"] <= 0
fixes["Physical-Systolic_BP == 0"] = mask.sum()
df_clean.loc[mask, "Physical-Systolic_BP"] = np.nan

# BIA-BIA_Fat < 0 (negative body fat impossible)
mask = df_clean["BIA-BIA_Fat"] <= 0
fixes["BIA-BIA_Fat < 0"] = mask.sum()
df_clean.loc[mask, "BIA-BIA_Fat"] = np.nan

# BIA-BIA_FMI < 0
mask = df_clean["BIA-BIA_FMI"] <= 0
fixes["BIA-BIA_FMI < 0"] = mask.sum()
df_clean.loc[mask, "BIA-BIA_FMI"] = np.nan

# BIA-BIA_ECW < 0 (negative extracellula

=== Cell 23 (code) ===
# --- 5c. Handle BIA outliers ---
# BIA features have extreme values that are clearly measurement/recording errors
# (e.g. BMR = 83,000 vs median ~1,200; FFM = 8,800 vs median ~30).
#
# Strategy:
#   1. REMOVE rows with truly extreme BIA values (beyond 10×IQR) — these are
#      impossible measurements, not real biological variation.
#   2. CAP remaining moderate outliers at the 1st/99th percentile (winsorization)
#      to reduce their influence without losing data.

bia_continuous = [c for c in df_clean.columns
                  if c.startswith("BIA-") and c not in
                  ["BIA-BIA_Activity_Level_num", "BIA-BIA_Frame_num"]]

# Step 1: identify rows with extreme BIA values (10×IQR)
extreme_mask = pd.Series(False, index=df_clean.index)
for col in bia_continuous:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 10 * iqr
    upper = q3 + 10 * iqr
    extreme_mask |= df_clean[col].notna() & ((df_clean[col] < lower) | (df_clean[col] > upper))

n_extreme = extreme_mask.sum()
print(f"Rows with extreme BIA outliers (10×IQR): {n_extreme} ({n_extreme/len(df_clean)*100:.2f}%)")
df_clean = df_clean[~extreme_mask].reset_i

=== Cell 25 (markdown) ===
---
## Step 7. Feature Engineering

=== Cell 26 (markdown) ===
The seven FGC Zone columns classify fitness outcomes per exercise:
- Five columns (`CU`, `PU`, `SRL`, `SRR`, `TL`) use a **binary** encoding: 0 = below Healthy Fitness Zone, 1 = in Healthy Fitness Zone.
- Two columns (`GSND`, `GSD`) use a **ternary** encoding: 1 = Weak, 2 = Normal, 3 = Strong.

The ternary columns are recoded to the same binary scale (1 → 0; 2 or 3 → 1), then all seven are summed into `FGC_FitnessScore` (range 0–7): higher = more fitness zones met.

=== Cell 27 (code) ===
fgc_zone_cols = [
    "FGC-FGC_CU_Zone",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD_Zone",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL_Zone",
]

# Recode ternary columns (1=Weak, 2=Normal, 3=Strong) -> binary (0 / 1)
ternary_cols = ["FGC-FGC_GSND_Zone", "FGC-FGC_GSD_Zone"]
for col in ternary_cols:
    df_clean[col] = df_clean[col].map({1.0: 0, 2.0: 1, 3.0: 1})

# Fitness score = count of zones met (0-7)
df_clean["FGC_FitnessScore"] = df_clean[fgc_zone_cols].sum(axis=1, min_count=1)

print(f"FGC_FitnessScore - non-null: {df_clean['FGC_FitnessScore'].notna().sum()}, "
      f"null: {df_clean['FGC_FitnessScore'].isna().sum()}")
df_clean["FGC_FitnessScore"].value_counts().sort_index()

=== Cell 28 (markdown) ===
`PAQ_A-PAQ_A_Total` refers to Activity Summary Score for adolescents, meanwhile `PAQ_C-PAQ_C_Total` refers to children.

These metrics are almost always mutually exclusive, therefore they are merged into one.

When both are present, the average is used.

=== Cell 29 (code) ===
paq_a = df_clean["PAQ_A-PAQ_A_Total"]
paq_c = df_clean["PAQ_C-PAQ_C_Total"]

df_clean["PAQ_Combined"] = np.where(
    paq_a.notna() & paq_c.notna(), (paq_a + paq_c) / 2,
    np.where(paq_a.notna(), paq_a, paq_c)
)

# Drop originals
df_clean = df_clean.drop(columns=["PAQ_A-PAQ_A_Total", "PAQ_C-PAQ_C_Total",
                                   "PAQ_A-Season", "PAQ_C-Season"], errors="ignore")

print(f"PAQ_Combined — non-null: {df_clean['PAQ_Combined'].notna().sum()}, "
      f"null: {df_clean['PAQ_Combined'].isna().sum()}")
df_clean["PAQ_Combined"].describe()

=== Cell 30 (markdown) ===
`Fitness_Endurance_TotalTime` is calculated as the total number of seconds, using `Mins` and `Sec`.

=== Cell 31 (code) ===
mins = df_clean["Fitness_Endurance-Time_Mins"]
secs = df_clean["Fitness_Endurance-Time_Sec"]
either_present = mins.notna() | secs.notna()

df_clean["Fitness_Endurance_TotalTime"] = np.where(
    either_present,
    mins.fillna(0) * 60 + secs.fillna(0),
    np.nan
)

# Drop the two original time columns (keep Max_Stage)
df_clean = df_clean.drop(columns=["Fitness_Endurance-Time_Mins",
                                   "Fitness_Endurance-Time_Sec"])

non_null = df_clean["Fitness_Endurance_TotalTime"].notna().sum()
null = df_clean["Fitness_Endurance_TotalTime"].isna().sum()
print(f"Fitness_Endurance_TotalTime — non-null: {non_null}, null: {null} ({null/len(df_clean)*100:.1f}%)")
df_clean["Fitness_Endurance_TotalTime"].describe()

=== Cell 32 (markdown) ===
---
## Step 8. Unit Conversion (Imperial → Metric)

The Physical measures use imperial units. We convert them to metric for easier interpretation:
- Height: inches → centimetres (× 2.54)
- Weight: pounds → kilograms (× 0.4536)
- Waist Circumference: inches → centimetres (× 2.54)

BMI is already in kg/m² (universal) — no conversion needed.

=== Cell 33 (code) ===
# Convert imperial → metric
df_clean["Physical-Height"] = df_clean["Physical-Height"] * 2.54          # in → cm
df_clean["Physical-Weight"] = df_clean["Physical-Weight"] * 0.453592      # lbs → kg
df_clean["Physical-Waist_Circumference"] = df_clean["Physical-Waist_Circumference"] * 2.54  # in → cm

print("Converted to metric:")
print(f"  Height: {df_clean['Physical-Height'].describe()[['mean','min','max']].to_string()}")
print(f"  Weight: {df_clean['Physical-Weight'].describe()[['mean','min','max']].to_string()}")
print(f"  Waist:  {df_clean['Physical-Waist_Circumference'].describe()[['mean','min','max']].to_string()}")
Now I can simulate df_clean exactly as the notebook does at cell 35.

Bash
IN
python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('CMI/base_data/cmi_internet.csv')

# Cell 21: drop id, Season, PCIAT
drop = ['id'] + [c for c in df.columns if 'Season' in c or 'PCIAT' in c]
df_clean = df.drop(columns=drop)

# Cell 22: fix impossible values
df_clean.loc[df_clean['CGAS-CGAS_Score'] > 100, 'CGAS-CGAS_Score'] = np.nan
df_clean.loc[df_clean['Physical-BMI'] <= 0, 'Physical-BMI'] = np.nan
df_clean.loc[df_clean['Physical-Weight'] <= 0, 'Physical-Weight'] = np.nan
df_clean.loc[df_clean['Physical-Diastolic_BP'] <= 0, 'Physical-Diastolic_BP'] = np.nan
df_clean.loc[df_clean['Physical-Systolic_BP'] <= 0, 'Physical-Systolic_BP'] = np.nan
df_clean.loc[df_clean['BIA-BIA_Fat'] <= 0, 'BIA-BIA_Fat'] = np.nan
df_clean.loc[df_clean['BIA-BIA_FMI'] <= 0, 'BIA-BIA_FMI'] = np.nan

# Cell 23: remove rows with extreme BIA outliers (10xIQR)
bia_cont = [c for c in df_clean.columns if c.startswith('BIA-') and c not in ['BIA-BIA_Activity_Level_num','BIA-BIA_Frame_num']]
extreme_mask = pd.Series(False, index=df_clean.index)
for col in bia_cont:
    q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    iqr = q3 - q1
    extreme_mask |= df_clean[col].notna() & ((df_clean[col] < q1 - 10*iqr) | (df_clean[col] > q3 + 10*iqr))
df_clean = df_clean[~extreme_mask].reset_index(drop=True)

# Cell 27: FGC_FitnessScore
zone_cols = ['FGC-FGC_CU_Zone','FGC-FGC_GSND_Zone','FGC-FGC_GSD_Zone','FGC-FGC_PU_Zone','FGC-FGC_SRL_Zone','FGC-FGC_SRR_Zone','FGC-FGC_TL_Zone']
for col in ['FGC-FGC_GSND_Zone','FGC-FGC_GSD_Zone']:
    df_clean[col] = df_clean[col].map({1.0:0, 2.0:1, 3.0:1})
df_clean['FGC_FitnessScore'] = df_clean[zone_cols].sum(axis=1, min_count=1)

# Cell 29: PAQ_Combined
paq_a, paq_c = df_clean['PAQ_A-PAQ_A_Total'], df_clean['PAQ_C-PAQ_C_Total']
df_clean['PAQ_Combined'] = np.where(paq_a.notna() & paq_c.notna(), (paq_a+paq_c)/2, np.where(paq_a.notna(), paq_a, paq_c))
df_clean = df_clean.drop(columns=['PAQ_A-PAQ_A_Total','PAQ_C-PAQ_C_Total'], errors='ignore')

# Cell 31: Fitness_Endurance_TotalTime
mins, secs = df_clean['Fitness_Endurance-Time_Mins'], df_clean['Fitness_Endurance-Time_Sec']
df_clean['Fitness_Endurance_TotalTime'] = np.where(mins.notna()|secs.notna(), mins.fillna(0)*60+secs.fillna(0), np.nan)
df_clean = df_clean.drop(columns=['Fitness_Endurance-Time_Mins','Fitness_Endurance-Time_Sec'])

# Cell 33: unit conversion
df_clean['Physical-Height'] *= 2.54
df_clean['Physical-Weight'] *= 0.453592
df_clean['Physical-Waist_Circumference'] *= 2.54

# Now at cell 35: compute missing% and corr with sii
n = len(df_clean)
miss = (df_clean.isnull().sum() / n * 100).round(1)
corr = df_clean.select_dtypes('number').corrwith(df_clean['sii']).round(3)

summary = pd.DataFrame({'miss%': miss, 'r_sii': corr}).dropna(subset=['r_sii'])
summary['abs_r'] = summary['r_sii'].abs()
summary = summary.sort_values('abs_r', ascending=False)
print(f'df_clean shape at cell 35: {df_clean.shape}')
print()
print(summary[['miss%','r_sii']].to_string())
"

OUT
df_clean shape at cell 35: (8243, 48)

                                        miss%  r_sii
sii                                       0.0  1.000
BIA-BIA_FFM                              21.7  0.239
BIA-BIA_LST                              21.7  0.239
Physical-Height                          10.4  0.238
BIA-BIA_ICW                              21.8  0.227
Physical-Weight                          10.8  0.220
PreInt_EduHx-computerinternet_hoursday    7.2  0.215
BIA-BIA_BMR                              21.6  0.214
Basic_Demos-Age                           0.0  0.212
BIA-BIA_TBW                              21.8  0.211
SDS-SDS_Total_T                          42.0  0.210
BIA-BIA_DEE                              21.6  0.209
BIA-BIA_SMM                              21.6  0.200
FGC-FGC_CU                               18.0  0.200
SDS-SDS_Total_Raw                        42.0  0.196
BIA-BIA_ECW                              21.6  0.194
BIA-BIA_LDM                              21.7  0.187
BIA-BIA_Fat                              22.9  0.163
BIA-BIA_FFMI                             21.7  0.154
BIA-BIA_BMI                              21.6  0.152
FGC-FGC_PU                               18.1  0.148
Physical-BMI                             10.5  0.126
BIA-BIA_FMI                              22.1  0.119
Physical-Systolic_BP                     11.1  0.111
FGC-FGC_GSD                              31.6  0.111
Physical-Waist_Circumference             33.7  0.108
FGC-FGC_GSD_Zone                         32.0 -0.101
BIA-BIA_Frame_num                        21.7  0.100
FGC-FGC_GSND                             31.6  0.100
FGC-FGC_GSND_Zone                        32.0 -0.096
Basic_Demos-Sex                           0.0 -0.084
FGC-FGC_SRL_Zone                         18.6 -0.076
BIA-BIA_BMC                              21.9  0.071
FGC-FGC_SRR_Zone                         18.6 -0.060
FGC-FGC_TL_Zone                          18.4 -0.055
FGC-FGC_PU_Zone                          18.6  0.049
Physical-Diastolic_BP                    11.0  0.048
CGAS-CGAS_Score                          17.0 -0.045
FGC-FGC_TL                               18.0  0.044
Physical-HeartRate                       10.8 -0.033
Fitness_Endurance-Max_Stage              35.3 -0.032
FGC-FGC_SRR                              18.2 -0.029
BIA-BIA_Activity_Level_num               21.7  0.028
FGC-FGC_SRL                              18.3 -0.018
FGC_FitnessScore                          0.2 -0.016
PAQ_Combined                              9.1  0.008
FGC-FGC_CU_Zone                          18.4  0.005
Fitness_Endurance_TotalTime              14.5 -0.003
Here's the full picture of df_clean at cell 35, after all cleaning/feature engineering, with signal and missingness:


KEEP — strong signal, low/medium missing
─────────────────────────────────────────────────────────────────
  0.0%  r=+0.212  Basic_Demos-Age
  0.0%  r=-0.084  Basic_Demos-Sex
  7.2%  r=+0.215  PreInt_EduHx-computerinternet_hoursday
 10.4%  r=+0.238  Physical-Height
 10.8%  r=+0.220  Physical-Weight
 10.5%  r=+0.126  Physical-BMI            (or BIA-BIA_BMI at 0.152 ← better)
 11.1%  r=+0.111  Physical-Systolic_BP
 11.0%  r=+0.048  Physical-Diastolic_BP
 10.8%  r=-0.033  Physical-HeartRate       ← borderline, near-zero
 14.5%  r=-0.003  Fitness_Endurance_TotalTime  ← drop
 18.0%  r=+0.200  FGC-FGC_CU
 18.1%  r=+0.148  FGC-FGC_PU
 18.6%  r=-0.076  FGC-FGC_SRL_Zone
 18.6%  r=-0.060  FGC-FGC_SRR_Zone
 18.4%  r=-0.055  FGC-FGC_TL_Zone
 18.0%  r=+0.044  FGC-FGC_TL               ← zone is better
 18.3%  r=-0.018  FGC-FGC_SRL              ← zone is better
 18.2%  r=-0.029  FGC-FGC_SRR              ← zone is better
 18.4%  r=+0.005  FGC-FGC_CU_Zone          ← raw is better, drop
 18.6%  r=+0.049  FGC-FGC_PU_Zone          ← raw is better, drop
  0.2%  r=-0.016  FGC_FitnessScore         ← near-zero, DROP
 17.0%  r=-0.045  CGAS-CGAS_Score          ← near-zero
  9.1%  r=+0.008  PAQ_Combined             ← near-zero, drop
 21.6%  r=+0.239  BIA-BIA_FFM              ★ strongest
 21.7%  r=+0.239  BIA-BIA_LST              (r=0.91 with FFM — redundant)
 21.8%  r=+0.227  BIA-BIA_ICW              (r=0.85 with FFM — redundant)
 21.6%  r=+0.214  BIA-BIA_BMR              (r=0.92 with FFM — redundant)
 21.8%  r=+0.211  BIA-BIA_TBW              (r=0.88 with FFM — redundant)
 21.6%  r=+0.209  BIA-BIA_DEE              (derived from BMR)
 21.6%  r=+0.200  BIA-BIA_SMM              (r=0.84 with FFM — redundant)
 21.6%  r=+0.194  BIA-BIA_ECW              (r=0.86 with FFM — redundant)
 21.7%  r=+0.187  BIA-BIA_LDM              (r=0.82 with FFM — redundant)
 22.9%  r=+0.163  BIA-BIA_Fat
 21.7%  r=+0.154  BIA-BIA_FFMI
 21.6%  r=+0.152  BIA-BIA_BMI              ← better than Physical-BMI
 22.1%  r=+0.119  BIA-BIA_FMI
 21.7%  r=+0.100  BIA-BIA_Frame_num
 21.7%  r=+0.028  BIA-BIA_Activity_Level_num  ← near-zero
 21.9%  r=+0.071  BIA-BIA_BMC

HIGH MISSING — judgment call for forests
─────────────────────────────────────────────────────────────────
 31.6%  r=+0.111  FGC-FGC_GSD
 32.0%  r=-0.101  FGC-FGC_GSD_Zone
 31.6%  r=+0.100  FGC-FGC_GSND
 32.0%  r=-0.096  FGC-FGC_GSND_Zone
 33.7%  r=+0.108  Physical-Waist_Circumference
 35.3%  r=-0.032  Fitness_Endurance-Max_Stage  ← near-zero too
 42.0%  r=+0.210  SDS-SDS_Total_T          ← strong signal but 42% missing
On FGC_FitnessScore: r=-0.016 in the actual cleaned data. It's not a rounding error — the zone columns genuinely mix positive and negative signals when summed, and they cancel out. No model complexity will recover signal that isn't there. Drop it.

On BIA-BIA_BMI: r=0.152 vs Physical-BMI r=0.126. Worth keeping — just remove it from dup_drop in cell 38.

For forests/ensembles, what cell 38 over-drops:

The BIA lean-mass cluster (LST, ICW, BMR, TBW, SMM, ECW, LDM) — all have r≈0.19-0.24. They're highly correlated with FFM, but forests can still use them for splits. You could keep 1-2 of them (e.g. BIA-BIA_BMR as a metabolic rate proxy, distinct from pure mass).
The current cols approach of hand-picking 10 features is too aggressive for ensemble models.
Practical recommendation: instead of a fixed cols list, let cell 38 define the dropped set and pass everything remaining into MICE. That gives ~25 features, which is fine for forests.