# Data Mining 2 — Project Plan
## Dataset: Child Mind Institute — Problematic Internet Use

---

## 1. Dataset Overview

- **Source**: Child Mind Institute (CMI) study on children/adolescents' internet use and its relation to physical/psychological health.
- **Size**: 8,460 rows × 82 columns (81 features + 1 target).
- **Unit of observation**: One child/adolescent participant (identified by `id`).
- **Target variable**: `sii` (Severity Impairment Index) — an ordinal variable with 4 levels:
  | Value | Label    | Count | % of data |
  |-------|----------|-------|-----------|
  | 0     | None     | 5,833 | 68.9 %    |
  | 1     | Mild     | 1,587 | 18.8 %    |
  | 2     | Moderate |   952 | 11.3 %    |
  | 3     | Severe   |    88 |  1.0 %    |

  The target is **heavily imbalanced** (class 3 represents only 1 % of the data).
  `sii` is derived from `PCIAT-PCIAT_Total` via the Severity Impairment Index thresholds (0–30 → 0, 31–49 → 1, 50–79 → 2, 80–100 → 3).

---

## 2. Variable Groups (Instruments)

The 81 features come from **9 instruments**, each with its own Season column:

| Instrument | Prefix | # Vars | Key Variables | Type |
|---|---|---|---|---|
| **Demographics** | `Basic_Demos-` | 2 | Age (5–22, mean 10.2), Sex (0=M 60%, 1=F 40%) | mixed |
| **Children's Global Assessment Scale** | `CGAS-` | 1 | CGAS_Score (functional impairment, 25–100) | int |
| **Physical Measures** | `Physical-` | 8 | BMI, Height, Weight, Waist Circ., Diastolic/Systolic BP, HeartRate | float/int |
| **FitnessGram Treadmill** | `Fitness_Endurance-` | 4 | Max_Stage, Time_Mins, Time_Sec | int |
| **FitnessGram Child** | `FGC-` | 14 | Curl-ups, Grip Strength (dom/non-dom), Push-ups, Sit&Reach (L/R), Trunk Lift + their fitness zones | float + categorical |
| **Bio-electric Impedance (BIA)** | `BIA-` | 17 | BMC, BMI, BMR, DEE, Body Fat %, FFM, FMI, SMM, TBW, etc. + Activity Level, Frame | float + categorical |
| **Physical Activity Questionnaire** | `PAQ_A-` / `PAQ_C-` | 4 | PAQ_A_Total (adolescents), PAQ_C_Total (children) | float |
| **Parent-Child Internet Addiction Test** | `PCIAT-` | 22 | 20 individual items (0–5 Likert) + Total score | categorical int |
| **Sleep Disturbance Scale** | `SDS-` | 3 | Total Raw Score, Total T-Score | int |
| **Internet Use** | `PreInt_EduHx-` | 2 | computerinternet_hoursday (0–3 ordinal) | categorical int |

**Season columns** (11 total): each instrument has its own Season of participation (Spring/Summer/Fall/Winter). These will be **dropped** — they carry low predictive signal for internet addiction severity and would add 44 one-hot columns for minimal benefit.

---

## 3. Key Data Quality Issues Identified

### 3.1 Missing Values (significant)
| Severity | Columns | Missing % |
|---|---|---|
| **Very high** | All 22 PCIAT columns | **67.9 %** |
| **High** | SDS (Sleep) | 42.0 % |
| **High** | PAQ_A (Adolescent activity) | 38.2 % |
| **Medium-high** | Fitness Endurance | 35.2 % |
| **Medium** | Waist Circumference | 33.5 % |
| **Medium** | FGC Grip Strength zones | 31.7 % |
| **Medium** | BIA body composition | 21.6 % |
| **Medium** | PAQ_C (Child activity) | 24.5 % |
| **Low-medium** | FGC fitness tests | 17–18 % |
| **Low** | Physical measures (BP, HR) | 10–11 % |
| **Low** | Internet hours/day | 7.2 % |

**Note on PCIAT**: Since `sii` is directly derived from `PCIAT_Total`, the PCIAT items are **not independent predictors** — they are the source of the target itself. They must be **excluded from the feature set** to avoid data leakage, or treated very carefully.

### 3.2 Outliers / Erroneous Values
- **CGAS_Score = 999** (4 cases): impossible value (scale is 1–100) → treat as missing.
- **Physical-BMI = 0** (7 cases): biologically impossible → treat as missing.
- **Physical-Weight = 0** (81 cases): biologically impossible → treat as missing.
- **BIA-BIA_Fat < 0** (112 cases): negative body fat % is impossible → treat as missing or clip.
- **BIA-BIA_FMI < 0** (51 cases): same issue.
- **BIA extreme outliers**: BMR max 83,152 (vs mean 1,237), FFM max 8,799, etc. — likely measurement errors or unit inconsistencies. Need capping / removal.
- **Physical-Diastolic_BP = 0** and **Systolic_BP = 0**: impossible → treat as missing.

### 3.3 PAQ_A vs PAQ_C
- PAQ_A = Physical Activity Questionnaire for **Adolescents**, PAQ_C = for **Children**.
- 3,920 participants have **both** scores → unexpected but usable.
- **Decision**: merge into a single `PAQ_Combined` feature (average when both present, use whichever is available otherwise). Then drop the original `PAQ_A_Total` and `PAQ_C_Total`.

---

## 4. Detailed Execution Plan

### Step 1 — Exploratory Data Analysis (EDA)

1. **Univariate analysis**:
   - Distribution of every numeric variable (histograms / KDE plots).
   - Bar plots for categorical variables (Sex, fitness zones, internet hours/day).
   - Box plots to visually detect outliers.

2. **Target analysis**:
   - Bar plot of `sii` class distribution.
   - Show the `sii` ↔ `PCIAT_Total` derivation explicitly (scatter/step plot with thresholds).

3. **Bivariate analysis**:
   - Correlation heatmap of all numeric features (grouped by instrument).
   - Top non-PCIAT correlates with `sii`: Height (0.24), Internet hours/day (0.22), Weight (0.22), Sleep T-score (0.21), Age (0.21).
   - Box plots of key features grouped by `sii` class.
   - Cross-tabulations for categorical features vs `sii` (Chi-squared tests).

4. **Multicollinearity check**:
   - BIA features are highly correlated with each other and with Physical measures (e.g., BIA-BMI vs Physical-BMI). Identify redundant pairs (correlation > 0.9).
   - Physical Height, Weight, BMI are mechanically related.

5. **Missingness analysis**:
   - Missingness heatmap (are missing values random or systematic?).
   - Check if missingness correlates with `sii` or Age (e.g., PCIAT missing more for younger children?).

### Step 2 — Data Pre-processing

#### 2.1 Handle erroneous values (before imputation)
- Replace CGAS_Score = 999 → NaN.
- Replace Physical-BMI = 0, Physical-Weight = 0, Physical-Diastolic_BP = 0, Physical-Systolic_BP = 0 → NaN.
- Replace BIA-BIA_Fat < 0 → NaN.
- Replace BIA-BIA_FMI < -10 (or similar threshold) → NaN.
- Cap BIA extreme outliers using IQR-based or domain-based thresholds.

#### 2.2 Drop PCIAT columns and all Season columns
- **Drop all 22 PCIAT columns**: `sii` is a direct transformation of `PCIAT_Total`. Using PCIAT items as features would create **data leakage** (trivially predicting the target from its own source). Any model must predict `sii` from the *other* instruments.
- **Drop all 11 Season columns**: Season of participation has negligible predictive value for internet addiction severity and would inflate dimensionality with one-hot encoding (11 × 4 = 44 dummies) for minimal gain.

#### 2.3 Encode categorical variables
| Variable | Encoding Strategy |
|---|---|
| Season columns (11 cols) | **Drop** (see Step 2.2). |
| `Basic_Demos-Sex` (0/1) | Already binary — keep as-is. |
| `FGC-*_Zone` (0/1 binary) | Already binary — keep as-is. |
| `FGC-FGC_GSND_Zone`, `FGC-FGC_GSD_Zone` (1/2/3) | **Ordinal** (Weak < Normal < Strong) — keep numeric. |
| `BIA-BIA_Activity_Level_num` (1–5) | **Ordinal** — keep numeric. |
| `BIA-BIA_Frame_num` (1/2/3) | **Ordinal** — keep numeric. |
| `PreInt_EduHx-computerinternet_hoursday` (0–3) | **Ordinal** — keep numeric. |

#### 2.4 Missing value imputation
Given the varying missingness patterns, use a **multi-strategy approach**:

1. **Columns already dropped**: all PCIAT (leakage) and all Season columns (low signal) — no imputation needed.
2. **PAQ_A / PAQ_C**: combine into `PAQ_Combined` first (see Step 3), then impute the combined column.
3. **Numeric features (continuous)**: use **KNN imputation** or **Iterative Imputer** (MICE-like), as features within the same instrument are correlated and can inform each other.
4. **Categorical features with low missingness**: impute with **mode**.

#### 2.5 Feature scaling
- Apply **StandardScaler** (z-score normalization) to all continuous features after imputation.
- Do **not** scale binary/ordinal categoricals.
- Fit scaler on training set only, transform test set (prevent leakage).

### Step 3 — Feature Engineering (New Variables)

1. **`PAQ_Combined`**: Merge PAQ_A_Total and PAQ_C_Total into a single physical activity score.
   - If both present: average them (or prefer the age-appropriate one based on `Age`).
   - If only one present: use the available one.

2. **`Fitness_Endurance_TotalTime`**: Combine `Time_Mins` × 60 + `Time_Sec` into total seconds on treadmill.

3. **`BP_Category`**: Derive blood pressure category from Systolic/Diastolic (Normal, Elevated, High).

4. **`BMI_Category`**: Derive BMI category from BMI + Age + Sex using pediatric percentile charts (Underweight / Normal / Overweight / Obese), or at least a simple binning.

5. **`Grip_Strength_Ratio`**: `FGC_GSD / FGC_GSND` (dominant-to-non-dominant ratio, measure of laterality/symmetry).

6. **`Flexibility_Avg`**: Average of `FGC_SRL` and `FGC_SRR`.

7. **`Fitness_Zone_Score`**: Sum of all binary fitness zones (CU_Zone + PU_Zone + SRL_Zone + SRR_Zone + TL_Zone) → a composite "how many fitness tests in Healthy Zone" score (0–5).

8. **`Body_Composition_Index`**: Ratio `BIA_Fat / BIA_FFM` (fat to lean mass ratio).

9. **`Missingness_Count`**: Number of missing features per participant (could be informative — more missing data may correlate with less engagement / higher impairment).

---

## 5. Summary of Columns After Pre-processing

| Category | Action |
|---|---|
| `id` | Drop (identifier only) |
| 11 Season columns | **Drop** (low signal, high dimensionality cost) |
| 22 PCIAT columns | **Drop** (data leakage) |
| `PAQ_A_Total` + `PAQ_C_Total` | **Replace** with `PAQ_Combined`, then drop originals |
| Erroneous values | Replace with NaN, then impute |
| Remaining ~36 numeric features | Impute → Scale |
| ~9 new engineered features | Compute → Scale where needed |
| **Target `sii`** | Keep as-is (ordinal 0–3) |

**Expected final feature matrix**: ~45 features × 8,460 rows.

---

## 6. Tools & Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

---

## 7. Notebook Structure

The work should be organized in a single Jupyter notebook with these sections:

1. **Data Loading & Dictionary Review**
2. **EDA — Univariate** (distributions, outlier detection)
3. **EDA — Bivariate & Multivariate** (correlations, target analysis)
4. **Data Cleaning** (erroneous values, PCIAT removal)
5. **Encoding Categorical Variables**
6. **Missing Value Imputation**
7. **Feature Engineering**
8. **Feature Scaling**
9. **Final Dataset Summary & Export**
