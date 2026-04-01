# CMI Internet Dataset — Column Bounds Reference

For each column, natural and plausible lower/upper limits are defined based on the data dictionary and domain knowledge. Values outside these ranges should be treated as erroneous.

---

## Demographics

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `Basic_Demos-Age` | 5 | 22 | Study range; data confirmed |
| `Basic_Demos-Sex` | 0 | 1 | Categorical only |

---

## CGAS (Children's Global Assessment Scale)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `CGAS-CGAS_Score` | 1 | 100 | Official scale is 1–100. Max in data is **999** (4 rows) — clearly a missing/invalid code. Values like 595, 703, 819 also present. |

---

## Physical Measures

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `Physical-BMI` | 10 | 55 | 0 in data is impossible; 59 is extreme morbid obesity |
| `Physical-Height` | 36 | 80 | inches (≈91cm–203cm); min 33in in data is borderline for a 5yo |
| `Physical-Weight` | 25 | 400 | lbs; 0 in data is impossible |
| `Physical-Waist_Circumference` | 15 | 55 | inches; data range 18–50 looks reasonable |
| `Physical-Diastolic_BP` | 40 | 110 | mmHg; 0 in data is impossible; 179 is hypertensive crisis |
| `Physical-HeartRate` | 40 | 160 | bpm; 27 in data is dangerously low and likely erroneous |
| `Physical-Systolic_BP` | 60 | 180 | mmHg; 0 and 203 in data are impossible |

---

## FitnessGram Endurance

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `Fitness_Endurance-Max_Stage` | 1 | 15 | PACER test standard protocol; 28 in data is far above norm |
| `Fitness_Endurance-Time_Mins` | 0 | 20 | minutes; 0 valid (didn't complete), 20 is generous upper |
| `Fitness_Endurance-Time_Sec` | 0 | 59 | seconds; any value ≥60 is impossible |

---

## FitnessGram Child (FGC)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `FGC-FGC_CU` | 0 | 75 | curl-ups; FitnessGram protocol caps at 75; 115 in data is suspicious |
| `FGC-FGC_GSND` / `FGC-FGC_GSD` | 2 | 75 | kg grip strength; 0 suspicious; 124 is very high for a child |
| `FGC-FGC_PU` | 0 | 60 | push-ups; 51 in data is within range |
| `FGC-FGC_SRL` / `FGC-FGC_SRR` | -10 | 15 | inches; negative = doesn't reach toes, valid |
| `FGC-FGC_TL` | 3 | 24 | trunk lift inches; FitnessGram scores 6–12in as the healthy zone, physically 1–24 possible |

Zone columns (`_Zone`) are fixed categoricals per dictionary — check only for out-of-range enumerated values.

---

## BIA (Bio-electric Impedance Analysis)

The BIA continuous columns have **severe outliers**: maxima are 100–1000× the 99th percentile, indicating data entry errors (e.g., misplaced decimal point). The 99th percentile is used as a guide for realistic upper bounds.

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `BIA-BIA_BMC` | 0.5 | 5.0 | kg bone mineral content; negative in data is impossible; 4115 is absurd (99th pct = 12) |
| `BIA-BIA_BMI` | 10 | 55 | same as physical BMI |
| `BIA-BIA_BMR` | 500 | 3000 | kcal/day; 83152 in data is impossible (99th pct ≈ 1986) |
| `BIA-BIA_DEE` | 800 | 5000 | kcal/day; 124728 is impossible (99th pct ≈ 3534) |
| `BIA-BIA_ECW` | 5 | 50 | liters extracellular water; 3233 is impossible (99th pct ≈ 53) |
| `BIA-BIA_FFM` | 15 | 150 | kg fat-free mass; 8799 is impossible (99th pct ≈ 161) |
| `BIA-BIA_FFMI` | 10 | 25 | kg/m² fat-free mass index |
| `BIA-BIA_FMI` | 0 | 20 | kg/m² fat mass index; −194 in data is impossible |
| `BIA-BIA_Fat` | 5 | 60 | % body fat; −8745 and 153 in data are impossible |
| `BIA-BIA_ICW` | 10 | 60 | liters intracellular water; 2458 is impossible (99th pct ≈ 66) |
| `BIA-BIA_LDM` | 3 | 45 | kg lean dry mass; 3108 is impossible (99th pct ≈ 48) |
| `BIA-BIA_LST` | 15 | 140 | kg lean soft tissue; 4684 is impossible (99th pct ≈ 148) |
| `BIA-BIA_SMM` | 5 | 90 | kg skeletal muscle mass; 3608 is impossible (99th pct ≈ 95) |
| `BIA-BIA_TBW` | 15 | 100 | liters total body water; 5691 is impossible (99th pct ≈ 114) |

`BIA-BIA_Activity_Level_num` (1–5) and `BIA-BIA_Frame_num` (1–3) are categorical — check only for out-of-range values.

---

## Physical Activity Questionnaires (PAQ)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `PAQ_A-PAQ_A_Total` | 1 | 5 | Scale is 1–5; min 0.66 in data is slightly below minimum |
| `PAQ_C-PAQ_C_Total` | 1 | 5 | Scale is 1–5; min 0.58 in data is below minimum |

---

## PCIAT (Parent-Child Internet Addiction Test)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `PCIAT-PCIAT_01` … `PCIAT-PCIAT_20` | 0 | 5 | Categorical per dictionary; data max is 5 ✓ |
| `PCIAT-PCIAT_Total` | 0 | 100 | 20 items × max 5 = 100; data max is 93 ✓ |

---

## Sleep Disturbance Scale (SDS)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `SDS-SDS_Total_Raw` | 26 | 130 | Standard SDSC is 26 items × 1–5; min 17 in data is below the theoretical minimum |
| `SDS-SDS_Total_T` | 30 | 100 | T-score (mean=50, SD=10); data range 38–100 looks fine |

---

## Internet Use & Target

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `PreInt_EduHx-computerinternet_hoursday` | 0 | 3 | Categorical 0–3 per dictionary; data confirmed |
| `sii` | 0 | 3 | Severity Impairment Index derived from PCIAT_Total |

---

## Summary of Major Anomalies Found in Data

| Column | Issue |
|--------|-------|
| `CGAS-CGAS_Score` | Values of 595, 703, 819, 999 present — scale max is 100 |
| All BIA continuous columns | Extreme maxima (100–1000× 99th pct) — likely decimal/unit entry errors |
| `Physical-Diastolic_BP`, `Physical-Systolic_BP` | Zero values present — physiologically impossible |
| `BIA-BIA_FMI`, `BIA-BIA_Fat` | Negative values present — impossible |
| `SDS-SDS_Total_Raw` | Values below 26 (the scale's theoretical minimum) |
| `PAQ_A-PAQ_A_Total`, `PAQ_C-PAQ_C_Total` | Values below 1 (scale minimum) |
| `Physical-BMI`, `Physical-Weight` | Zero values present — impossible |
