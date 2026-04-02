# CMI Internet Dataset — Column Bounds Reference

For each column, only **physically or logically impossible** limits are defined. Values outside these ranges are erroneous by definition (not just extreme). Outliers within these bounds are retained.

---

## Demographics

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `Basic_Demos-Age` | 0 | — | Negative age is impossible; study range 5–22 is not a hard constraint |
| `Basic_Demos-Sex` | 0 | 1 | Categorical only |

---

## CGAS (Children's Global Assessment Scale)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `CGAS-CGAS_Score` | 1 | 100 | Official scale is 1–100; values like 595, 703, 999 are impossible by definition |

---

## Physical Measures

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `Physical-BMI` | 0 (exclusive) | — | BMI ≤ 0 is physically impossible |
| `Physical-Height` | 0 (exclusive) | — | Height ≤ 0 is physically impossible |
| `Physical-Weight` | 0 (exclusive) | — | Weight ≤ 0 is physically impossible |
| `Physical-Waist_Circumference` | 0 (exclusive) | — | ≤ 0 is physically impossible |
| `Physical-Diastolic_BP` | 0 (exclusive) | — | BP ≤ 0 is physically impossible |
| `Physical-HeartRate` | 0 (exclusive) | — | Heart rate ≤ 0 is physically impossible |
| `Physical-Systolic_BP` | 0 (exclusive) | — | BP ≤ 0 is physically impossible |

---

## FitnessGram Endurance

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `Fitness_Endurance-Max_Stage` | 0 (exclusive) | — | Must be a positive stage number |
| `Fitness_Endurance-Time_Mins` | 0 | — | 0 is valid (test not completed) |
| `Fitness_Endurance-Time_Sec` | 0 | 59 | Seconds ≥ 60 are logically impossible |

---

## FitnessGram Child (FGC)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `FGC-FGC_CU` | 0 | — | Count; negative is impossible |
| `FGC-FGC_GSND` / `FGC-FGC_GSD` | 0 (exclusive) | — | Grip strength ≤ 0 is impossible |
| `FGC-FGC_PU` | 0 | — | Count; negative is impossible |
| `FGC-FGC_SRL` / `FGC-FGC_SRR` | — | — | Negative values are valid (doesn't reach toes) |
| `FGC-FGC_TL` | 0 (exclusive) | — | Trunk lift must be positive |

Zone columns (`_Zone`) are fixed categoricals per dictionary — check only for out-of-range enumerated values.

---

## BIA (Bio-electric Impedance Analysis)

Extreme upper-end values are suspicious outliers but are not removed here. Only negative values that are physically impossible are flagged.

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `BIA-BIA_BMC` | 0 (exclusive) | — | Negative bone mineral content is impossible |
| `BIA-BIA_BMI` | 0 (exclusive) | — | BMI ≤ 0 is impossible |
| `BIA-BIA_BMR` | 0 (exclusive) | — | Basal metabolic rate must be positive |
| `BIA-BIA_DEE` | 0 (exclusive) | — | Daily energy expenditure must be positive |
| `BIA-BIA_ECW` | 0 (exclusive) | — | Negative extracellular water is impossible |
| `BIA-BIA_FFM` | 0 (exclusive) | — | Negative fat-free mass is impossible |
| `BIA-BIA_FFMI` | 0 (exclusive) | — | Negative fat-free mass index is impossible |
| `BIA-BIA_FMI` | 0 | — | Negative fat mass index is impossible |
| `BIA-BIA_Fat` | 0 | — | Negative body fat % is impossible |
| `BIA-BIA_ICW` | 0 (exclusive) | — | Negative intracellular water is impossible |
| `BIA-BIA_LDM` | 0 (exclusive) | — | Negative lean dry mass is impossible |
| `BIA-BIA_LST` | 0 (exclusive) | — | Negative lean soft tissue is impossible |
| `BIA-BIA_SMM` | 0 (exclusive) | — | Negative skeletal muscle mass is impossible |
| `BIA-BIA_TBW` | 0 (exclusive) | — | Negative total body water is impossible |

`BIA-BIA_Activity_Level_num` (1–5) and `BIA-BIA_Frame_num` (1–3) are categorical — check only for out-of-range values.

---

## Physical Activity Questionnaires (PAQ)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `PAQ_A-PAQ_A_Total` | 1 | 5 | Scale is 1–5; values outside this range are impossible by definition |
| `PAQ_C-PAQ_C_Total` | 1 | 5 | Scale is 1–5; values outside this range are impossible by definition |

---

## PCIAT (Parent-Child Internet Addiction Test)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `PCIAT-PCIAT_01` … `PCIAT-PCIAT_20` | 0 | 5 | Categorical per dictionary |
| `PCIAT-PCIAT_Total` | 0 | 100 | 20 items × max 5 = 100 |

---

## Sleep Disturbance Scale (SDS)

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `SDS-SDS_Total_Raw` | 26 | 130 | 26 items × 1–5; outside this range is impossible by definition |
| `SDS-SDS_Total_T` | — | — | T-score; no hard physical bounds |

---

## Internet Use & Target

| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| `PreInt_EduHx-computerinternet_hoursday` | 0 | 3 | Categorical 0–3 per dictionary |
| `sii` | 0 | 3 | Derived from PCIAT_Total |

---

## Summary of Physically Impossible Anomalies in Data

| Column | Issue |
|--------|-------|
| `CGAS-CGAS_Score` | Values of 595, 703, 819, 999 — scale max is 100 |
| `Physical-Diastolic_BP`, `Physical-Systolic_BP` | Zero values — physiologically impossible |
| `Physical-BMI`, `Physical-Weight` | Zero values — impossible |
| `BIA-BIA_FMI`, `BIA-BIA_Fat` | Negative values — impossible |
| `BIA-BIA_ECW`, `BIA-BIA_ICW`, `BIA-BIA_BMC` | Negative values — impossible |
| `Fitness_Endurance-Time_Sec` | Values ≥ 60 — logically impossible for seconds |
| `SDS-SDS_Total_Raw` | Values below 26 — below theoretical minimum of the scale |
| `PAQ_A-PAQ_A_Total`, `PAQ_C-PAQ_C_Total` | Values below 1 — below scale minimum |
