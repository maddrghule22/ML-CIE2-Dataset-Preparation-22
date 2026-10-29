# Machine Learning CIE 2 — Dataset Preparation & Exploratory Data Analysis

**Activity 1:** Dataset Preparation Activity — Unit I: Foundations & Data Preparation  
**Academic Reference & Faculty Profile:** [https://github.com/Princy2129](https://github.com/Princy2129)  
**Target Student Level:** Second Year Engineering (Machine Learning Laboratory)

---

## 1. Project Title & Objective

### 1.1 Activity Title
**Machine Learning CIE 2 — Activity 1: Dataset Preparation Activity**  
*Unit I: Foundations & Data Preparation*

### 1.2 Project Objective
The objective of this project is to build an end-to-end, fully reproducible, and beginner-friendly data preparation pipeline using Python on a standard public tabular dataset. Real-world raw datasets frequently contain missing observations, duplicate records, unstandardised text formats, unencoded categorical attributes, and wildly varying numerical scales. 

This project demonstrates the complete data engineering lifecycle:
1. Ingesting and structurally inspecting a public tabular dataset.
2. Detecting and repairing data-quality defects (handling missing values through column-specific statistical justification).
3. Deduplicating rows and enforcing categorical consistency.
4. Conducting exploratory data analysis (EDA) with clear, evidence-based visualisations.
5. Performing categorical encoding without multicollinearity (One-Hot Encoding with reference category dropping).
6. Standardising continuous numerical features using `StandardScaler` to prepare the feature space for distance- and gradient-based algorithms.
7. Validating the prepared dataset using programmatic assertions and exporting the result to `prepared_titanic.csv`.

> **Scope Boundary:** In strict adherence to the project instructions, **no machine learning prediction model is trained or evaluated**. The scope is strictly focused on foundational data preparation.

---

## 2. Dataset Description & Public Source

We utilize the canonical **Titanic Passenger Survival Dataset** curated in Seaborn's public repository:
* **Dataset Name:** Titanic Passenger Manifest
* **Source Repository:** [https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)
* **Direct Raw Download URL:** [https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv)
* **File Format:** CSV (Comma-Separated Values)
* **Raw Dimensions:** 891 rows, 15 columns

### Schema and Column Definitions
| Column Name | Data Type | Feature Nature | Description & Meaning |
| :--- | :--- | :--- | :--- |
| `survived` | `int64` | Binary Target | Ground truth survival status (`0` = Did not survive, `1` = Survived). Potential future target. |
| `pclass` | `int64` | Ordinal Feature | Passenger ticket class (`1` = 1st, `2` = 2nd, `3` = 3rd); represents socio-economic tier. |
| `sex` | `object` | Nominal Feature | Passenger biological gender (`'male'`, `'female'`). |
| `age` | `float64` | Continuous Feature | Passenger age in years (`0.42` to `80.0` yrs). Contains 177 missing entries. |
| `sibsp` | `int64` | Discrete Feature | Count of siblings and spouses aboard with the passenger. |
| `parch` | `int64` | Discrete Feature | Count of parents and children aboard with the passenger. |
| `fare` | `float64` | Continuous Feature | Passenger ticket price in British Pounds (`£0.00` to `£512.33`). |
| `embarked` | `object` | Nominal Feature | Port of embarkation (`'C'` = Cherbourg, `'Q'` = Queenstown, `'S'` = Southampton). 2 missing. |
| `class` | `object` | Redundant | Textual label of ticket class (`'First'`, `'Second'`, `'Third'`); identical to `pclass`. |
| `who` | `object` | Nominal Feature | Demographic categorization (`'man'`, `'woman'`, `'child'`). |
| `adult_male`| `bool` | Binary Feature | Boolean flag (`True`/`False`) indicating adult male status. |
| `deck` | `object` | Sparse Categorical | Cabin deck letter (`'A'` through `'G'`). 688 missing (>77%). |
| `embark_town`| `object`| Redundant | Port city name (`'Cherbourg'`, `'Queenstown'`, `'Southampton'`); identical to `embarked`. |
| `alive` | `object` | Redundant | Survival status string (`'yes'`, `'no'`); identical to `survived`. |
| `alone` | `bool` | Binary Feature | Boolean flag (`True`/`False`) indicating passenger had 0 family members aboard. |

---

## 3. Technology Stack

* **Programming Language:** Python 3.12 (fully compatible with Google Colab and local Jupyter)
* **Data Manipulation & Analysis:** `pandas` (>=2.0.0), `numpy` (>=1.24.0)
* **Exploratory Visualisation:** `matplotlib` (>=3.7.0), `seaborn` (>=0.12.0)
* **Feature Preprocessing & Scaling:** `scikit-learn` (>=1.3.0)
* **Notebook Runner & Execution:** `jupyter`, `ipykernel`, `nbformat`
* **Academic PDF Compilation:** `reportlab` (>=4.0.0)

---

## 4. Project Structure

```text
ML-CIE2-Dataset-Preparation/
├── dataset_preparation.ipynb    # Fully executed Jupyter Notebook with all outputs & visualisations
├── prepared_titanic.csv         # Cleaned, encoded, and scaled CSV dataset (775 rows × 13 columns)
├── README.md                    # Comprehensive academic and project documentation
├── requirements.txt             # Exact Python package dependencies
├── reports/
│   ├── Activity_Report.pdf      # Typeset multi-page formal academic laboratory report
│   └── figures/
│       ├── age_distribution.png # EDA Figure 1: Histogram of passenger ages
│       ├── survival_counts.png  # EDA Figure 2: Bar chart of survival counts
│       └── age_vs_fare.png      # EDA Figure 3: Scatter plot of age vs. ticket fare
└── build_notebook.py            # Reproducible script used to compile and execute the notebook
```

---

## 5. Instructions for Running the Notebook

### Option A: Running in Google Colab (Recommended for Quick Review)
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File** > **Upload Notebook** and upload `dataset_preparation.ipynb`.
3. Run all cells sequentially (**Runtime** > **Run all**).
4. In Section 11, the notebook automatically triggers a local download of `prepared_titanic.csv` via `google.colab.files.download`.

### Option B: Running Locally with Jupyter / VS Code
1. Clone or download the repository:
   ```bash
   git clone https://github.com/maddrghule22/ML-CIE2-Dataset-Preparation.git
   cd ML-CIE2-Dataset-Preparation
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook dataset_preparation.ipynb
   ```
5. Execute the notebook cells sequentially from top to bottom.

---

## 6. Data Cleaning & Preparation Methodology

### 6.1 Missing-Value Remediation
An initial audit revealed 867 total missing cells across four features:
* **`deck` (688 missing / 77.22%):** Cabin assignment was primarily recorded for wealthy first-class passengers. Imputing a synthetic value for over 77% of rows would inject massive artificial bias. Hence, `deck` was **excluded (dropped)**.
* **`age` (177 missing / 19.87%):** Age shows mild right-skewness (ranging from 0.42 to 80 years). The **median** ($28.0$ years) was selected as the imputation value because it is robust against extreme values, unlike the arithmetic mean ($29.7$ years).
* **`embarked` & `embark_town` (2 missing / 0.22%):** Imputed with the **mode** (`'S'` / `'Southampton'`), which represents over 72% of all boarding passengers.

> **Machine Learning Concept — Preventing Data Leakage:**  
> In an operational predictive workflow, imputation parameters (such as the median age) must be derived **solely from the training partition** ($X_{train}$) and subsequently applied to test and validation partitions. Calculating statistics across the entire dataset prior to splitting introduces subtle information leakage.

### 6.2 Deduplication & Consistency Auditing
* **Duplicate Detection:** Identified 116 exact duplicate rows in the cleaned working copy (where identical survival, pclass, sex, age, fare, and family features occurred).
* **Removal:** Exact duplicate rows were removed using `drop_duplicates().reset_index(drop=True)` to prevent identical observations from unfairly weighting loss functions or contaminating train/test splits.
* **Text Sanitation:** Stripped leading/trailing whitespace and checked categorical values in `sex`, `embarked`, `who`, `embark_town`, `class`, and `alive` to verify zero spelling anomalies.
* **Redundancy Elimination:** Dropped `alive` (duplicate of `survived`), `class` (duplicate of `pclass`), and `embark_town` (duplicate of `embarked`) to prevent perfect multicollinearity.

---

## 7. Exploratory Data Analysis (EDA) & Visualisations

The project includes three clear, readable visualisations generated via Matplotlib and Seaborn (saved at 300 DPI in `reports/figures/`):

### 1. Histogram of Passenger Ages (`reports/figures/age_distribution.png`)
* **Visual Form:** 25 bins with Kernel Density Estimation (KDE) and vertical indicators for median (28.0 yrs) and mean (29.6 yrs).
* **Observation:** Passenger ages span from 0.42 to 80.0 years. The primary mode occurs in the young adult bracket (20 to 35 years). A distinct secondary peak is observable for children under 8 years.

### 2. Bar Chart of Passenger Survival (`reports/figures/survival_counts.png`)
* **Visual Form:** Categorical bar plot comparing non-survivors (`0`) and survivors (`1`) with annotated counts and percentages.
* **Observation:** In the cleaned 775-passenger cohort, 455 passengers (58.7%) did not survive, while 320 passengers (41.3%) survived. This documents a moderate class imbalance.

### 3. Scatter Plot of Passenger Age vs. Ticket Fare (`reports/figures/age_vs_fare.png`)
* **Visual Form:** Two-dimensional scatter plot of age against fare, colored and styled by survival status.
* **Observation:** Over 75% of passengers paid under £35.00 across all age groups. High ticket fares (£100.00 to £512.33) are concentrated among passengers who survived (green markers), showing a strong empirical correlation between fare paid and survival outcome.

*Note on Scientific Rigor:* Observations are strictly separated from causal assumptions. We observe that high-fare passengers survived at higher rates; we do not claim that ticket price alone was the biological or operational cause of survival.

---

## 8. Categorical Encoding & Numerical Scaling

### 8.1 Categorical Encoding
1. **Boolean Variables:** Converted `adult_male` and `alone` to integer values (`0` and `1`).
2. **Ordinal Features:** Preserved `pclass` as an integer (`1`, `2`, `3`) because its values inherently carry meaningful mathematical ordering representing socio-economic status.
3. **Nominal Features (One-Hot Encoding):** Applied `pd.get_dummies` with `drop_first=True` to prevent the dummy variable trap (multicollinearity):
   * `sex`: Encoded as `sex_male` (`1` = male, `0` = female [reference]).
   * `embarked`: Encoded as `embarked_Q` and `embarked_S` (`C` [Cherbourg] is reference).
   * `who`: Encoded as `who_man` and `who_woman` (`child` is reference).

### 8.2 Numerical Feature Scaling
* **Target Features:** `age` (range: 0.42–80.0) and `fare` (range: £0.00–£512.33).
* **Method:** `StandardScaler` from Scikit-learn:
  $$z = \frac{x - \mu}{\sigma}$$
* **Result:** Transformed `age` and `fare` to mean $\approx 0.0$ and standard deviation $\approx 1.0$.
* **Exclusions:** The target variable (`survived`), ordinal ranks (`pclass`), discrete family counts (`sibsp`, `parch`), and all binary/dummy flags (`0`/`1`) were strictly excluded from continuous scaling to preserve their natural discrete interpretability.
* **Algorithmic Justification:** Distance-based models (KNN, SVM) and gradient-based models (Logistic Regression, Neural Networks) are severely distorted when features differ by multiple orders of magnitude. StandardScaler balances the feature landscape.

---

## 9. Final Dataset Dimensions & Validation Summary

| Metric / Check | Raw Dataset | Prepared Dataset (`prepared_titanic.csv`) |
| :--- | :--- | :--- |
| **Row Count** | 891 | **775** (116 duplicate rows purged) |
| **Column Count** | 15 | **13** (3 redundant dropped, 1 sparse dropped, 3 OHE features added) |
| **Missing Cells** | 867 | **0** (Verified via assertion) |
| **Duplicate Rows** | 107 (raw) | **0** (Verified via assertion) |
| **Scaled Features** | None | `age` ($\mu=0.0, \sigma=1.0$), `fare` ($\mu=0.0, \sigma=1.0$) |
| **Target Variable** | `survived` (unscaled) | `survived` (int64, unscaled, preserved) |

### Programmatic Assertions Verified in Notebook
```python
assert df_scaled.isnull().sum().sum() == 0   # 0 missing values
assert df_scaled.duplicated().sum() == 0     # 0 duplicate rows
assert df_scaled.shape == (775, 13)          # Exact final dimensions
assert 'survived' in df_scaled.columns       # Target column intact
```

---

## 10. Checklist Mapping to Evaluation Criteria

| Evaluation Criterion | Marks | Evidence in Project Deliverables |
| :--- | :---: | :--- |
| **1. Dataset selection and understanding** | **1 Mark** | Authentic Seaborn URL ingested; shape (891, 15) documented; column semantics, types, and potential prediction target clearly explained in Sections 1–3 of notebook, README, and Report. |
| **2. Missing-value and data-quality handling** | **1 Mark** | Complete audit table calculated; `deck` dropped due to 77.2% missingness; `age` median-imputed; `embarked` mode-imputed; 116 duplicate rows removed; whitespace trimmed; 0 missing values remain. |
| **3. EDA and visualisations** | **1 Mark** | Three publication-standard visualisations (Age histogram, Survival bar chart, Age vs. Fare scatter plot) generated with Matplotlib & Seaborn; verified empirical findings documented. |
| **4. Encoding and appropriate scaling** | **1 Mark** | Redundancies dropped; booleans cast to 0/1; nominal features one-hot encoded (`drop_first=True`); continuous features (`age`, `fare`) standard-scaled ($\mu=0, \sigma=1$); target excluded from scaling. |
| **5. Prepared dataset and conclusion** | **1 Mark** | Final DataFrame exported to `prepared_titanic.csv` (775 × 13); verified by reloading; student-friendly conclusion provided; explicit disclaimer stating no ML model was trained. |

---

## 11. Publishing to GitHub with Public Visibility

To ensure the repository is publicly accessible and can be reviewed without authentication barriers:
1. Initialize a git repository and commit all required artifacts:
   ```bash
   git init
   git add dataset_preparation.ipynb prepared_titanic.csv README.md requirements.txt reports/
   git commit -m "feat: complete ML CIE 2 dataset preparation and EDA project"
   ```
2. Create and push to a **public** GitHub repository using the GitHub CLI:
   ```bash
   gh repo create ML-CIE2-Dataset-Preparation-22 --public --source=. --remote=origin --push
   ```
3. **Verify Public Accessibility:**
   * Open an Incognito / Private browsing window (where you are not logged in to GitHub).
   * Navigate to the repository URL: `https://github.com/maddrghule22/ML-CIE2-Dataset-Preparation-22`.
   * Confirm that the repository contents, README, notebook, and `prepared_titanic.csv` are immediately readable without requiring login or access requests.

---

## 12. Acknowledgement & Faculty Reference

“This project was completed as part of the Machine Learning CIE 2 activity. Faculty reference and Machine Learning learning resources: [https://github.com/Princy2129”](https://github.com/Princy2129%E2%80%9D)
