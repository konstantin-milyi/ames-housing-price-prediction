# Housing Price Prediction 🏡

**Project Objective** — To develop a predictive model for accurately forecasting the final house price (target variable — `SalePrice`).

## 📊 Practical Value
Implementing such algorithms automates the property valuation process, minimizes human error, and provides clear insights into exactly which architectural and structural features add the most market value to a property.

---

## 📂 Data Description
The project is based on a real estate dataset (derived from the classic Ames Housing dataset). It contains detailed information about each property, which can be divided into two main groups:
* **Quantitative Variables**: First and second-floor square footage, total lot area, garage size, and the age of the building.
* **Categorical Variables**: Nominal (neighborhood, foundation type) and ordinal (finish quality, exterior condition).

---

## 🛠 Workflow Steps

### 1. Exploratory Data Analysis (EDA)
* **Target Variable:** The mean is $180k, the median is $163k. The distribution was right-skewed due to luxury properties (skewness: 1.88). After log transformation, the distribution became almost perfectly symmetric (skewness: 0.12).
* **Outlier Removal:** Removed extreme outliers (`GrLivArea < 4000`), following the official recommendation from the dataset creator, Prof. Dean De Cock, as these represent anomalous non-market transactions.

### 2. Feature Engineering & Selection
* **Statistical Selection:** Used Mutual Information and ANOVA to evaluate the predictive power of features.
* **Handling Missing Values:** Missing values typically indicated the physical absence of a feature (e.g., no garage or pool). Imputed with zeros for numerical features, or a 'Missing' category for categorical ones. Contextual imputation (Neighborhood medians) was used for `LotFrontage`.
* **New Features Created:** * `TotalSF`: Total Square Footage (living area + basement).
  * `SF_per_Room`: Average square footage per room.
  * `SinceRemod`: Years elapsed from construction/remodel until sale.
* **Neighborhood Clustering:** Grouped similar locations into broader market clusters: `Townhouse` (high-density), `Historic` (university zones), and `Suburban` (spacious lots).

### 3. Pipeline & Encoding
* **Ordinal Variables:** Merged rare categories with their closest logical neighbors. Applied manual ordinal encoding based on logical hierarchy (e.g., Po < Fa < TA < Gd < Ex).
* **Nominal Variables:** Rare categories (< 3%) were collapsed into a single 'Rare' group. Applied target-based encoding (`CatBoostEncoder`).
* **Multicollinearity:** Used `SmartCorrelatedSelection` to remove highly correlated features (coefficient > 0.8), keeping those that optimized the baseline Random Forest score.

### 4. Modeling & Final Results
* **Models Tested:** Transitioned from baseline algorithms to Gradient Boosting ensemble methods (GBM, XGBoost, LightGBM).
* **Performance Metrics:** Transitioning to gradient boosting yielded an immediate performance boost, with the average error (RMSE) dropping by $4k–$5k.

| Model | RMSE ($) | R² Score |
| :--- | :---: | :---: |
| **LightGBM** | `21,500` | `0.912` |
| **XGBoost** | `22,100` | `0.905` |
| **Gradient Boosting** | `23,000` | `0.891` |
| *Random Forest (Baseline)* | *27,500* | *0.850* |

* **Feature Importance:** The most critical features for pricing were the engineered `TotalSF`, `OverallQual`, and `Neighborhood`.
* **Error Analysis:** Scatter plot analysis revealed that the model systematically underestimates extremely expensive houses (>$400k), attempting to average them with their less expensive neighbors.

---

## 📖 Data Dictionary (Key Variables)

<details>
<summary>Click to expand Data Dictionary</summary>

### Numeric Variables
* **LotFrontage** — Linear feet of street connected to property.
* **LotArea** — Lot size in square feet.
* **YearBuilt** — Original construction date.
* **GrLivArea** — Above grade (ground) living area square feet.
* **TotalBsmtSF** — Total square feet of basement area.

### Categorical Variables
* **MSSubClass** — The building class (e.g., 20 - 1-STORY, 60 - 2-STORY).
* **OverallQual** — Overall material and finish quality (1-10).
* **OverallCond** — Overall condition rating (1-9).
* **ExterQual** / **ExterCond** — Exterior material quality and condition.
* **BsmtQual** / **BsmtCond** — Basement height and general condition.
* **CentralAir** — Central air conditioning (Y/N).
</details>
