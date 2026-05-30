# Housing Price Prediction 

**Project Objective** — To develop a predictive model for accurately forecasting the final house price (target variable — `SalePrice`).

## Practical Value
Implementing such algorithms automates the property valuation process, minimizes human error, and provides clear insights into exactly which architectural and structural features add the most market value to a property.

---

## Data Description
The project is based on a real estate dataset (derived from the classic Ames Housing dataset). It contains detailed information about each property, which can be divided into two main groups:
* **Quantitative Variables**: First and second-floor square footage, total lot area, garage size, and the age of the building.
* **Categorical Variables**: Nominal (neighborhood, foundation type) and ordinal (finish quality, exterior condition).

---

## Workflow Steps

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

## Data Dictionary (Key Variables)

<details>
<summary>Click to expand Data Dictionary</summary>

## Numeric Variables

### Continuous Variables

**Lot and Exterior**

* **LotFrontage** — Linear feet of street connected to property.
* **LotArea** — Lot size in square feet.
* **MasVnrArea** — Masonry veneer area in square feet.

**Dates Built/Remodeled**

* **YearBuilt** — Original construction date.
* **YearRemodAdd** — Remodel date.
* **GarageYrBlt** — Year garage was built.

**Basement**

* **BsmtFinSF1** — Type 1 finished square feet.
* **BsmtFinSF2** — Type 2 finished square feet.
* **BsmtUnfSF** — Unfinished square feet of basement area.
* **TotalBsmtSF** — Total square feet of basement area.

**Living Area (Above Grade)**

* **1stFlrSF** — First Floor square feet.
* **2ndFlrSF** — Second floor square feet.
* **LowQualFinSF** — Low quality finished square feet (all floors).
* **GrLivArea** — Above grade (ground) living area square feet (sum of 1st and 2nd floors, excluding basement).

**Garage**

* **GarageArea** — Size of garage in square feet.

**Porch & Deck**

* **WoodDeckSF** — Wood deck area in square feet.
* **OpenPorchSF** — Open porch area in square feet.
* **EnclosedPorch** — Enclosed porch area in square feet.
* **3SsnPorch** — Three-season porch area in square feet (glass-enclosed but not winterized).
* **ScreenPorch** — Screen porch area in square feet.

### Discrete Variables

* **MSSubClass** — The building class
  * 20 - 1-STORY WITH WOOD FRAME
  * 30 - 1-STORY 1945 & OLDER
  * 40 - 1-STORY POST-1945
  * 45 - 1.5-STORY PRE-1945
  * 50 - 1.5-STORY POST-1945
  * 60 - 2-STORY WITH WOOD FRAME
  * 70 - 2-STORY PRE-1945
  * 75 - 2.5-STORY PRE-1945
  * 80 - SPLIT/MULTI-LEVEL
  * 85 - SPLIT FOYER
  * 90 - DUPLEX
  * 120 - 1-STORY PUD (Planned Unit Development)
  * 160 - 2-STORY PUD
  * 180 - MULTI-FAMILY
  * 190 - CONVERSION (Living rooms above shop/garage)


* **OverallQual** — Overall material and finish quality (1-10 from worst to best).
* **OverallCond** — Overall condition rating (1-9 from worst to best).
* **BsmtFullBath** — Basement full bathrooms.
* **BsmtHalfBath** — Basement half bathrooms.
* **FullBath** — Full bathrooms above grade.
* **HalfBath** — Half baths above grade.
* **BedroomAbvGr** — Number of bedrooms above basement level.
* **KitchenAbvGr** — Number of kitchens.
* **TotRmsAbvGrd** — Total rooms above grade (does not include bathrooms).
* **Fireplaces** — Number of fireplaces.
* **GarageCars** — Size of garage in car capacity.
  </details>

<details>
<summary>Click to expand Data Dictionary</summary>
## Categorical Variables

### Quality Variables

Ex - Excellent
Gd - Good
TA - Typical/Average
Fa - Fair
Po - Poor

* **ExterQual** — Exterior material quality.
* **ExterCond** — Present condition of the material on the exterior.
* **BsmtQual** — Height/quality of the basement.
* **BsmtCond** — General condition of the basement.
* **HeatingQC** — Heating quality and condition.
* **KitchenQual** — Kitchen quality.
* **FireplaceQu** — Fireplace quality.
* **GarageQual** — Garage quality.
* **GarageCond** — Garage condition.

### Lot and Surroundings

* **LotShape** — General shape of property
  * Reg - Regular
  * IR1 - Slightly irregular
  * IR2 - Moderately irregular
  * IR3 - Severely irregular


* **LandContour** — Flatness of the property
  * Lvl - Near Flat/Level
  * Bnk - Banked (Quick and significant rise from street grade to building)
  * Low - Depression (Below street level)
  * HLS - Hillside (Significant slope from side to side)


* **LandSlope** — Slope of property
  * Gtl - Gentle slope
  * Mod - Moderate slope
  * Sev - Severe slope


* **Condition1 & Condition2** — Proximity to main road or railroad
  * Norm - Normal
  * PosN, PosA - Near positive off-site feature (park, greenbelt, etc.)
  * Feedr - Adjacent to feeder street
  * Artery - Adjacent to arterial street
  * RRAe, RRNn, RRAn, RRNe - Adjacent to/Near Railroad


* **PavedDrive** — Paved driveway
  * Y - Paved
  * P - Partial Pavement
  * N - Dirt/Gravel



### Other Features

* **CentralAir** — Central air conditioning
  * Y - Yes
  * N - No


* **BsmtExposure** — Walkout or garden level basement walls
  * Gd - Good Exposure
  * Av - Average Exposure
  * Mn - Minimum Exposure
  * No - No Exposure


* **GarageFinish** — Interior finish of the garage
  * Fin - Finished
  * RFn - Rough Finished
  * Unf - Unfinished


* **BsmtFinType1** — Quality of basement finished area
  * GLQ - Good Living Quarters
  * ALQ - Average Living Quarters
  * BLQ - Below Average Living Quarters
  * Rec - Rec Room
  * LwQ - Low Quality
  * Unf - Unfinished


* **Electrical** — Electrical system
  * SBrkr - Standard Circuit Breakers & Romex
  * FuseA - Fuse Box over 60 AMP and all Romex wiring (Average)
  * FuseF - 60 AMP Fuse Box and mostly Romex wiring (Fair)
  * FuseP - 60 AMP Fuse Box and mostly knob & tube wiring (Poor)
  * Mix - Mixed


* **Functional** — Home functionality rating
  * Typ - Typical Functionality
  * Min1 - Minor Deductions 1
  * Min2 - Minor Deductions 2
  * Mod - Moderate Deductions
  * Maj1 - Major Deductions 1
  * Maj2 - Major Deductions 2
  * Sev - Severely Damaged
</details>
