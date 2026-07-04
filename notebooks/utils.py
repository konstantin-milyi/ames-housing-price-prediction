import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import f_oneway

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score


def tree_on_mutual_info_features(df, target, top_n=10):
    """
    Analyzes the nature of missing values in the target column.
    
    Creates a binary missingness flag (1 - missing, 0 - present) 
    and uses the Mutual Information metric to return the top-N features 
    that best predict the occurrence of these missing values.
    """
    df = df.copy()
    df['na_flag'] = df[target].isnull().astype(int)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop([target, 'na_flag', 'SalePrice', 'SalePrice_log'], axis=1).fillna(-999)
    y = df['na_flag']
     
    discrete_features = [col in cat_cols or df[col].nunique() <= 16 for col in X.columns]

    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)

    mi_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_df.head(top_n)


def missing_table(df, group_col, target):
    """
    Creates a summary table showing the distribution of missing values 
    in the target column across the categories of the group_col feature.
    
    Returns a DataFrame with the total number of rows, the number of 
    missing values, and the proportion of missing values for each category.
    """
    df = df.copy()
    df['na_flag'] = df[target].isnull().astype(int)

    grouped = df.groupby(group_col)['na_flag'].agg([
        ('total_rows', 'count'),
        ('rows_with_missing', 'sum')
    ])
    grouped['missing_percent'] = grouped['rows_with_missing'] / grouped['total_rows']

    grouped = grouped.sort_values(['rows_with_missing', 'missing_percent'], ascending=[False, False])

    return grouped.reset_index()
    

def quasi_constant(df):
    quasi_constant_feat = []
    for feature in df.columns:
        predominant = (df[feature].value_counts() / float(len(df))).sort_values(ascending=False).values[0]
        if predominant > 0.99:
            quasi_constant_feat.append(feature)

    return quasi_constant_feat


def analyze_numeric_with_graphs(df, numeric_cols, target='SalePrice_log', clip_q=0.99):
    """
    Comprehensive analysis of numeric features.
    
    For each variable, calculates statistics (Pearson, MI, Skew, outliers) 
    and plots 3 graphs: 
    1. Histogram (distribution)
    2. Q-Q plot (check for normal distribution)
    3. Scatter plot (relationship with the target, highlighting outliers in red)
    """
    for col in numeric_cols:
        x = df[col]
        y = df[target]

        # Calculate metrics and outliers
        upper = x.quantile(clip_q)
        outliers_mask = x > upper
        outliers_count = outliers_mask.sum()

        mi = mutual_info_regression(x.fillna(x.median()).values.reshape(-1, 1), y, random_state=0)[0]
        pearson_corr = stats.pearsonr(x.fillna(0), y.fillna(0))[0]
        skew_val = x.skew()

        # Plotting the graphs
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Histogram
        sns.histplot(x, kde=True, bins=30, ax=ax[0])
        ax[0].set_title(f'{col} (Skew={skew_val:.2f})')

        # 2. Q-Q plot (check for normality)
        stats.probplot(x.dropna(), dist="norm", plot=ax[1])
        ax[1].set_title(f'{col} - Q-Q Plot')

        # 3. Scatter plot
        sns.regplot(data=df[~outliers_mask], x=col, y=target,
                    scatter_kws={'alpha': 0.6}, line_kws={'color': 'blue'}, ax=ax[2])
        sns.scatterplot(data=df[outliers_mask], x=col, y=target,
                        color='red', s=60, edgecolor='black', ax=ax[2])
        
        # Output statistics in the title
        ax[2].set_title(
            f'{col} vs {target}\nPearson={pearson_corr:.2f} | MI={mi:.2f} | Outliers: {outliers_count}'
        )

        plt.tight_layout()
        plt.show()


def analyze_categorical_features(df, cat_cols, target='SalePrice'):
    """
    The function iterates over the provided list of columns, calculates statistical metrics 
    (ANOVA and Mutual Information) relative to the target for each, and plots two graphs: 
    the distribution of the target variable within categories (Boxplot) and the frequency 
    of the categories themselves in %. 
    It prints the statistical metrics and displays the plots.

    Arguments:
        df: Source dataframe containing the data.
        cat_cols: List of categorical or discrete column names.
        target: Name of the target variable column.
    """
    for var in cat_cols:
        print(var)
        
        x = df[var]
        y = df[target].copy()

        # Collect target variable values into separate lists for each unique category
        groups = [y[x == cat] for cat in x.dropna().unique()]

        # Analysis of Variance (ANOVA)
        anova_stat, anova_p = f_oneway(*groups)

        # Calculate Mutual Information (MI) between the feature and the target variable
        mi = mutual_info_regression(LabelEncoder().fit_transform(x).reshape(-1, 1), y.values, discrete_features=True, random_state=0)[0]

        print(f"ANOVA F = {anova_stat:.3f}, p = {anova_p:.4f} {'+' if anova_p < 0.05 else '-'}")
        print(f"Mutual Information = {mi:.4f}")

        colors = sns.color_palette("deep").as_hex()

        plt.figure(figsize=(14, 5))

        # Boxplot with order sorted by descending category medians 
        plt.subplot(1, 2, 1)
        median_order = df.groupby(var)[target].median().sort_values(ascending=False).index
        sns.boxplot(data=df[[var, target]].copy(), x=var, y=target, order=median_order, color=colors[0])
        plt.xticks(rotation=45)
        plt.title('Boxplot SalePrice (Sorted by Median)')
        plt.xlabel(var)
        plt.ylabel('Sale Price')

        # Category frequency (in percent)
        plt.subplot(1, 2, 2)
        percents = (df[var].value_counts() / len(df)) * 100
        ax2 = percents.plot.bar(color=colors[0], edgecolor='black')
        labels = [f'{p:.1f}' for p in percents]
        plt.xticks(rotation=45)
        ax2.set_title('Category Frequency (%)')
        ax2.bar_label(ax2.containers[0], labels=labels)
        
        plt.tight_layout()
        plt.show()


def lotfrontage_impute(X):
    medians = X.groupby('Neighborhood')['LotFrontage'].transform('median')
    X['LotFrontage'] = X['LotFrontage'].fillna(medians)
    return X 


manual_mappings = {
    'LotShape': {'Reg': 0, 'IR1': 1},
    'ExterQual': {'TA': 0, 'Gd': 1},
    'BsmtQual': {'Missing': 0, 'TA': 1, 'Gd': 2, 'Ex': 3},
    'BsmtExposure': {'Missing': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'HeatingQC': {'TA': 0, 'Gd': 1, 'Ex': 2},
    'CentralAir': {'N': 0, 'Y': 1},
    'Electrical': {'Rare': 0, 'SBrkr': 1},
    'KitchenQual': {'TA': 0, 'Gd': 1, 'Ex': 2},
    'FireplaceQu': {'Missing': 0, 'TA': 1, 'Gd': 2},
    'GarageFinish': {'Missing': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'PavedDrive': {'N': 0, 'Y': 1},
    'LotConfig': {'FR2': 0, 'FR3': 1}
}

def manual_ordinal_encoding(df):
    mappings = manual_mappings
    for col, mapping in mappings.items():
        df[col] = df[col].replace(mapping)
    return df


binning_col = ['MasVnrArea']

def custom_quantile_binning(df):
    variables = binning_col
    for var in variables:
        non_zero = df[var][df[var] != 0]
        q1 = non_zero.quantile(0.5)  
        df[var] = np.select(
            [
                df[var] == 0,
                (df[var] > 0) & (df[var] <= q1),
                df[var] > q1
            ],
            [0, 1, 2]
        )
    return df


def plot_metrics(model, X_train, y_train, X_test, y_test, name="Model"):
    """Calculates RMSE and R2 metrics, prints text and bar charts."""
    
    # Predictions and inverse logarithm
    y_tr_exp, y_te_exp = np.expm1(y_train), np.expm1(y_test)
    p_tr_exp, p_te_exp = np.expm1(model.predict(X_train)), np.expm1(model.predict(X_test))

    # Calculate metrics
    rmse_tr, rmse_te = np.sqrt(mean_squared_error(y_tr_exp, p_tr_exp)), np.sqrt(mean_squared_error(y_te_exp, p_te_exp))
    r2_tr, r2_te = r2_score(y_tr_exp, p_tr_exp), r2_score(y_te_exp, p_te_exp)

    print(f"--- {name} ---")
    print(f"R2 Diff: {(r2_tr - r2_te):.4f}\n")

    # Plots
    # --- R2 and RMSE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = sns.color_palette("deep").as_hex()

    ax1.bar(['Train', 'Test'], [r2_tr, r2_te], color=colors, edgecolor='black', linewidth=1.2)
    ax1.set(title='R2 Score', ylim=(0, 1.1))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.bar_label(ax1.containers[0], fmt='%.4f', padding=3)

    ax2.bar(['Train', 'Test'], [rmse_tr, rmse_te], color=colors, edgecolor='black', linewidth=1.2)
    ax2.set(title='RMSE ($)', ylim=(0, max(rmse_tr, rmse_te) * 1.15))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar_label(ax2.containers[0], labels=[f"{rmse_tr:,.0f}", f"{rmse_te:,.0f}"], padding=3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, X_train, name="Model"):
    """Extracts feature weights from the pipeline (accounting for SFS) and plots a bar chart."""
    preprocessor = model.named_steps['preprocessor']
    best_model = model.named_steps['model'].best_estimator_
    
    # Get column names after preprocessing 
    X_train_pre = preprocessor.transform(X_train)
    if hasattr(X_train_pre, 'columns'):
        feature_names = np.array(X_train_pre.columns)
    else:
        feature_names = preprocessor.get_feature_names_out()

    # Filter for SFS
    if 'sfs' in model.named_steps:
        sfs = model.named_steps['sfs']
        selected_idx = list(sfs.k_feature_idx_)
        feature_names = feature_names[selected_idx]
    else:
        feature_names = feature_names 

    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    importances.sort_values(ascending=False, inplace=True)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    importances.plot.bar(color=sns.color_palette("deep").as_hex()[0], edgecolor='black')
    plt.title(f'{name} - Feature Importances', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Weight')
    plt.xticks(rotation=90, fontsize=8) 
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(model, X_train, y_train, X_test, y_test, name="Model"):
    """Plots 'Actual vs Predicted' scatter plots for Train and Test."""
    
    # Predictions and inverse logarithm
    y_tr_exp, y_te_exp = np.expm1(y_train), np.expm1(y_test)
    p_tr_exp, p_te_exp = np.expm1(model.predict(X_train)), np.expm1(model.predict(X_test))

    colors = sns.color_palette("deep").as_hex()
    
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Common limits for axes
    min_val = min(y_tr_exp.min(), y_te_exp.min(), p_tr_exp.min(), p_te_exp.min())
    max_val = max(y_tr_exp.max(), y_te_exp.max(), p_tr_exp.max(), p_te_exp.max())

    # Left plot - Train
    sns.scatterplot(x=y_tr_exp, y=p_tr_exp, alpha=0.5, color=colors[0], ax=ax_tr, edgecolor='black')
    ax_tr.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax_tr.set(title=f'{name} - Train: Actual vs Predicted', xlabel='Actual Value ($)', ylabel='Predicted Value ($)')
    ax_tr.grid(True, linestyle='--', alpha=0.5)

    # Right plot - Test
    sns.scatterplot(x=y_te_exp, y=p_te_exp, alpha=0.5, color=colors[1], ax=ax_te, edgecolor='black')
    ax_te.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax_te.set(title=f'{name} - Test: Actual vs Predicted', xlabel='Actual Value ($)', ylabel='Predicted Value ($)')
    ax_te.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()