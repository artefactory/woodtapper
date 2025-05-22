import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold


def get_rules_grid_p0(clf,X_train,y_train,p0_exploration_grid = np.linspace(0.01, 0.05, 15),verbose=1):
    """
    From grdi of pO values (p0_exploration_grid), compute the associated number of rules.
    """
    # --- Store results ---
    p0_rule_counts = []
    if verbose==1:
        print(f"\nStarting p0 exploration for rule counts...")
        print(f"Exploring {len(p0_exploration_grid)} p0 values from {p0_exploration_grid.min():.5f} to {p0_exploration_grid.max():.5f}")

    # --- Exploration Loop ---
    for p0_val in p0_exploration_grid:
        if verbose==1:
            print(f"  Testing p0 = {p0_val:.5f}...")
        try:
            
            sirus_model_explore = clf(
                max_depth=10,       
                random_state=0,     
                splitter="quantile" 
            )
            sirus_model_explore.fit(
                X_train, y_train,
                quantile=10,                      
                batch_size_post_treatment=50,     
                p0=p0_val
            )

            
            n_rules = len(sirus_model_explore.all_possible_rules_list)
            

            p0_rule_counts.append({'p0': p0_val, 'n_rules': n_rules})
            if verbose==1:
                print(f"    p0 = {p0_val:.5f} -> {n_rules} rules generated.")

        except Exception as e:
            print(f"    ERROR for p0 = {p0_val:.5f}: {e}")
            p0_rule_counts.append({'p0': p0_val, 'n_rules': np.nan}) # Record error as NaN
            continue

    # --- Convert results to DataFrame for easier analysis ---
    results_exploration_df = pd.DataFrame(p0_rule_counts)
    if verbose==1:
        print("\n--- Exploration Results (p0 vs. n_rules) ---")
        print(results_exploration_df)

def print_nrules_grid(results_exploration_df):
    """
    Plot the grid of the couples (p0,n_rules) obtain from get_rules_grid_p0.
    """
    # --- Plotting the results ---
    plt.figure(figsize=(12, 7))
    plt.plot(results_exploration_df['p0'], results_exploration_df['n_rules'], marker='o', linestyle='-')
    plt.xlabel("p0 value")
    plt.ylabel("Number of Generated Rules")
    plt.title("Number of Generated Rules vs. p0 Parameter")
    plt.grid(True, which="both", ls="--")

    # Add horizontal lines for the target rule range
    plt.axhline(y=1, color='red', linestyle='--', linewidth=1, label="Target Min Rules (1)")
    plt.axhline(y=25, color='green', linestyle='--', linewidth=1, label="Target Max Rules (25)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_grid_nrules(results_exploration_df,n_rules_max=25,verbose=1):
    # --- Suggesting a p0 range for your main tuning ---
    # Filter the DataFrame for p0 values that yielded rules in the 1-25 range
    target_rules_df = results_exploration_df[
        (results_exploration_df['n_rules'] >= 1) &
        (results_exploration_df['n_rules'] <= n_rules_max) &
        (results_exploration_df['n_rules'].notna())
    ].copy()

    if not target_rules_df.empty:
        min_p0_for_target_rules = target_rules_df['p0'].min()
        max_p0_for_target_rules = target_rules_df['p0'].max()
        if verbose==1:
            print(f"\n--- Suggested p0 Range for 1-25 Rules ---")
            print(f"Based on this exploration, p0 values between roughly {min_p0_for_target_rules:.5f} and {max_p0_for_target_rules:.5f} produced 1-25 rules.")
            print(f"Consider using a grid like this for your detailed tuning:")
        # Suggest a linspace with 20 points as in your original script
        suggested_tuning_grid = np.linspace(min_p0_for_target_rules, max_p0_for_target_rules, 20)
        if verbose==1:
            print(f"p0_grid = np.linspace({min_p0_for_target_rules:.5f}, {max_p0_for_target_rules:.5f}, 20)")
            print("\nRaw suggested grid values:")
            print(suggested_tuning_grid)
        return np.linspace(min_p0_for_target_rules,max_p0_for_target_rules,20)
    else:      
        raise ValueError("\n--- No p0 Range Found for 1-25 Rules --- No p0 values in the explored range consistently produced 1-25 rules.")

def train_optimal_extractor_p0(clf,X_train,y_train,scoring):
    # --- Tuning Configuration ---
    n_cv_splits = 5
    n_cv_repeats = 5 
    results_exploration_df = get_rules_grid_p0(clf,X_train,y_train,scoring)
    p0_grid = get_grid_nrules(results_exploration_df,n_rules_max=25,verbose=1) 
    p0_results_list = []

    # --- Main Tuning Loop ---
    print(f"Starting p0 tuning with {n_cv_repeats} repetitions of {n_cv_splits}-fold CV...")

    for p0_val in p0_grid:
        print(f"  Tuning for p0 = {p0_val:.4f}...")
        repetition_mean_1_auc_scores = []
        repetition_mean_n_rules = []

        for rep_num in range(n_cv_repeats):
            print(f"    Repetition {rep_num + 1}/{n_cv_repeats}")
            kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42 + rep_num) 
            fold_auc_scores_this_rep = []
            fold_n_rules_this_rep = []

            for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                try:
                    sirus_model = clf(
                        max_depth=10,
                        random_state=0, 
                        splitter="quantile"
                    )
                    sirus_model.fit(
                        X_fold_train, y_fold_train,
                        quantile=10,
                        batch_size_post_treatment=50,
                        p0=p0_val
                    )

                    y_pred_proba_val = sirus_model.predict_proba(X_fold_val)[:, 1]
                    auc_score = scoring(y_fold_val, y_pred_proba_val)
                    fold_auc_scores_this_rep.append(auc_score)

                    n_rules = len(sirus_model.all_possible_rules_list)
                    fold_n_rules_this_rep.append(n_rules)
                    print(f"{n_rules} found.")

                except Exception as e:
                    print(f"    ERROR during CV Rep {rep_num+1}, Fold {fold_num+1} for p0={p0_val}: {e}")
                    fold_auc_scores_this_rep.append(np.nan)
                    fold_n_rules_this_rep.append(np.nan)
                    continue
            
            repetition_mean_1_auc_scores.append(1 - np.nanmean(fold_auc_scores_this_rep))
            repetition_mean_n_rules.append(np.nanmean(fold_n_rules_this_rep))

        overall_mean_1_auc_for_p0 = np.nanmean(repetition_mean_1_auc_scores)
        std_of_repetition_means_1_auc = np.nanstd(repetition_mean_1_auc_scores) 
        overall_mean_n_rules_for_p0 = np.nanmean(repetition_mean_n_rules)

        p0_results_list.append({
            'p0': p0_val,
            'mean_1_auc': overall_mean_1_auc_for_p0,
            'std_of_means_1_auc': std_of_repetition_means_1_auc, 
            'mean_n_rules': overall_mean_n_rules_for_p0
        })
        print(f"  p0 = {p0_val:.4f}: Mean(1-AUC)={overall_mean_1_auc_for_p0:.4f}, Std(Means 1-AUC)={std_of_repetition_means_1_auc:.4f}, Mean_N_Rules={overall_mean_n_rules_for_p0:.2f}")

    results_df = pd.DataFrame(p0_results_list)
    print("\n--- Tuning Results (p0, mean_1_auc, std_of_means_1_auc, mean_n_rules) ---")
    print(results_df)

    # --- Select Optimal p0 ---
    optimal_p0 = None
    final_sirus_model = None

    if not results_df.empty and not results_df['mean_1_auc'].isnull().all():
        min_mean_1_auc = results_df['mean_1_auc'].min()
        print(f"\nMinimum Mean (1-AUC) found: {min_mean_1_auc:.4f}")


        best_model_by_auc_row = results_df.loc[results_df['mean_1_auc'].idxmin()]
        std_dev_for_threshold = best_model_by_auc_row['std_of_means_1_auc']
        
        threshold_auc = min_mean_1_auc + 2 * std_dev_for_threshold
        print(f"AUC Threshold (min_1_auc + 2 * std_of_means_1_auc_of_best_model): {threshold_auc:.4f}")

        candidate_models_df = results_df[results_df['mean_1_auc'] <= threshold_auc].copy() 
        print("\n--- Candidate Models (mean_1_auc <= threshold) ---")
        print(candidate_models_df)

        if not candidate_models_df.empty:
            candidate_models_df.sort_values(by=['mean_n_rules', 'p0'], ascending=[True, False], inplace=True)
            optimal_row = candidate_models_df.iloc[0] 
            
            optimal_p0 = optimal_row['p0']
            optimal_n_rules = optimal_row['mean_n_rules']
            optimal_1_auc = optimal_row['mean_1_auc']

            print(f"\n--- Optimal Selection ---")
            print(f"Optimal p0: {optimal_p0:.4f}")
            print(f"Achieved Mean (1-AUC): {optimal_1_auc:.4f}")
            print(f"Achieved Mean Number of Rules: {optimal_n_rules:.2f}")

            print("\nRetraining final model with optimal p0 on full training data...")
            final_sirus_model = clf(
                max_depth=10, random_state=0, splitter="quantile"
            )
            final_sirus_model.fit(
                X_train, y_train, quantile=10, batch_size_post_treatment=50, p0=optimal_p0
            )
            print("Final model trained.")
        else:
            print("\nNo candidate models found within the threshold. Optimal p0 not determined.")
    else:
        print("\nNo valid results from tuning. Optimal p0 not determined.")

    if optimal_p0 is not None:
        print(f"\nThe selected optimal p0 value is: {optimal_p0}")
    else:
        print("\nOptimal p0 could not be determined.")

#TODO : enlver les commentaires inutiles