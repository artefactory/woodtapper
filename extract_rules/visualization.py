import numpy as np




#######################################################
################## Print rules   ######################
#######################################################

def show_rules(
    RulesExtractorModel, max_rules=9, target_class_index=1, list_indices_features_bin=None
):
    """
    Display the rules in a structured format, showing the conditions and associated probabilities for a specified target class.
    Parameters
    ----------
    RulesExtractorModel : object
        The fitted rules extraction model containing the rules and probabilities.
    max_rules : int, optional (default=9)
        The maximum number of rules to display.
    target_class_index : int, optional (default=1)
        The index of the target class for which to display probabilities.
    list_indices_features_bin : list of int, optional (default=None)
        List of feature indices that are binary (0/1) for special formatting.
    Returns
    ----------
    None
    1. Validate the presence of necessary attributes in the model.
    2. Extract rules and their associated probabilities.
    3. Format and display the rules in a tabular format.
    4. Include estimated average rates for the specified target class.
    5. Handle feature names for better readability, using provided mappings if available.
    6. Adjust formatting for binary features if specified.
    7. Ensure that the display is clear and informative, with appropriate headers and alignment.
    8. If the model lacks the required attributes, print an error message and exit.
    9. If there are no rules to display, print a corresponding message and exit.
    10. Calculate and display the estimated average probability for the target class based on 'else' clauses.
    11. Print the rules along with their conditions, 'then' probabilities, and 'else' probabilities in a structured table.
    """
    if (
        not hasattr(RulesExtractorModel, "all_possible_rules_list")
        or not hasattr(RulesExtractorModel, "list_probas_by_rules")
        or not hasattr(RulesExtractorModel, "list_probas_outside_by_rules")
    ):
        print(
            "Model does not have the required rule attributes. Ensure it's fitted."
        )
        return

    rules_all = RulesExtractorModel.all_possible_rules_list
    probas_if_true_all = RulesExtractorModel.list_probas_by_rules
    probas_if_false_all = RulesExtractorModel.list_probas_outside_by_rules

    if not (len(rules_all) == len(probas_if_true_all) == len(probas_if_false_all)):
        print("Error: Mismatch in lengths of rule attributes.")
        return

    num_rules_to_show = min(max_rules, len(rules_all))
    if num_rules_to_show == 0:
        print("No rules to display.")
        return

    # Attempt to build/use feature mapping
    feature_mapping = None
    if hasattr(RulesExtractorModel, "feature_names_in_"):  # Standard scikit-learn attribute
        # Create a mapping from index to name if feature_names_in_ is a list
        feature_mapping = {i: name for i, name in enumerate(RulesExtractorModel.feature_names_in_)}
    elif hasattr(RulesExtractorModel, "feature_names_"):  # Custom attribute for feature names
        if isinstance(RulesExtractorModel.feature_names_, dict):
            feature_mapping = RulesExtractorModel.feature_names_  # Assumes it's already index:name
        elif isinstance(RulesExtractorModel.feature_names_, list):
            feature_mapping = {
                i: name for i, name in enumerate(RulesExtractorModel.feature_names_)
            }
    # If no mapping, column_name will default to using indices.

    base_ps_text = ""
    if (
        probas_if_false_all
        and probas_if_false_all[0]
        and len(probas_if_false_all[0]) > target_class_index
    ):
        avg_outside_target_probas = [
            p[target_class_index]
            for p in probas_if_false_all
            if p and len(p) > target_class_index
        ]
        if avg_outside_target_probas:
            estimated_avg_target_prob = np.mean(avg_outside_target_probas) * 100
            base_ps_text = (
                f"Estimated average rate for target class {target_class_index} (from 'else' clauses) p_s = {estimated_avg_target_prob:.0f}%.\n"
                f"(Note: True average rate should be P(Class={target_class_index}) from training data).\n"
            )

    print(base_ps_text)
    header_condition = "IF Condition"
    header_then = f"     THEN P(C{target_class_index})"
    header_else = f"     ELSE P(C{target_class_index})"

    max_condition_len = 0
    condition_strings_for_rules = []

    for i in range(num_rules_to_show):
        current_rule_conditions = rules_all[i]
        condition_parts_str = []
        for j in range(len(current_rule_conditions)):
            dimension, treshold, sign_internal = RulesExtractorModel.from_rules_to_constraint(
                rule=current_rule_conditions[j]
            )

            column_name = f"Feature[{dimension}]"  # Default if no mapping
            if feature_mapping and dimension in feature_mapping:
                column_name = feature_mapping[dimension]
            elif (
                feature_mapping
                and isinstance(dimension, str)
                and dimension in feature_mapping.values()
            ):
                # If dimension is already a name that's in the mapping's values (less common for index)
                column_name = dimension
            if (
                list_indices_features_bin is not None
                and dimension in list_indices_features_bin
            ):
                sign_display = "is" #if sign_internal == "L" else "is not"
                #treshold_display = str(treshold)
                treshold_display = str(0) if sign_internal == "L" else str(1)
            else:
                sign_display = "<=" if sign_internal == "L" else ">"
                treshold_display = (
                    f"{treshold:.2f}"
                    if isinstance(treshold, float)
                    else str(treshold)
                )
            condition_parts_str.append(
                f"{column_name} {sign_display} {treshold_display}"
            )

        full_condition_str = " & ".join(condition_parts_str)
        condition_strings_for_rules.append(full_condition_str)
        if len(full_condition_str) > max_condition_len:
            max_condition_len = len(full_condition_str)

    condition_col_width = max(max_condition_len, len(header_condition)) + 2

    print(
        f"{header_condition:<{condition_col_width}} {header_then:<15} {header_else:<15}"
    )
    print("-" * (condition_col_width + 15 + 15 + 2 + 5))

    for i in range(num_rules_to_show):
        condition_str_formatted = condition_strings_for_rules[i]

        prob_if_true_list = probas_if_true_all[i]
        prob_if_false_list = probas_if_false_all[i]

        then_val_str = "N/A"
        else_val_str = "N/A"
        
        if prob_if_true_list and len(prob_if_true_list) > target_class_index:
            p_s_if_true = prob_if_true_list[target_class_index] * 100
            then_val_str = f"{p_s_if_true:.0f}%"

        if prob_if_false_list and len(prob_if_false_list) > target_class_index:
            p_s_if_false = prob_if_false_list[target_class_index] * 100
            else_val_str = f"{p_s_if_false:.0f}%"

        print(
            f"if   {condition_str_formatted:<{condition_col_width }} then {then_val_str:<18} else {else_val_str:<18}"
        )