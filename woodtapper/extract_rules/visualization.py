import numpy as np
from .utils import _from_rules_to_constraint


#######################################################
################## Print rules   ######################
#######################################################


def show_rules(
    RulesExtractorModel,
    max_rules=9,
    target_class_index=1,
    is_regression=False,
    value_mappings=None,
):
    """
    Display the rules in a structured format.

    Parameters
    ----------
    RulesExtractorModel : object
        Fitted rules extraction model.
    max_rules : int, default=9
        Max number of rules to display.
    target_class_index : int, default=1
        Class index whose probability to show (classification).
    is_regression : bool, default=False
        Switch to regression formatting.
    value_mappings : dict, optional
        {
            <feature_index or feature_name>: {
                <raw_value>: <display_string>,
                ...
            },
            ...
        }
        For binary features with both 0 and 1 mapped, rules become:
            FeatureName is <mapped_1>   (if sign_internal == "R")
            FeatureName is <mapped_0>   (if sign_internal == "L")
        (Instead of using negations.)

    """
    if (
        not hasattr(RulesExtractorModel, "rules_")
        or not hasattr(RulesExtractorModel, "list_probas_by_rules")
        or not hasattr(RulesExtractorModel, "list_probas_outside_by_rules")
    ):
        raise ValueError(
            "Model does not have the required rule attributes. Ensure it's fitted."
        )
    if is_regression and not hasattr(
        RulesExtractorModel, "list_probas_by_rules_without_coefficients"
    ):
        raise ValueError(
            "For regression, model must have 'list_probas_by_rules_without_coefficients' attribute."
        )

    list_indices_features_bin = getattr(
        RulesExtractorModel, "_list_categorical_indexes", None
    )

    rules_all = RulesExtractorModel.rules_
    if is_regression:
        probas_if_true_all = (
            RulesExtractorModel.list_probas_by_rules_without_coefficients
        )
        probas_if_false_all = (
            RulesExtractorModel.list_probas_outside_by_rules_without_coefficients
        )
        coefficients_all = RulesExtractorModel.list_coefficients_by_rules
        coeff_intercept = RulesExtractorModel.coeff_intercept
    else:
        probas_if_true_all = RulesExtractorModel.list_probas_by_rules
        probas_if_false_all = RulesExtractorModel.list_probas_outside_by_rules

    if not (len(rules_all) == len(probas_if_true_all) == len(probas_if_false_all)):
        raise ValueError("Error: Mismatch in lengths of rule attributes.")

    num_rules_to_show = min(max_rules, len(rules_all))
    if num_rules_to_show == 0:
        raise ValueError(
            "No rules to display. try to increase the number of rules extracted or check model fitting."
        )

    # Feature name mapping
    feature_mapping = None
    if hasattr(RulesExtractorModel, "feature_names_in_"):
        feature_mapping = {
            i: name for i, name in enumerate(RulesExtractorModel.feature_names_in_)
        }
    elif hasattr(RulesExtractorModel, "feature_names_"):
        if isinstance(RulesExtractorModel.feature_names_, dict):
            feature_mapping = RulesExtractorModel.feature_names_
        elif isinstance(RulesExtractorModel.feature_names_, list):
            feature_mapping = {
                i: name for i, name in enumerate(RulesExtractorModel.feature_names_)
            }

    base_ps_text = ""
    if not is_regression:
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
    header_condition = "   Condition"
    header_then = f"     THEN P(C{target_class_index})"
    header_else = f"     ELSE P(C{target_class_index})"

    max_condition_len = 0
    condition_strings_for_rules = []

    def _map_value(dim, dim_name, raw_val):
        if value_mappings is None:
            return None
        candidates = [dim]
        if dim_name is not None:
            candidates.append(dim_name)
        for c in candidates:
            if c in value_mappings:
                nested = value_mappings[c]
                if raw_val in nested:
                    return nested[raw_val]
                if isinstance(raw_val, (float, np.floating)) and int(raw_val) in nested:
                    return nested[int(raw_val)]
        return None

    def _format_binary_condition(dimension, column_name, sign_internal):
        # Determine which side of binary (0 or 1) the rule represents.
        positive_val = 1
        negative_val = 0
        # Try to map both
        mapped_pos = _map_value(dimension, column_name, positive_val)
        mapped_neg = _map_value(dimension, column_name, negative_val)

        # If both mapped, choose directly
        if mapped_pos is not None and mapped_neg is not None:
            if sign_internal == "R":  # >
                return f"{column_name} is {mapped_pos}"
            else:  # "<=" side
                return f"{column_name} is {mapped_neg}"
        # If only one mapped
        if mapped_pos is not None:
            if sign_internal == "R":
                return f"{column_name} is {mapped_pos}"
            else:
                return f"{column_name} is not {mapped_pos}"
        if mapped_neg is not None:
            if sign_internal == "L":
                return f"{column_name} is {mapped_neg}"
            else:
                return f"{column_name} is not {mapped_neg}"

        # Fallback numeric
        raw_indicator = 0 if sign_internal == "L" else 1
        return f"{column_name} is {raw_indicator}"

    for i in range(num_rules_to_show):
        current_rule_conditions = rules_all[i]
        condition_parts_str = []
        for j in range(len(current_rule_conditions)):
            dimension, treshold, sign_internal = _from_rules_to_constraint(
                rule=current_rule_conditions[j]
            )

            column_name = f"Feature[{dimension}]"
            if feature_mapping and dimension in feature_mapping:
                column_name = feature_mapping[dimension]
            elif (
                feature_mapping
                and isinstance(dimension, str)
                and dimension in feature_mapping.values()
            ):
                column_name = dimension

            is_binary = (
                list_indices_features_bin is not None
                and dimension in list_indices_features_bin
            )

            if is_binary:
                condition_parts_str.append(
                    _format_binary_condition(dimension, column_name, sign_internal)
                )
            else:
                sign_display = "<=" if sign_internal == "L" else ">"
                if isinstance(treshold, float):
                    treshold_display_raw = float(f"{treshold:.2f}")
                else:
                    treshold_display_raw = treshold
                mapped = _map_value(dimension, column_name, treshold_display_raw)
                treshold_display = (
                    mapped
                    if mapped is not None
                    else (
                        f"{treshold:.2f}"
                        if isinstance(treshold, float)
                        else str(treshold)
                    )
                )
                condition_parts_str.append(
                    f"{column_name} {sign_display} {treshold_display}"
                )

        full_condition_str = " & ".join(condition_parts_str)
        condition_strings_for_rules.append(full_condition_str)
        if len(full_condition_str) > max_condition_len:
            max_condition_len = len(full_condition_str)

    condition_col_width = max(max_condition_len, len(header_condition)) + 2

    if not is_regression:
        print(
            f"{header_condition:<{condition_col_width}} {header_then:<15} {header_else:<15}"
        )
    print("-" * (condition_col_width + 15 + 15 + 2 + 5))
    if is_regression:
        print("Intercept :", coeff_intercept)

    for i in range(num_rules_to_show):
        condition_str_formatted = condition_strings_for_rules[i]

        prob_if_true_list = probas_if_true_all[i]
        prob_if_false_list = probas_if_false_all[i]

        then_val_str = "N/A"
        else_val_str = "N/A"
        if is_regression:
            p_s_if_true = prob_if_true_list
            then_val_str = f"{p_s_if_true:.2f}"
            p_s_if_false = prob_if_false_list
            else_val_str = f"{p_s_if_false:.2f} | coeff={coefficients_all[i]:.2f}"
        else:
            if prob_if_true_list and len(prob_if_true_list) > target_class_index:
                p_s_if_true = prob_if_true_list[target_class_index] * 100
                then_val_str = f"{p_s_if_true:.0f}%"
            if prob_if_false_list and len(prob_if_false_list) > target_class_index:
                p_s_if_false = prob_if_false_list[target_class_index] * 100
                else_val_str = f"{p_s_if_false:.0f}%"

        print(
            f"if   {condition_str_formatted:<{condition_col_width}} then {then_val_str:<18} else {else_val_str:<18}"
        )
