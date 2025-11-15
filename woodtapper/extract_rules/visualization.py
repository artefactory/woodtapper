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
    Display the rules in a structured format, showing the conditions and associated probabilities for a specified
    target class (classification) or contributions (regression).

    Parameters
    ----------
    RulesExtractorModel : object
        The fitted rules extraction model containing the rules and probability / score attributes.
    max_rules : int, optional (default=9)
        The maximum number of rules to display.
    target_class_index : int, optional (default=1)
        The index of the target class for which to display probabilities (classification only).
    is_regression : bool, optional (default=False)
        If True, interpret probas lists as regression contributions instead of class probabilities.
    value_mappings : dict, optional (default=None)
        Optional mapping to display human-readable categorical levels instead of raw 0/1 or numeric codes.
        Expected structure:
            {
                <feature_id or feature_name>: {
                    <raw_value>: <display_string>,
                    ...
                },
                ...
            }
        Keys for the outer dict can be either:
            - the feature index (int) as used internally
            - the resolved feature name (str) from feature_names_in_ / feature_names_
        Nested dict maps raw threshold/indicator values (e.g., 0, 1, 2) to strings.
        For one-hot / binary encoded features:
            - If both 0 and 1 are available, the function will attempt to render
              "FeatureName is <mapped_1>" when the rule implies presence (> threshold),
              and "FeatureName is not <mapped_1>" (or "is <mapped_0>") when implying absence.
        If a mapping is missing for a value, the raw numeric value is shown.

    Returns
    -------
    None
        Prints a table of rules with their THEN / ELSE probabilities (classification) or contributions (regression).

    Notes
    -----
    1. Validates required model attributes.
    2. Builds feature name mapping if available.
    3. Applies value_mappings when provided.
    4. Handles binary / categorical formatting.
    5. Prints estimated outside probability baseline (classification).
    6. For regression, prints intercept and coefficient info.
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

    # Build feature mapping (index -> name)
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

    # Helper to resolve mapped value
    def _map_value(dim, dim_name, raw_val):
        if value_mappings is None:
            return None
        candidates = []
        # Try using index, then name
        candidates.append(dim)
        if dim_name is not None:
            candidates.append(dim_name)
        for c in candidates:
            if c in value_mappings:
                # Try exact raw_val, int(raw_val), string cast
                nested = value_mappings[c]
                if raw_val in nested:
                    return nested[raw_val]
                # If raw_val is float like 0.0 / 1.0
                if isinstance(raw_val, (float, np.floating)) and int(raw_val) in nested:
                    return nested[int(raw_val)]
        return None

    for i in range(num_rules_to_show):
        current_rule_conditions = rules_all[i]
        condition_parts_str = []
        for j in range(len(current_rule_conditions)):
            dimension, treshold, sign_internal = _from_rules_to_constraint(
                rule=current_rule_conditions[j]
            )

            # Resolve feature name
            column_name = f"Feature[{dimension}]"
            if feature_mapping and dimension in feature_mapping:
                column_name = feature_mapping[dimension]
            elif (
                feature_mapping
                and isinstance(dimension, str)
                and dimension in feature_mapping.values()
            ):
                column_name = dimension  # Already a name

            is_binary = (
                list_indices_features_bin is not None
                and dimension in list_indices_features_bin
            )

            if is_binary:
                # Interpret binary presence/absence
                # Original logic forced "is 0" / "is 1". We improve display using mapping if available.
                # sign_internal == "L" -> "<=" threshold (often absence if threshold ~0.5)
                # sign_internal == "R" -> ">" threshold (presence)
                # We still manufacture a pseudo raw indicator (0 or 1) to feed mapping.
                raw_indicator = 0 if sign_internal == "L" else 1
                mapped = _map_value(dimension, column_name, raw_indicator)

                if mapped is not None:
                    if sign_internal == "R":
                        # presence
                        condition_parts_str.append(f"{column_name} is {mapped}")
                    else:
                        # absence (choose a readable negation form)
                        # If both 0 and 1 mapped and we used 0, we can show "is {mapped}" or "is not <value_of_1>"
                        mapped_one = _map_value(dimension, column_name, 1)
                        if mapped_one is not None:
                            condition_parts_str.append(
                                f"{column_name} is not {mapped_one}"
                            )
                        else:
                            condition_parts_str.append(f"{column_name} is {mapped}")
                else:
                    # Fallback to numeric
                    sign_display = "is"
                    treshold_display = str(raw_indicator)
                    condition_parts_str.append(
                        f"{column_name} {sign_display} {treshold_display}"
                    )
            else:
                # Numeric / non-binary
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
        else:  # classification
            if prob_if_true_list and len(prob_if_true_list) > target_class_index:
                p_s_if_true = prob_if_true_list[target_class_index] * 100
                then_val_str = f"{p_s_if_true:.0f}%"
            if prob_if_false_list and len(prob_if_false_list) > target_class_index:
                p_s_if_false = prob_if_false_list[target_class_index] * 100
                else_val_str = f"{p_s_if_false:.0f}%"

        print(
            f"if   {condition_str_formatted:<{condition_col_width}} then {then_val_str:<18} else {else_val_str:<18}"
        )
