### IMPORTS ###

#utility imports
import pandas as pd
import numpy as np
from copy import deepcopy
import timeit
import random
from .models import myModel

# Counterfactual imports
from raiutils.exceptions import UserConfigValidationException
from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
import dice_ml
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic

#DICE Monkey-patching costumization for regression models is done at the end of this file

unequal_array = lambda a, b: np.where(np.isnan(a) | np.isnan(b), np.nan, a != b) #to handle missing values in the counterfactuals

def cf_metrics(cfs, query_instances,data_interface,plausibility,features_to_vary=None):
    """
    Computes and summarizes quality metrics for a set of counterfactuals.

    This function evaluates the generated counterfactuals (CFs) based on three key criteria:
    1.  **Proximity**: How close the CFs are to the original instances. Measured by the L1 norm
        for 'Insulin' and 'CHO' features.
    2.  **Sparsity**: How many features were changed to create the CFs. A higher sparsity
        score means fewer changes, which is generally desirable.
    3.  **Physiological Plausibility**: Whether the changes in the CFs make sense from a
        physiological standpoint, based on pre-computed plausibility assessments.

    Parameters
    ----------
    cfs : pd.DataFrame
        A DataFrame containing the generated counterfactual examples. Each row is a CF.
    query_instances : pd.DataFrame
        The original instances for which the counterfactuals were generated.
    data_interface : object
        A data interface object from DiCE. Currently unused but kept for API consistency.
    plausibility : pd.DataFrame
        A DataFrame containing the results of the physiological plausibility assessment.
        It must include 'plausibility' and 'type' ('hypo' or 'hyper') columns.
    features_to_vary : list, optional
        A list of feature names to consider when calculating metrics. If None, all
        features in the counterfactuals DataFrame are used. Defaults to None.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics:
        - 'plausibility': A dictionary with counts of plausible, implausible, ambiguous, and
                          not-found results, broken down by 'hypo' and 'hyper' types.
        - 'proximity': A dictionary with L1 proximity scores for 'Insulin' and 'CHO' features.
        - 'sparsity': A dictionary with sparsity scores for 'Insulin' and 'CHO' features.
    """
    # Use all features if 'features_to_vary' is not specified
    features_to_vary = features_to_vary if features_to_vary is not None else cfs.columns
    # Calculate the difference between original and counterfactual instances
    diff= query_instances.loc[:,features_to_vary] - cfs.loc[:,features_to_vary]

    # Identify features related to insulin and CHO
    ins_features = [feat for feat in features_to_vary if "Insulin" in feat]
    cho_features = [feat for feat in features_to_vary if "CHO" in feat]

    # Calculate L1 norm for proximity (lower is better)
    proximity_ins = np.linalg.norm(diff[ins_features].values, axis=1,ord=1)
    proximity_cho = np.linalg.norm(diff[cho_features].values, axis=1,ord=1)
    proximity = {
        "Insulin": proximity_ins,
        "CHO": proximity_cho
    }

    # Calculate sparsity (higher is better, 1 means no change)
    sparsity_ins = 1-np.mean(unequal_array(diff[ins_features].values, 0), axis=1)
    sparsity_cho = 1-np.mean(unequal_array(diff[cho_features].values, 0), axis=1)
    sparsity = {
        "Insulin": sparsity_ins,
        "CHO": sparsity_cho
    }

    # Analyze physiological plausibility results
    categories = {
        "Ambiguous": plausibility['plausibility'] == "ambiguous",
        "Not found": plausibility['plausibility'].isna(),
        "Plausible": plausibility['plausibility'] == True,
        "Implausible": plausibility['plausibility'] == False
    }

    # Count plausibility outcomes for hypo and hyper glycemic events
    results = {}
    for cat, mask in categories.items():
        for t in ["hypo", "hyper"]:
            key = f"{cat}_{t}"
            results[key] = len(plausibility[mask & (plausibility['type'] == t)])

    return {'plausibility': results, 'proximity': proximity, 'sparsity' : sparsity}

def phys_plausibility(cfs:pd.DataFrame,queries:pd.DataFrame,features_to_vary=None,type_phy:str="hypo")-> pd.DataFrame:
    """
    Assesses the physiological plausibility of counterfactuals based on domain rules.

    This function checks if the changes made in a counterfactual (CF) are physiologically
    sensible for correcting hypoglycemia or hyperglycemia.

    Plausibility Rules:
    - **For Hyperglycemia (high blood sugar)**:
      - Plausible: Increase insulin OR decrease CHO.
      - Implausible: Decrease insulin OR increase CHO.
    - **For Hypoglycemia (low blood sugar)**:
      - Plausible: Decrease insulin OR increase CHO.
      - Implausible: Increase insulin OR decrease CHO.
    - **Ambiguous**: If both insulin and CHO are changed in the same direction
      (e.g., both increased), the effect is unpredictable.
    - **Not Found**: If the genetic algorithm failed to find a valid CF, it's marked as NaN.

    Parameters
    ----------
    cfs : pd.DataFrame
        A DataFrame of counterfactual examples.
    queries : pd.DataFrame
        The original query instances corresponding to the counterfactuals.
    features_to_vary : list, optional
        The features to be evaluated for plausibility. If None, all features are
        considered. Defaults to None.
    type_phy : str, optional
        The physiological context, either 'hypo' for hypoglycemia or 'hyper' for
        hyperglycemia. Defaults to "hypo".

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - 'plausibility': A boolean (True/False), "ambiguous", or np.nan.
        - 'type': The physiological context ('hypo' or 'hyper').
    """
    # Use all features if not specified
    features_to_vary= features_to_vary if features_to_vary is not None else cfs.columns
    # Identify CHO and insulin-related features
    features_CHO = [feat for feat in features_to_vary if "CHO" in feat]
    features_insulin = [feat for feat in features_to_vary if "Insulin" in feat]

    # --- Plausibility Checks ---
    # Check for NaN values (indicates CF not found)
    nan_flag = np.any(np.isnan(cfs[features_to_vary].values), axis=1)
    # Check for changes in CHO/insulin intake
    more_carbs= np.any(cfs[features_CHO] > queries[features_CHO], axis=1)
    more_insulin = np.any(cfs[features_insulin] > queries[features_insulin], axis=1)
    less_carbs = np.any(cfs[features_CHO] < queries[features_CHO], axis=1)
    less_insulin = np.any(cfs[features_insulin] < queries[features_insulin], axis=1)

    # Flag cases that are physiologically ambiguous (both inputs changed in the same direction)
    flag_ambiguous = np.logical_or(
        np.logical_and(more_carbs, more_insulin),
        np.logical_and(less_carbs, less_insulin)
    )

    # Apply plausibility rules based on the physiological context ('hypo' or 'hyper')
    if type_phy == "hypo":
        # For low blood sugar, plausible to decrease insulin or increase carbs
        plausibility =  np.logical_or(less_insulin, more_carbs)
    elif type_phy == "hyper":
        # For high blood sugar, plausible to increase insulin or decrease carbs
        plausibility =  np.logical_or(more_insulin, less_carbs)
    else:
        raise ValueError("type_phy must be either 'hypo' or 'hyper'")

    # Set plausibility status for ambiguous and unfound cases
    plausibility[flag_ambiguous] = "ambiguous"
    plausibility[nan_flag] = np.nan

    # Create a DataFrame to store the results
    plausibility_df = pd.DataFrame({
        'plausibility': plausibility,
        'type': type_phy
    })

    return plausibility_df

def linear_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse,
                    current_pred_orig, limit_steps_ls):
    """
    Performs a linear search to make a counterfactual sparser.

    This method iteratively adjusts a continuous feature of a counterfactual, moving its
    value closer to the original query instance's value. The search stops when the
    counterfactual is no longer valid (i.e., the model's prediction falls outside the
    desired range) or the maximum number of steps is reached. This helps to find the
    minimal change required for a feature, thus increasing sparsity.

    This function is intended to monkey-patch a method in the DiCE library.

    Parameters
    ----------
    self : object
        The DiCE explainer instance.
    diff : float
        The initial difference between the counterfactual and query feature values.
    decimal_prec : dict
        A dictionary mapping feature names to their decimal precision.
    query_instance : pd.DataFrame
        The original instance being explained.
    cf_ix : int
        The index of the counterfactual being modified in `final_cfs_sparse`.
    feature : str
        The name of the feature to be adjusted.
    final_cfs_sparse : pd.DataFrame
        The DataFrame of counterfactuals being refined for sparsity.
    current_pred_orig : np.ndarray
        The original prediction of the counterfactual before this adjustment.
    limit_steps_ls : int
        The maximum number of steps to perform in the linear search.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the adjusted (sparser) counterfactual.
    """

    # First, check if setting the feature to the query value directly is valid
    old_val = final_cfs_sparse.at[cf_ix, feature]
    final_cfs_sparse.at[cf_ix, feature] = query_instance[feature].iat[0]
    # Prediction of the query instance
    current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])

    # If it's still a valid CF, we are done with this feature
    if self.is_cf_valid(current_pred):
        return final_cfs_sparse
    else:
        # Otherwise, revert to the last valid value
        final_cfs_sparse.at[cf_ix, feature] = old_val

    old_diff = diff
    change = (10.**-decimal_prec[feature])  # Minimal change possible for the feature
    current_pred = current_pred_orig
    count_steps = 0

    # Iteratively adjust the feature value towards the original query value
    while ((abs(diff) > 10e-4) and (np.sign(diff*old_diff) > 0) and
            self.is_cf_valid(current_pred)) and (count_steps < limit_steps_ls):

        old_val = final_cfs_sparse.at[cf_ix, feature]
        # Move the feature value by a small amount
        final_cfs_sparse.at[cf_ix, feature] += np.sign(diff)*change
        current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])
        old_diff = diff

        # If the CF becomes invalid, revert the last change and return
        if not self.is_cf_valid(current_pred):
            final_cfs_sparse.at[cf_ix, feature] = old_val
            return final_cfs_sparse

        # Update the difference for the next iteration
        diff = query_instance[feature].iat[0] - final_cfs_sparse.at[cf_ix, feature]
        count_steps += 1

    return final_cfs_sparse

def random_init(self, num_inits, features_to_vary, query_instance, desired_class, desired_range,
                max_attempts=50):
    """
    Initializes a population of candidate counterfactuals using random sampling.

    This function generates starting points for the genetic algorithm by randomly
    sampling feature values within their permitted ranges. It attempts to find
    initializations that are already valid counterfactuals.

    This function is intended to monkey-patch a method in the DiCE library.

    Parameters
    ----------
    self : object
        The DiCE explainer instance.
    num_inits : int
        The number of initial counterfactuals to generate (population size).
    features_to_vary : list
        The list of feature names that are allowed to be changed.
    query_instance : np.ndarray
        The encoded original instance.
    desired_class : any
        The target class for classification problems (unused in this regression context).
    desired_range : tuple
        The desired outcome range for regression problems.
    max_attempts : int, optional
        The maximum number of attempts to find a valid initialization before giving up.
        Defaults to 50.

    Returns
    -------
    np.ndarray
        An array of shape (num_inits, number_of_features) containing the initialized
        counterfactual vectors.
    """
    max_attempts = min(max_attempts, num_inits)
    remaining_cfs = np.zeros((num_inits, self.data_interface.number_of_features))
    kx = 0  # Number of valid initializations found
    counter = 0  # Number of attempts
    precisions = self.data_interface.get_decimal_precisions()
    non_valid_inits = list()

    # Loop until enough valid initializations are found or max attempts are reached
    while kx < num_inits and counter < max_attempts:
        one_init = np.zeros(self.data_interface.number_of_features)
        for jx, feature in enumerate(self.data_interface.feature_names):
            if feature in features_to_vary:
                # Sample from the feature's range
                if feature in self.data_interface.continuous_feature_names:
                    one_init[jx] = np.round(np.random.uniform(
                        self.feature_range[feature][0], self.feature_range[feature][1]), precisions[jx])
                else: # Categorical feature
                    one_init[jx] = np.random.choice(self.feature_range[feature])
            else:
                # Keep original value for features that are not varied
                one_init[jx] = query_instance[jx]

        # Check if the random initialization is a valid counterfactual
        if self.is_cf_valid(self.predict_fn_scores(one_init)):
            remaining_cfs[kx] = one_init
            kx += 1
            counter = 0 # Reset counter if a valid init is found
        else:
            non_valid_inits.append(one_init)
        counter += 1

    # If not enough valid inits were found, warn the user and fill the rest with non-valid inits
    if kx < num_inits:
        print(f"\nWarning: only {kx} valid initializations found out of {num_inits} requested. "
            f"Consider increasing max_attempts or changing the features to vary.")
        remaining_cfs[kx:] = non_valid_inits[:num_inits - kx]

    return remaining_cfs

def generate_counterfactuals(self, query_instance, total_CFs, initialization="kdtree",
                            desired_range=None, desired_class="opposite", proximity_weight=0.2,
                            sparsity_weight=0.2, diversity_weight=5.0, categorical_penalty=0.1,
                            algorithm="DiverseCF", features_to_vary="all", permitted_range=None,
                            yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                            feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                            posthoc_sparsity_algorithm="binary", maxiterations=500, thresh=1e-2, verbose=False,
                            limit_steps_ls=100):
    """
    Generates diverse counterfactual explanations for a given query instance.

    This is the main function for generating counterfactuals. It orchestrates the DiCE
    genetic algorithm to explore the feature space and find a diverse set of valid
    explanations that meet the desired outcome. It also includes a post-hoc step to
    improve the sparsity of the results.

    This function is intended to monkey-patch a method in the DiCE library.

    Parameters
    ----------
    query_instance : dict
        The instance for which to generate counterfactuals, as a dictionary.
    total_CFs : int
        The total number of counterfactuals to generate.
    initialization : str, optional
        Method for initializing the genetic algorithm's population. Defaults to "kdtree".
    desired_range : tuple, optional
        The desired outcome range for regression problems (e.g., [80, 180]).
    desired_class : str, optional
        The desired class for classification problems. Defaults to "opposite".
    proximity_weight : float, optional
        Weight for the proximity loss component. Defaults to 0.2.
    sparsity_weight : float, optional
        Weight for the sparsity loss component. Defaults to 0.2.
    diversity_weight : float, optional
        Weight for the diversity loss component. Defaults to 5.0.
    algorithm : str, optional
        The algorithm for counterfactual generation. Defaults to "DiverseCF".
    features_to_vary : list or str, optional
        Features that are allowed to be changed. Defaults to "all".
    permitted_range : dict, optional
        A dictionary specifying the permitted range for continuous features.
    posthoc_sparsity_param : float, optional
        Parameter for post-hoc sparsity enhancement. Defaults to 0.1.
    posthoc_sparsity_algorithm : str, optional
        Algorithm for post-hoc sparsity enhancement ('linear' or 'binary'). Defaults to "binary".
    maxiterations : int, optional
        Maximum number of iterations for the genetic algorithm. Defaults to 500.
    verbose : bool, optional
        Whether to print progress messages. Defaults to False.
    limit_steps_ls : int, optional
        Maximum number of steps for the linear search in post-hoc enhancement. Defaults to 100.
    **other_params :
        Other DiCE parameters are accepted but not explicitly listed.

    Returns
    -------
    dice_ml.CounterfactualExamples
        An object containing the generated counterfactual explanations and other related data.
    """

    if not hasattr(self.data_interface, 'data_df') and initialization == "kdtree":
        raise UserConfigValidationException(
                "kd-tree initialization is not supported for private data"
                " interface because training data to build kd-tree is not available.")

    # Population size is a multiple of the number of CFs to generate
    self.population_size = 10 * total_CFs

    self.start_time = timeit.default_timer()

    # Set up features to vary, query instance, and feature weights
    features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

    # Prepare the query instance for DiCE's internal processing
    query_instance_orig = query_instance
    query_instance_orig = self.data_interface.prepare_query_instance(
            query_instance=query_instance_orig)
    query_instance = self.data_interface.prepare_query_instance(
            query_instance=query_instance)
    self.num_output_nodes = None # Unused for regression

    # Encode and transform the query instance into a numpy array
    query_instance = self.label_encode(query_instance)
    query_instance = np.array(query_instance.values[0])
    self.x1 = query_instance

    # Find the predicted value of the original query_instance
    test_pred = self.predict_fn_scores(query_instance)
    self.test_pred = test_pred
    desired_class = self.misc_init(stopping_threshold, desired_class, desired_range, test_pred[0])

    # Prepare a one-hot encoded version of the query instance
    query_instance_df_dummies = pd.get_dummies(query_instance_orig)
    for col in self.data_interface.get_all_dummy_colnames():
        if col not in query_instance_df_dummies.columns:
            query_instance_df_dummies[col] = 0

    # Initialize parameters for the genetic algorithm
    self.do_param_initializations(total_CFs, initialization, desired_range, desired_class, query_instance,
                                query_instance_df_dummies, algorithm, features_to_vary, permitted_range,
                                yloss_type, diversity_loss_type, feature_weights, proximity_weight,
                                sparsity_weight, diversity_weight, categorical_penalty, verbose)

    # Run the genetic algorithm to find counterfactuals
    query_instance_df = self.find_counterfactuals(query_instance, desired_range, desired_class, features_to_vary,
                                                maxiterations, thresh, verbose)

    # Apply post-hoc sparsity enhancement if enabled
    if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and \
            self.final_cfs is not None and self.final_cfs_df is not None and 'data_df' in self.data_interface.__dict__:
        final_cfs_df_sparse = self.final_cfs_df.copy().reset_index(drop=True)
        self.final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                query_instance_df,
                                                                posthoc_sparsity_param,
                                                                posthoc_sparsity_algorithm,
                                                                limit_steps_ls=limit_steps_ls)

    # Prepare the final output object
    desired_class_param = self.decode_model_output(pd.Series(self.target_cf_class[0]))[0] \
        if hasattr(self, 'target_cf_class') else desired_class
    return exp.CounterfactualExamples(data_interface=self.data_interface,
                                    test_instance_df=query_instance_df,
                                    final_cfs_df=self.final_cfs_df,
                                    final_cfs_df_sparse=self.final_cfs_df_sparse,
                                    posthoc_sparsity_param=posthoc_sparsity_param,
                                    desired_range=desired_range,
                                    desired_class=desired_class_param,
                                    model_type=self.model.model_type)

def posthoc_sparsity_enhancement(self, final_cfs_sparse, query_instance, posthoc_sparsity_param,
                                    posthoc_sparsity_algorithm, limit_steps_ls, features_to_vary=None):
    """
    Applies a post-hoc method to improve the sparsity of generated counterfactuals.

    This method iterates through the features of a counterfactual and attempts to revert
    them to their original values from the query instance, as long as the counterfactual
    remains valid. This process is done feature by feature, ordered by their deviation
    from the median in the training data.

    This function is intended to monkey-patch a method in the DiCE library.

    Parameters
    ----------
    self : object
        The DiCE explainer instance.
    final_cfs_sparse : pd.DataFrame
        The DataFrame of final counterfactuals to be made sparser.
    query_instance : pd.DataFrame
        The original query instance.
    posthoc_sparsity_param : float
        A parameter controlling the quantile for feature deviation, used for sorting.
    posthoc_sparsity_algorithm : str
        The search algorithm to use: 'linear' or 'binary'.
    limit_steps_ls : int
        The maximum number of steps for the linear search.
    features_to_vary : list, optional
        A list of features that are allowed to be changed. If None, all features that
        were varied during generation are considered. Defaults to None.

    Returns
    -------
    pd.DataFrame
        The DataFrame with sparser counterfactuals.
    """
    if final_cfs_sparse is None:
        return final_cfs_sparse

    # Get quantiles of deviation from the median for each continuous feature
    quantiles = self.data_interface.get_quantiles_from_training_data(quantile=posthoc_sparsity_param)

    # Sort features by their quantile deviation in descending order
    features_sorted = sorted(quantiles.items(), key=lambda kv: kv[1], reverse=True)
    features_sorted = [item[0] for item in features_sorted]

    # Filter to only include features that are allowed to vary
    if features_to_vary is not None:
        features_sorted = [feat for feat in features_sorted if feat in features_to_vary]

    # Get decimal precisions for rounding
    precs = np.minimum(self.data_interface.get_decimal_precisions(), 2)
    decimal_prec = dict(zip(self.data_interface.continuous_feature_names, precs))

    cfs_preds_sparse = []

    # Process each counterfactual individually
    for cf_ix in list(final_cfs_sparse.index):
        current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])
        # Iterate through sorted features to attempt sparsity enhancement
        for feature in features_sorted:
            diff = query_instance[feature].iat[0] - final_cfs_sparse.at[cf_ix, feature]
            if posthoc_sparsity_algorithm == "linear":
                final_cfs_sparse = self.do_linear_search(diff, decimal_prec, query_instance, cf_ix,
                                                            feature, final_cfs_sparse, current_pred, limit_steps_ls)
            elif posthoc_sparsity_algorithm == "binary":
                final_cfs_sparse = self.do_binary_search(
                    diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred)

        # Store the final prediction for the sparse CF
        temp_preds = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])
        cfs_preds_sparse.append(temp_preds[0])

    # Update the outcome column with the new predictions
    final_cfs_sparse[self.data_interface.outcome_name] = self.get_model_output_from_scores(cfs_preds_sparse)
    return final_cfs_sparse

def find_counterfactuals(self, query_instance, desired_range, desired_class,
                        features_to_vary, maxiterations, thresh, verbose):
    """
    Evolves a population of candidates using a genetic algorithm to find counterfactuals.

    This is the core of the genetic search. It iteratively applies selection, crossover,
    and mutation to a population of candidate solutions to find individuals that satisfy
    the counterfactual condition (i.e., produce a prediction within the `desired_range`)

    This function is intended to monkey-patch a method in the DiCE library.

    Parameters
    ----------
    self : object
        The DiCE explainer instance.
    query_instance : np.ndarray
        The encoded feature vector of the query sample.
    desired_range : tuple
        The allowed prediction interval for a valid counterfactual in regression.
    desired_class : any
        The target class for classification (not used here).
    features_to_vary : list
        A list of feature names that can be mutated and crossed over.
    maxiterations : int
        The maximum number of generations for the algorithm to run.
    thresh : float
        The convergence threshold for the loss function to trigger early stopping.
    verbose : bool
        If True, prints a summary of the search outcome and elapsed time.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the original query instance, including its prediction.
    """
    population = self.cfs.copy()
    iterations = 0
    previous_best_loss = -np.inf
    current_best_loss = np.inf
    stop_cnt = 0
    cfs_preds = [np.inf] * self.total_CFs
    to_pred = None
    original_population = deepcopy(population)

    # Normalize the query instance for distance calculations
    self.query_instance_normalized = self.data_interface.normalize_data(self.x1)
    self.query_instance_normalized = self.query_instance_normalized.astype('float')

    while iterations < maxiterations and self.total_CFs > 0:
        # Check for early stopping condition
        if abs(previous_best_loss - current_best_loss) <= thresh and \
                (self.model.model_type == ModelTypes.Classifier and all(i == desired_class for i in cfs_preds) or
                (self.model.model_type == ModelTypes.Regressor and
                all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds))):
            stop_cnt += 1
        else:
            stop_cnt = 0
        if stop_cnt >= 5:
            break

        previous_best_loss = current_best_loss
        # Ensure population has unique individuals
        population = np.unique(tuple(map(tuple, population)), axis=0)
        if len(population) == 1:
            # Add a random individual if population collapses
            population = np.vstack((population, random.choice(original_population)))

        # Calculate fitness for each individual in the population
        population_fitness = self.compute_loss(population, desired_range, desired_class)
        population_fitness = population_fitness[population_fitness[:, 1].argsort()] # Sort by fitness

        current_best_loss = population_fitness[0][1]
        # Get the top individuals based on total_CFs
        to_pred = np.array([population[int(tup[0])] for tup in population_fitness[:self.total_CFs]])

        if self.total_CFs > 0:
            if self.model.model_type == ModelTypes.Classifier:
                cfs_preds = self._predict_fn_custom(to_pred, desired_class)
            else:
                cfs_preds = self.predict_fn(to_pred)

        # Elitism: Keep the top `total_CFs` members for the next generation
        top_members = self.total_CFs
        new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])

        # Crossover: Create the rest of the new generation from the fittest half
        rest_members = self.population_size - top_members
        new_generation_2 = None
        if rest_members > 0:
            new_generation_2 = np.zeros((rest_members, self.data_interface.number_of_features))
            for new_gen_idx in range(rest_members):
                parent1 = random.choice(population_fitness[:int(len(population_fitness) / 2),0])
                parent2 = random.choice(population_fitness[:int(len(population_fitness) / 2),0])
                child = self.mate(population[int(parent1)], population[int(parent2)], features_to_vary, query_instance)
                new_generation_2[new_gen_idx] = child

        # Combine elite members and new children to form the next generation's population
        if new_generation_2 is not None:
            population = np.concatenate([new_generation_1, new_generation_2])
        else:
            # This should not happen if population_size > total_CFs
            raise SystemError("The number of total_Cfs is greater than the population size!")
        iterations += 1

    # After evolution, collect the final valid counterfactuals
    self.cfs_preds = []
    self.final_cfs = []
    i = 0
    while i < self.total_CFs:
        predictions = self.predict_fn_scores(population[i])[0]
        if self.is_cf_valid(predictions):
            self.final_cfs.append(population[i])
            # checking if predictions is a float before taking the length as len() works only for array-like
            # elements. isinstance(predictions, (np.floating, float)) checks if it's any float (numpy or otherwise)
            # We do this as we take the argmax if the prediction is a vector -- like the output of a classifier
            if not isinstance(predictions, (np.floating, float)) and len(predictions) > 1:
                self.cfs_preds.append(np.argmax(predictions))
            else:
                self.cfs_preds.append(predictions)
        i += 1

    # Convert final counterfactuals to a DataFrame and decode them
    query_instance_df = self.label_decode(query_instance)
    query_instance_df[self.data_interface.outcome_name] = self.get_model_output_from_scores(self.test_pred)
    self.final_cfs_df = self.label_decode_cfs(self.final_cfs)
    self.final_cfs_df_sparse = deepcopy(self.final_cfs_df)

    if self.final_cfs_df is not None:
        self.final_cfs_df[self.data_interface.outcome_name] = self.cfs_preds
        self.final_cfs_df_sparse[self.data_interface.outcome_name] = self.cfs_preds

    # Decode categorical features back to their original labels
    query_instance_df, self.final_cfs_df, self.final_cfs_df_sparse = \
        self.decode_to_original_labels(query_instance_df, self.final_cfs_df, self.final_cfs_df_sparse)

    self.elapsed = timeit.default_timer() - self.start_time
    m, s = divmod(self.elapsed, 60)

    if verbose:
        if len(self.final_cfs) == self.total_CFs:
            print('Diverse Counterfactuals found! total time taken: %02d' %
                m, 'min %02d' % s, 'sec')
        else:
            print('Only %d (required %d) ' % (len(self.final_cfs), self.total_CFs),
                'Diverse Counterfactuals found for the given configuation, perhaps ',
                'change the query instance or the features to vary...'  '; total time taken: %02d' % m,
                'min %02d' % s, 'sec')

    return query_instance_df

def binary_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred):
    """
    Performs a binary search to make a counterfactual sparser.

    This method is an alternative to linear search for post-hoc sparsity enhancement.
    It efficiently finds the point at which a feature's value can be moved towards the
    original query value before the counterfactual becomes invalid. It is generally
    faster than linear search but assumes a monotonic relationship between the feature
    and the model's output.

    This function is intended to monkey-patch a method in the DiCE library.

    Parameters
    ----------
    self : object
        The DiCE explainer instance.
    diff : float
        The signed difference between the query and counterfactual feature values.
    decimal_prec : dict
        A dictionary mapping feature names to their decimal precision.
    query_instance : pd.DataFrame
        The original instance being explained.
    cf_ix : int
        The index of the counterfactual being modified in `final_cfs_sparse`.
    feature : str
        The name of the feature to be adjusted.
    final_cfs_sparse : pd.DataFrame
        The DataFrame of counterfactuals being refined for sparsity.
    current_pred : np.ndarray
        The prediction of the counterfactual before this adjustment.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the adjusted (sparser) counterfactual.
    """
    valid_val = final_cfs_sparse.at[cf_ix, feature] # Store the last known valid value
    # First, check if setting the feature to the query value directly is valid
    final_cfs_sparse.at[cf_ix, feature] = query_instance[feature].iat[0]
    # Prediction of the query instance
    current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])

    if self.is_cf_valid(current_pred):
        return final_cfs_sparse # If it's still valid, we're done
    else:
        # Otherwise, revert to the last valid value and start the search
        final_cfs_sparse.at[cf_ix, feature] = valid_val

    # Binary search logic
    if diff > 0: # Search from current CF value up to the query value
        left = final_cfs_sparse.at[cf_ix, feature]
        right = query_instance[feature].iat[0]

        while left <= right:
            mid_point = left + ((right - left) / 2)
            current_val = round(mid_point, decimal_prec[feature])

            final_cfs_sparse.at[cf_ix, feature] = current_val
            current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])

            if current_val == right or current_val == left:
                break

            if self.is_cf_valid(current_pred):
                valid_val = current_val # Update the last valid value
                left = current_val + (10. ** -decimal_prec[feature])
            else:
                right = current_val - (10. ** -decimal_prec[feature])

    else: # Search from query value up to the current CF value
        left = query_instance[feature].iat[0]
        right = final_cfs_sparse.at[cf_ix, feature]

        while right >= left:
            mid_point = right - ((right - left) / 2)
            current_val = round(mid_point, decimal_prec[feature])

            final_cfs_sparse.at[cf_ix, feature] = current_val
            current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])

            if current_val == right or current_val == left:
                break

            if self.is_cf_valid(current_pred):
                valid_val = current_val # Update the last valid value
                right = current_val - (10.**-decimal_prec[feature])
            else:
                left = current_val + (10.**-decimal_prec[feature])

    # Set the feature to the last valid value found
    final_cfs_sparse.at[cf_ix, feature] = valid_val
    return final_cfs_sparse

def DataPreparationDice(x_train,y_train, x_explain, y_explain, PH_dice=30,Ts=5, features_label= ["BG","Insulin","CHO"] ):
    """
    Prepares time-series data for use with the DiCE library.

    This function transforms 3D time-series data (samples, timesteps, features) into a
    2D pandas DataFrame that DiCE can understand. It flattens the time steps for each
    feature into separate columns and prepares the data for a specific prediction horizon.

    Parameters
    ----------
    x_train : np.ndarray
        Training input data of shape (n_samples, timesteps, n_features).
    y_train : np.ndarray
        Training target data.
    x_explain : np.ndarray
        Input data to be explained, with the same shape format as x_train.
    y_explain : np.ndarray
        Target data for the instances to be explained.
    PH_dice : int, optional
        The prediction horizon in minutes (e.g., 30 minutes). This determines which
        target column will be used as the outcome variable. Defaults to 30.
    Ts : int, optional
        The sampling time in minutes. Defaults to 5.
    features_label : list, optional
        A list of the base feature names. Defaults to ["BG", "Insulin", "CHO"].

    Returns
    -------
    tuple
        A tuple containing:
        - data (dice_ml.Data): The DiCE data object.
        - query_instances_og (pd.DataFrame): The instances to be explained.
        - output_values_og (pd.Series): The original outcomes for the query instances.
        - feature_names (list): A list of the flattened feature names.
    """
    seq_length = x_train[0].shape[0]
    flatten_order = 'F' # Flatten column-wise to group by feature

    # Flatten the 3D arrays into 2D arrays
    x_train_flat = np.array([x.flatten(order=flatten_order) for x in x_train])
    x_train_flat = np.hstack((x_train_flat, y_train))
    x_explain_flat = np.array([x.flatten(order=flatten_order) for x in x_explain])
    x_explain_flat = np.hstack((x_explain_flat,y_explain)) #stack the examples to create a 2D array

    #array to dataframe
    columns = [f"{x}_{i}" for x in features_label for i in range(0,seq_length*Ts,Ts)] 
    columns += [f"y_{i}" for i in range(5,(seq_length+1)*Ts,Ts)] #add the label to the columns
    x_train_flat = pd.DataFrame(x_train_flat, columns=columns)
    x_explain_flat = pd.DataFrame(x_explain_flat, columns=columns) #input to dataframe

    #Dataset for providing information about data distribution to the DICE model
    x_train_flat.drop(columns=[f"y_{i}" for i in range(5,(seq_length+1)*Ts,Ts) if i != PH_dice],
                        inplace=True) #remove the outcome from the input data
    x_explain_flat.drop(columns=[f"y_{i}" for i in range(5,(seq_length+1)*Ts,Ts) if i != PH_dice],
                        inplace=True) #remove the outcome from the input data

    feature_names = x_train_flat.columns[:-1] #remove the outcome column
    data = dice_ml.Data(dataframe=x_train_flat, 
                        continuous_features=[col for col in feature_names], #-1 to remove the outcome column
                        outcome_name=f"y_{PH_dice}",
                        continuous_features_precision={
                            feat: 4 if "Insulin" in feat else 1 if "BG" in feat else 0 # 0 for CHO
                            for feat in feature_names
                        }) 
    
    #query instances and related output values
    query_instances_og = x_explain_flat.loc[:,feature_names]
    output_values_og = x_explain_flat.loc[:,f"y_{PH_dice}"]

    return data, query_instances_og, output_values_og, feature_names

def counterfactual_explanations(data, model, query_instances_og, total_cfs=1,
                                      features_to_vary=None, posthoc_sparsity_algorithm="linear",
                                      desired_range=[80,180], posthoc_sparsity_param=1):
    """
    A helper function to generate counterfactual explanations using DiCE.

    This function wraps the common workflow for generating counterfactuals:
    1. Wraps the Keras model in a DiCE-compatible adapter (`SklearnModel`).
    2. Initializes the DiCE model and explainer with the genetic algorithm.
    3. Calls the `generate_counterfactuals` method to get the explanations.

    Parameters
    ----------
    data : dice_ml.Data
        The DiCE Data object prepared with feature names and outcome.
    model : object
        A Keras model compatible with the `SklearnModel` wrapper.
    query_instances_og : pd.DataFrame
        A DataFrame with the query instances to explain (features only).
    total_cfs : int, optional
        The number of counterfactuals to generate per query instance. Defaults to 1.
    features_to_vary : list, optional
        A list of feature names that are allowed to be changed. If None, DiCE's
        default behavior is used. Defaults to None.
    posthoc_sparsity_algorithm : str, optional
        The algorithm for post-hoc sparsity enhancement ('linear' or 'binary').
        Defaults to "linear".
    desired_range : list or tuple, optional
        The target output range for the regression counterfactuals. Defaults to [80, 180].
    posthoc_sparsity_param : float, optional
        The parameter for post-hoc sparsity enhancement. Defaults to 1.

    Returns
    -------
    dice_ml.CounterfactualExamples
        The object returned by DiCE containing the counterfactuals.
    """
    # Wrap the Keras model to be compatible with DiCE
    mod = SklearnModel(model)

    # Build the DiCE model and explainer
    dice_model = dice_ml.Model(model=mod, backend="sklearn", model_type="regressor")
    cf_model = dice_ml.Dice(data, dice_model, method="genetic")

    print(f"Generating counterfactuals for {len(query_instances_og)} examples...")

    # Generate the counterfactuals
    counterfactuals = cf_model.generate_counterfactuals(
        query_instances=query_instances_og,
        initialization="random",
        total_CFs=total_cfs,
        features_to_vary=features_to_vary,
        desired_range=desired_range,
        posthoc_sparsity_param=posthoc_sparsity_param,
        posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
    )

    return counterfactuals

def counterfactual_analysis(data, model, query_instances_og, total_cfs=1,
                                      features_to_vary=None, posthoc_sparsity_algorithm="linear",
                                      desired_range=[80,180], posthoc_sparsity_param=1, PH_dice=30):
    """
    Performs a full counterfactual analysis pipeline.

    This function orchestrates the entire process:
    1. Generates counterfactual explanations using `counterfactual_explanations`.
    2. Reshapes the flat CFs back into the time-series format required by the model.
    3. Calculates the physiological plausibility of the generated CFs.
    4. Computes quality metrics (proximity, sparsity, plausibility) for the CFs.

    Parameters
    ----------
    data : dice_ml.Data
        The DiCE Data object.
    model : object
        A Keras model compatible with the `SklearnModel` wrapper.
    query_instances_og : pd.DataFrame
        A DataFrame with the query instances to explain.
    total_cfs : int, optional
        Number of counterfactuals to generate per instance. Defaults to 1.
    features_to_vary : list, optional
        List of features allowed to change. Defaults to None.
    posthoc_sparsity_algorithm : str, optional
        Sparsity algorithm ('linear' or 'binary'). Defaults to "linear".
    desired_range : list or tuple, optional
        Target output range for CFs. Defaults to [80, 180].
    posthoc_sparsity_param : float, optional
        Sparsity enhancement parameter. Defaults to 1.
    PH_dice : int, optional
        The prediction horizon in minutes. Defaults to 30.

    Returns
    -------
    tuple
        A tuple containing:
        - cfs_and_queries (dict): A dictionary storing the original and counterfactual
          instances and their predictions, formatted for saving.
        - metrics (dict): A dictionary of the calculated quality metrics.
    """
    # Generate counterfactuals
    counterfactuals = counterfactual_explanations(
        data=data,
        model=model,
        query_instances_og=query_instances_og,
        total_cfs=total_cfs,
        features_to_vary=features_to_vary,
        posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
        desired_range=desired_range,
        posthoc_sparsity_param=posthoc_sparsity_param,
    )

    
    if isinstance(model.get_model(),myModel): #for transformer models
        shape=(-1,*model.get_model().input_shape[0][1:])
    else:
        shape=(-1,*model.get_model().input_shape[1:])
    
    #function to reshape from dataframe format to model input shape
    reshape_df = lambda x: np.reshape(x.iloc[:,:-1].values, #-1 for only the features, not the target 
                        shape[1:],order='F') 

    # save the counterfactuals and the queries in a dictionary to save them in a .mat file
    tmp_dict=dict()
    for I in range(len(counterfactuals.cf_examples_list)):
        cf_object = counterfactuals.cf_examples_list[I]
        #query
        input_example = reshape_df(cf_object.test_instance_df) 
        output = model.predict([input_example],verbose=0)
        original_instance = {'input':input_example,'prediction':output}

        #counterfactuals
        if cf_object.final_cfs_df_sparse is None: #if no counterfactuals are found, fill with NaN
            cf_example = np.full_like(input_example, np.nan) 
            cf_output = np.full_like(output, np.nan)
        else:
            cf_example = reshape_df(cf_object.final_cfs_df_sparse.iloc[[0],:]) #reshape the counterfactual example to the model input shape
            cf_output = model.predict([cf_example],verbose=0)  #predict the output for the counterfactual example
        counterfactual_instance = {'input':cf_example,'prediction':cf_output}
        tmp_dict[f"Example{I}"]={'original':original_instance,'counterfactual':counterfactual_instance}
    cfs_and_queries=tmp_dict #save counterfactuals and queries for all examples

    #### Calculate physiological plausibility and metrics ####
    print("Calculating physiological plausibility and metrics...")
    query_instances, cfs = [], []
    for cf_object in counterfactuals.cf_examples_list:
        query_instances.append(cf_object.test_instance_df)
        if cf_object.final_cfs_df_sparse is None:
            # If no CF is found, create a placeholder of NaNs
            tmp = cf_object.test_instance_df.copy()
            tmp.iloc[:,:] = np.nan
            for _ in range(total_cfs):
                cfs.append(tmp)
        else:
            cfs.append(cf_object.final_cfs_df_sparse)

    # Concatenate all queries and CFs into single DataFrames
    query_instances = pd.concat(query_instances, ignore_index=True)
    cfs = pd.concat(cfs, ignore_index=True)
    query_instances = query_instances.loc[np.repeat(query_instances.index, total_cfs)].reset_index(drop=True)

    #take the counterfactuals for hypo/hyper examples
    hypo_flag= query_instances[f"y_{PH_dice}"] < 80  # flag for hypoexamples
    hypo_examples = query_instances.loc[hypo_flag]  # take the hypoexamples

    hyper_flag = query_instances[f"y_{PH_dice}"] > 180  # flag for hyperexamples
    hyper_examples = query_instances.loc[hyper_flag]  # take the hyperexamples

    hypo_cf= cfs.loc[hypo_flag]  # take the counterfactuals for the hypoexamples
    hyper_cf = cfs.loc[hyper_flag]  # take the counterfactuals for the hyperexamples

    # Calculate physiological plausibility for both cases
    hypo_phys = phys_plausibility(cfs=hypo_cf, queries=hypo_examples,
                    features_to_vary=features_to_vary, type_phy="hypo")
    hyper_phys = phys_plausibility(cfs=hyper_cf, queries=hyper_examples,
                    features_to_vary=features_to_vary, type_phy="hyper")

    # Combine results and calculate final metrics
    phys_tot = pd.concat([hypo_phys, hyper_phys], axis=0)
    metrics = cf_metrics(cfs, query_instances, data, phys_tot, features_to_vary)

    print("Counterfactual analysis completed.")

    return cfs_and_queries, metrics

class SklearnModel():
    """
    A wrapper to make Keras/Scikit-learn style models compatible with the DiCE library.

    This adapter handles two main tasks:
    1.  Reshapes the flat 2D input provided by DiCE into the 3D time-series format
        expected by the underlying model (e.g., LSTM, Transformer).
    2.  Selects the specific prediction from the model's output sequence that corresponds
        to the desired prediction horizon (`PH_dice`).

    Parameters
    ----------
    model : object
        A fitted Keras or Scikit-learn model object. It must expose a `predict` method
        and have an `input_shape` attribute (accessible via `get_model().input_shape`
        for Scikeras wrappers).
    reshape_order : {'F', 'C'}, optional
        The memory order for reshaping the input array. 'F' (Fortran/column-major) is
        used in this project to correctly group time steps by feature. Defaults to 'F'.
    PH_dice : int, optional
        The prediction horizon in minutes (e.g., 30). This is converted into an index
        to select the correct output from the model's prediction sequence. Defaults to 30.
    Ts : int, optional
        The sampling period in minutes, used to calculate the index from `PH_dice`.
        Defaults to 5.

    Attributes
    ----------
    PH_dice : int
        The calculated index for the prediction horizon.
    model : object
        The wrapped machine learning model.
    reshape_order : str
        The reshape order ('F' or 'C').
    """
    def __init__(self, model,reshape_order='F',PH_dice=30,Ts=5):
        # Convert prediction horizon from minutes to an index
        self.PH_dice = PH_dice//Ts-1
        self.model = model
        self.reshape_order = reshape_order

    def predict(self, x,**kwargs):
        """
        Generates predictions and selects the value at the specified horizon.

        Parameters
        ----------
        x : pd.DataFrame or np.ndarray
            Input data, typically a 2D array where each row is a flattened time series.
        **kwargs :
            Additional arguments passed to the underlying model's `predict` method.

        Returns
        -------
        np.ndarray
            A 1D array of scalar predictions, one for each input example.
        """
        x=self.transform(x,**kwargs)
        y = self.model.predict(x, verbose=0,**kwargs) #predicts one example at a time
        if y.ndim == 1: # only one example is passed to the model
            y=y.reshape(1,-1) #reshape to 2D array
        output=y[:,self.PH_dice] #select the output at the prediction horizon
        return output

    def transform(self, x):
        """
        Reshapes the flat input data into the 3D format required by the model.

        Parameters
        ----------
        x : pd.DataFrame or np.ndarray
            The 2D input data from DiCE.

        Returns
        -------
        list[numpy.ndarray]
            A list of numpy arrays (one per example) suitable for model.predict.
        """
        x=np.reshape(x.values, self.shape)#reshape the input to the model
        if self.reshape_order=="F":
            x=np.transpose(x,(0, 2, 1))
        x = [np.array(xi) for xi in x] #convert to list of np.arrays
        return x

    @property
    def shape(self):
        """
        Computes the target shape for reshaping the flat input.

        Returns
        -------
        tuple
            A shape tuple that can be passed to np.reshape(..., shape).

        Notes
        -----
        The implementation uses the wrapped model's input_shape (skipping batch dim).
        If reshape_order == 'C' the shape is returned as (-1, timesteps, features).
        Otherwise, for column-major handling it returns (-1, features, timesteps).
        """
        if isinstance(self.model.get_model(),myModel): #for transformer models
            shape=(-1,*self.model.get_model().input_shape[0][1:])
        else:
            shape=(-1,*self.model.get_model().input_shape[1:])
        if self.reshape_order=="C":
            self._shape = shape
        else: #column major order
            self._shape = (shape[0],shape[2],shape[1])
        return self._shape
    
# Customizing the ExplainerBase and DiceGenetic classes of DICE package, which doesn't work properly with regression models out-of-the-box
ExplainerBase.do_linear_search = linear_search 
ExplainerBase.do_binary_search = binary_search 
ExplainerBase.do_posthoc_sparsity_enhancement = posthoc_sparsity_enhancement
DiceGenetic.do_random_init = random_init
DiceGenetic._generate_counterfactuals = generate_counterfactuals 
DiceGenetic.find_counterfactuals = find_counterfactuals