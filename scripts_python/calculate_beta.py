import numpy as np
import pandas as pd
import math
#%% Load data
participants = pd.read_csv('../data_preprocessed/2023-02-16_participants__an-exp2_n-24.csv', sep=",")
transitions = pd.read_csv('../data_preprocessed/2023-02-17_transitions_an-exp2_n-24.csv', sep=",")
trials_learning = pd.read_csv('../data_preprocessed/2023-02-17_learning_an-exp2_n-24.csv', sep=",")
trials_prediction = pd.read_csv('../data_preprocessed/2023-02-17_test_prediction__an-exp2_n-8.csv', sep=",")
trials_control = pd.read_csv('../data_preprocessed/2023-02-17_test_control__an-exp2_n-8.csv', sep=",")
trials_explanation = pd.read_csv('../data_preprocessed/2023-02-17_test_explanation__an-exp2_n-8.csv', sep=',')
#%% Define functions
import numpy as np

def predict(start_state, first_input, second_input, fsm_temp, mode="normative"):
    """Return an array with probabilities of each final state starting from state 0 to state 4. E.g.: [0.4, 0.6, 0, 0].
    Higher probability = correct prediction."""
    states = [0, 1, 2, 3]
    if mode == "normative":
        return np.array([np.array([fsm_temp[start_state][first_input][middle_state] *
                                   fsm_temp[middle_state][second_input][final_state]
                                   for middle_state in states]).sum()
                         for final_state in states],
                        dtype=[(f"{start_state}—{first_input}–[All paths]–{second_input}–?", "float16")])
    elif mode == "an":
        max_middle_state = np.array([fsm_temp[start_state][first_input]]).argmax()
        ans = np.array([np.array([fsm_temp[middle_state][second_input][final_state]
                                  for middle_state in [max_middle_state]]).sum()
                        for final_state in states])
        return np.array(ans / ans.sum(),
                        dtype=[(f"{start_state}—{first_input}–[ML path]–{second_input}–?", "float16")])

def control(start_state, final_state, fsm_temp, mode="normative"):
    """ Returns a dictionary with input combinations (e.g., "ab" as keys) and probabilities of the outcome as values. Higher probability = correct control action.
    mode = "normative" calculates probabilities of each possible combination of inputs for every possible middle state.
    mode = "an" does the same, but only for the most probable middle state(s)
    """
    input_combinations = [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]]
    states = [0, 1, 2, 3]
    control_result = {}
    for inp in input_combinations:
        label = inp[0] + inp[1]
        if mode == "normative":
            control_result[label] = np.array([fsm_temp[start_state][inp[0]][middle_state] *
                                              fsm_temp[middle_state][inp[1]][final_state] for middle_state in states]).sum()
        elif mode == "an":
            max_middle_state = np.array([fsm_temp[start_state][inp[0]]]).argmax()
            control_result[label] = np.array([fsm_temp[middle_state][inp[1]][final_state] for middle_state in [max_middle_state]]).sum()
    return control_result

def explain(start_state, first_input, second_input, final_state, fsm_temp, mode="normative"):
    ''' Returns a dictionary with probabilities of the outcome under counterfactual reasoning. LOWER probability = correct explanation.
    ["1"] — probability of the outcome with a counterfactual first input.
    ["2"] — same, but with counterfactual second input'''
    ans = {}
    states = [0, 1, 2, 3]

    # first input counterfactual
    first_input_cf = "b" if first_input == "a" else "a" #counteractual input 1

    # second input counterfactual
    second_input_cf = "b" if second_input == "a" else "a" #counteractual input 2

    if mode == "normative":
        explanation_1 = np.array([fsm_temp[start_state][first_input_cf][middle_state] * fsm_temp[middle_state][second_input][final_state]
                                  for middle_state in states]).sum()
        explanation_2 = np.array([fsm_temp[start_state][first_input][middle_state] * fsm_temp[middle_state][second_input_cf][final_state]
                                  for middle_state in states]).sum()
    elif mode == "an":
        max_middle_state1 = np.array([fsm_temp[start_state][first_input_cf]]).argmax()
        max_middle_state2 = np.array([fsm_temp[start_state][first_input]]).argmax()
        explanation_1 = np.array([fsm_temp[middle_state][second_input][final_state] for middle_state in [max_middle_state1]]).sum()
        explanation_2 = np.array([fsm_temp[middle_state][second_input_cf][final_state] for middle_state in [max_middle_state2]]).sum()
    else:
        explanation_1 = None
        explanation_2 = None
    ans["1"] = explanation_1
    ans["2"] = explanation_2

    return ans

def compute_for_trials(trials, task, form, epsilon, beta, mode="normative"):
    import math
    log_p_answers = []
    for index in range(0, trials.shape[0]):
        if mode == "normative":
            option_1_p_mm = trials.iloc[[index]].option_1_p_mm.item()
            option_2_p_mm = trials.iloc[[index]].option_2_p_mm.item()
        elif mode == "an":
            option_1_p_mm = trials.iloc[[index]].option_1_p_mm_an.item()
            option_2_p_mm = trials.iloc[[index]].option_2_p_mm_an.item()
        response = trials.iloc[[index]].response.item()

        if task == "prediction":
            option_1 = int(trials.iloc[[index]].option_1.item())
            option_2 = int(trials.iloc[[index]].option_2.item())
        if task == "explanation":
            option_1 = 1
            option_2 = 2
        if task == "control":
            option_1 = trials.iloc[[index]].option_1.item() if form == "hidden" else "a"
            option_2 = trials.iloc[[index]].option_2.item() if form == "hidden" else "b"

        # R_i
        if option_1_p_mm > option_2_p_mm:
            R_i = math.log10((option_1_p_mm + epsilon) / (option_2_p_mm + epsilon))
        elif option_1_p_mm < option_2_p_mm:
            R_i = math.log10((option_2_p_mm + epsilon) / (option_1_p_mm + epsilon))
        else:
            R_i = 1

        p_correct = (math.e ** (R_i * beta)) / (math.e ** (R_i * beta) + math.e ** (-1 * R_i * beta))

        if task == "explanation":
            correct_mm = option_1 if option_1_p_mm < option_2_p_mm else option_2
        else:
            correct_mm = option_1 if option_1_p_mm > option_2_p_mm else option_2

        p_answer = p_correct if response == correct_mm else (1 - p_correct)
        log_p_answer = math.log10(p_answer)
        log_p_answers.append(log_p_answer)
    return log_p_answers


# %%
sample = {}  # created empty dict for all data
eps = 0.1
betas = np.arange(0, 1, 0.01)

for participant_id in trials_learning.participant_id.unique():
    # Import data for one subject
    ## get data for a single participant
    mm = transitions[transitions['participant_id'] == participant_id]  # get metal model as a DataFrame
    fsm = participants.study_fsm[participants.participant_id == participant_id].item()
    learning = trials_learning[trials_learning.participant_id == participant_id]

    # Get Visible and Hidden trials for P, E, and C

    if trials_prediction[trials_prediction.participant_id == participant_id].size > 0:
        trials_vis = trials_prediction[(trials_prediction.participant_id == participant_id) &
                                       (trials_prediction.trial_type == "visible")]
        trials_hid = trials_prediction[(trials_prediction.participant_id == participant_id) &
                                       (trials_prediction.trial_type == "hidden")]
        task = "prediction"

    if trials_control[trials_control.participant_id == participant_id].size > 0:
        trials_vis = trials_control[(trials_control.participant_id == participant_id) &
                                    (trials_control.trial_type == "visible")]
        trials_hid = trials_control[(trials_control.participant_id == participant_id) &
                                    (trials_control.trial_type == "hidden")]
        task = "control"

    if trials_explanation[trials_explanation.participant_id == participant_id].size > 0:
        trials_vis = trials_explanation[(trials_explanation.participant_id == participant_id) &
                                        (trials_explanation.trial_type == "visible")]
        trials_hid = trials_explanation[(trials_explanation.participant_id == participant_id) &
                                        (trials_explanation.trial_type == "hidden")]
        task = "explanation"

    ## select only informative columns
    trials_vis = trials_vis.iloc[:,4:]
    trials_hid = trials_hid.iloc[:,4:]

    # reformat mental model as dictionary
    MM = {}
    for state_current in mm.state_current.unique():
        inputs = {}
        for inp in ["a", "b"]:
            inputs[inp] = np.array([
                mm.state_next_p[
                    (mm.state_current == state_current) & (mm.response_current == inp) & (mm.state_next == i)].item()
                for i in range(4)
            ])
        MM[state_current] = {"a": inputs["a"], "b": inputs["b"]}

    # Use MM to calculate correct answers
    if task == "prediction":
        # Visible
        for index in range(0, trials_vis.shape[0]):  # shape[0] = number of rows
            state_2 = int(trials_vis.iloc[[index]].state_2.item())
            response_2 = trials_vis.iloc[[index]].response_2.item()
            option_1 = int(trials_vis.iloc[[index]].option_1.item())
            option_2 = int(trials_vis.iloc[[index]].option_2.item())
            trials_vis.loc[trials_vis.index[index], 'option_1_p_mm'] = MM[state_2][response_2][option_1]
            trials_vis.loc[trials_vis.index[index], 'option_2_p_mm'] = MM[state_2][response_2][option_2]

        # Hidden
        for index in range(0, trials_hid.shape[0]):
            state_1 = int(trials_hid.iloc[[index]].state_1.item())
            response_1 = trials_hid.iloc[[index]].response_1.item()
            response_2 = trials_hid.iloc[[index]].response_2.item()
            option_1 = int(trials_hid.iloc[[index]].option_1.item())
            option_2 = int(trials_hid.iloc[[index]].option_2.item())

            trials_hid.loc[trials_hid.index[index], 'option_1_p_mm'] = \
                predict(state_1, response_1, response_2, MM)[option_1][0]
            trials_hid.loc[trials_hid.index[index], 'option_2_p_mm'] = \
                predict(state_1, response_1, response_2, MM)[option_2][0]

            trials_hid.loc[trials_hid.index[index], 'option_1_p_mm_an'] = \
                predict(state_1, response_1, response_2, MM, mode="an")[option_1][0]
            trials_hid.loc[trials_hid.index[index], 'option_2_p_mm_an'] = \
                predict(state_1, response_1, response_2, MM, mode="an")[option_2][0]

    if task == "control":
        # Visible
        for index in range(0, trials_vis.shape[0]):  # shape[0] = number of rows
            state_2 = int(trials_vis.iloc[[index]].state_2.item())
            state_3 = int(trials_vis.iloc[[index]].state_3.item())
            option_1 = "a"
            option_2 = "b"
            trials_vis.loc[trials_vis.index[index], 'option_1_p_mm'] = MM[state_2][option_1][state_3]
            trials_vis.loc[trials_vis.index[index], 'option_2_p_mm'] = MM[state_2][option_2][state_3]

        # Hidden
        for index in range(0, trials_hid.shape[0]):
            state_1 = int(trials_hid.iloc[[index]].state_1.item())
            state_3 = int(trials_hid.iloc[[index]].state_3.item())
            option_1 = trials_hid.iloc[[index]].option_1.item()
            option_2 = trials_hid.iloc[[index]].option_2.item()

            trials_hid.loc[trials_hid.index[index], 'option_1_p_mm'] = control(state_1, state_3, MM, mode="normative")[
                option_1]
            trials_hid.loc[trials_hid.index[index], 'option_2_p_mm'] = control(state_1, state_3, MM, mode="normative")[
                option_2]

            trials_hid.loc[trials_hid.index[index], 'option_1_p_mm_an'] = control(state_1, state_3, MM, mode="an")[
                option_1]
            trials_hid.loc[trials_hid.index[index], 'option_2_p_mm_an'] = control(state_1, state_3, MM, mode="an")[
                option_2]

    if task == "explanation":
        # Visible
        for index in range(0, trials_vis.shape[0]):  # shape[0] = number of rows
            state_1 = int(trials_vis.iloc[[index]].state_1.item())
            response_1 = trials_vis.iloc[[index]].response_1.item()
            state_2 = int(trials_vis.iloc[[index]].state_2.item())
            response_2 = trials_vis.iloc[[index]].response_2.item()
            state_3 = int(trials_vis.iloc[[index]].state_3.item())
            response_1_cf = "a" if response_1 == "b" else "b"
            response_2_cf = "a" if response_2 == "b" else "b"
            trials_vis.loc[trials_vis.index[index], 'option_1_p_mm'] = MM[state_1][response_1_cf][state_2] * \
                                                                       MM[state_2][response_2][state_3]
            trials_vis.loc[trials_vis.index[index], 'option_2_p_mm'] = MM[state_1][response_1][state_2] * \
                                                                       MM[state_2][response_2_cf][state_3]

        # Hidden
        for index in range(0, trials_hid.shape[0]):  # shape[0] = number of rows
            state_1 = int(trials_hid.iloc[[index]].state_1.item())
            response_1 = trials_hid.iloc[[index]].response_1.item()
            response_2 = trials_hid.iloc[[index]].response_2.item()
            state_3 = int(trials_hid.iloc[[index]].state_3.item())
            response_1_cf = "a" if response_1 == "b" else "b"
            response_2_cf = "a" if response_2 == "b" else "b"
            trials_hid.loc[trials_hid.index[index], 'option_1_p_mm'] = \
                explain(state_1, response_1, response_2, state_3, MM)["1"]
            trials_hid.loc[trials_hid.index[index], 'option_2_p_mm'] = \
                explain(state_1, response_1, response_2, state_3, MM)["2"]

            trials_hid.loc[trials_hid.index[index], 'option_1_p_mm_an'] = \
                explain(state_1, response_1, response_2, state_3, MM, mode="an")["1"]
            trials_hid.loc[trials_hid.index[index], 'option_2_p_mm_an'] = \
                explain(state_1, response_1, response_2, state_3, MM, mode="an")["2"]

    # Add beta values
    beta_person = {}
    for form in ["visible", "hidden"]:
        log_p_answers = []
        if form == "visible":
            trials = trials_vis
        else: trials = trials_hid
        all_betas = []
        all_betas_an = []
        all_betas_fsm = []
        all_betas_fsm_an = []
        for beta in betas:
            log_p_answers = compute_for_trials(trials, task, form, eps, beta)
            all_betas.append([round(beta, 2), sum(log_p_answers)])
            # the same for current FSM, not MM
            log_p_answers_fsm = compute_for_trials(trials, task, form, eps, beta)
            all_betas_fsm.append([round(beta, 2), sum(log_p_answers_fsm)])
            if form == "hidden":
                log_p_answers_an = compute_for_trials(trials, task, form, eps, beta, mode="an")
                all_betas_an.append([round(beta, 2), sum(log_p_answers_an)])

                # the same for current FSM, not MM
                log_p_answers_fsm_an = compute_for_trials(trials, task, form, eps, beta, mode="an")
                all_betas_fsm_an.append([round(beta, 2), sum(log_p_answers_fsm_an)])
        beta_person[form + "_mm"] = all_betas
        beta_person[form + "_fsm"] = all_betas_fsm
    beta_person["hidden_an_mm"] = all_betas_an
    beta_person["hidden_an_fsm"] = all_betas_fsm_an

    max_betas = {}

    for condition in beta_person.keys():
        betas_task = np.array(beta_person[condition])[:,0]
        values = np.array(beta_person[condition])[:,1]
        max_beta = betas_task[values.argmax()]
        R_i = 3
        p_correct_r3 = (math.e ** (R_i * max_beta)) / (math.e ** (R_i * max_beta) + math.e ** (-1 * R_i * max_beta))
        max_betas[condition] = np.array([max_beta, p_correct_r3])

    ## record that data in a dictionary
    sample[participant_id] = {"task": task,
                              "fsm": fsm,
                              "learning": learning,
                              "MM": MM,
                              "visible": trials_vis,
                              "hidden": trials_hid,
                              "visible_beta": beta_person["visible_mm"],
                              "hidden_beta": beta_person["hidden_mm"],
                              "visible_beta_fsm": beta_person["visible_fsm"],
                              "hidden_beta_fsm": beta_person["hidden_fsm"],
                              "hidden_beta_fsm_an": beta_person["hidden_an_fsm"],
                              "hidden_beta_an": beta_person["hidden_an_mm"],
                              "max_betas": max_betas
                              } # make it list by adding ".values.tolist()"
#%%
# export Expected Accuracy results for participants
results = pd.DataFrame(columns=['participant_id','fsm','task','condition','beta_max', 'p_correct_r3'])
this_row = {}
for participant_id in sample.keys():
    results_participant = pd.DataFrame(columns=['participant_id','fsm','task','condition','beta_max', 'p_correct_r3'])
    fsm = sample[participant_id]["fsm"]
    task = sample[participant_id]["task"]
    for condition in sample[participant_id]["max_betas"].keys():
        betas = sample[participant_id]["max_betas"][condition]
        this_row = {"participant_id": participant_id,
                    "fsm": fsm,
                    "task":task,
                    'condition':condition,
                    "beta_max": betas[0],
                    "p_correct_r3": betas[1]}
        results_participant = results_participant.append(this_row, ignore_index = True)
    results = results.append(results_participant, ignore_index=True)

filepath = "../data_preprocessed/betas_predicted_accuracy.csv"
results.to_csv(filepath, index = False)