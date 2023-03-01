library(tidyverse)


# load data 
ts_raw <- read.csv("data_raw/2023-02-13trials_anonymized_an-exp2_n-24.csv")

# Learning phase ####
## Create a df with learning trials (but without responses)
ts_learning_1 <- ts_raw %>% select(participant_id, fsm_number, test_condition, learning_counter, sender, 
                                   state_current, response_current, state_next, state_1, response_1, state_2, response_2, state_3) %>% 
  filter(state_current >=0 & !(sender %in% c("control_screen_task"))) %>% 
  mutate(state_1 = ifelse(learning_counter == 1, NA, #add 1st and 2nd trial data
                                                             ifelse(learning_counter == 2, lag(state_current), state_1)), 
                                       response_1 = ifelse(learning_counter == 1, NA, #add 1st and 2nd trial data
                                                        ifelse(learning_counter == 2, lag(response_current), response_1)), 
                                       state_2 = ifelse(learning_counter == 1, state_current, #add 1st and 2nd trial data
                                                        ifelse(learning_counter == 2, state_current, state_2)),
                                       response_2 = ifelse(learning_counter == 1, response_current, #add 1st and 2nd trial data
                                                           ifelse(learning_counter == 2, response_current, response_2)), 
                                       state_3 = ifelse(learning_counter == 1, state_next, #add 1st and 2nd trial data
                                                        ifelse(learning_counter == 2, state_next, state_3)),
  ) 


## Get response times for interactions with the chatbot 
ts_learning_2 <- ts_raw %>% select(participant_id, learning_counter, sender, response, duration) %>% 
  filter(!is.na(response) & learning_counter > 0 & sender %in% c("response_1_screen","response_2_screen","response_1_screen_time", "response_2_screen_time","response_further_screen", "response_further_screen_time")) %>% 
  mutate(response_time = ifelse(sender %in% c("response_further_screen_time", "response_1_screen_time", "response_2_screen_time"), 
                                duration + 15000, 
                                duration)) %>% select(-duration, -sender, -response)  # account for extra time due to timeout

ts_learning <-  ts_learning_1 %>% left_join(ts_learning_2) %>% 
  mutate(fsm_type = ifelse(fsm_number == 21, "easy", "hard"), 
         learning_counter = learning_counter + 1, 
         response_time_lessThan300ms = ifelse(response_time <= 300, 1, 0),
         trial_id = paste0(state_current, response_current, state_next)) %>% select(-fsm_number) %>% relocate(fsm_type,  .after = 1)


n <- nrow(ts_learning %>% group_by(participant_id) %>% count())
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_learning", "_an-exp2_n-",n, ".csv")
write.csv(ts_learning, filename, row.names = F)


# Transitions (Mental Models) ####

# calculate frequencies and proportions of observed next states
tmp_transitions <- ts_learning %>% 
  group_by(participant_id, 
           state_current, 
           response_current, 
           state_next) %>% summarise(state_next_n = n()) %>% 
  group_by(participant_id, state_current, response_current) %>% 
  mutate(state_next_p = state_next_n / sum(state_next_n))


tmp_expanded <- ts_learning %>%  expand(participant_id, state_current, response_current, state_next)

transitions <- tmp_expanded %>% left_join(tmp_transitions) %>% 
  replace(is.na(.), 0)  %>%  ungroup() # replace all NA with 0 

n <- nrow(tmp_transitions %>% group_by(participant_id) %>% count())
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_transitions", "_an-exp2_n-",n, ".csv")
write.csv(transitions, filename, row.names = F)


# Test Preview data ####
### CONTROL PREVIEW
ts_preview_c_1 <- ts_raw %>% select(participant_id, learning_counter, sender, state_1, response_1, state_2, response_2, state_3, state_current, response_current, state_next) %>% 
  filter(sender %in% c("control_screen_task") & !is.na(state_current)) %>% select(-sender)

ts_preview_c_2 <- ts_raw %>% select(participant_id, learning_counter, sender, response, correctResponse, correct, duration) %>% 
  filter(sender %in% c("control_screen_response") & !is.na(response) & !is.na(learning_counter)) %>% select(-sender)

ts_preview_c <- ts_preview_c_1 %>% left_join(ts_preview_c_2) %>% mutate(correct = as.integer(correct))

n <- nrow(ts_preview_c %>% group_by(participant_id) %>% count())
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_preview_control", "_an-exp2_n-",n, ".csv")
write.csv(transitions, filename, row.names = F)

### PREDICTION PREVIEW
ts_preview_p_2 <- ts_raw %>% select(participant_id, learning_counter, option_1, option_2, response, correctResponse, duration, correct, prediction_correct) %>% 
  filter(!is.na(prediction_correct) & !is.na(learning_counter)) %>% mutate(correct = as.integer(correct))

tmp <- ts_learning %>% select(participant_id, learning_counter, state_1, response_1, response_2, state_3, state_current, response_current, state_next)

ts_preview_p <- ts_preview_p_2 %>% inner_join(tmp)


ts_preview_p <- ts_preview_c_1 %>% left_join(ts_preview_c_2)  

### ADD PREVIEW EXPLANATION EXPORT HERE


# Prediction Test Trials ####
ts_prediction_rt <- ts_raw %>% filter((test_prediction_counter >=0) & 
                                        !(is.na(response))) %>% select(participant_id, trial_number = test_prediction_counter, sender, duration) %>% 
  mutate(trial_number = trial_number + 1, 
         response_time = ifelse(sender == "prediction_screen_response", 
                                duration, duration + 15000)) %>% select(-duration, -sender)

ts_prediction <- ts_raw %>% filter((test_prediction_counter >=0)& 
                                     !(is.na(response))) %>% 
  select(participant_id, fsm_number, test_condition, trial_number = test_prediction_counter, trial_type = exp_type, 
         state_1, response_1, state_2, response_2, option_1, option_2,
         correctResponse, response, response_correct = correct,
         option_1_p, option_2_p) %>% mutate(fsm_type = ifelse(fsm_number == 21, "easy", "hard"), 
                                            trial_number = trial_number + 1, 
                                            trial_id = ifelse(trial_type == "visible", 
                                                              paste0(fsm_type,"_p_vis_", state_1, response_1, state_2, response_2, option_1, option_2), 
                                                              paste0(fsm_type,"_p_hid_", state_1, response_1, response_2, option_1, option_2))) %>% 
  select(participant_id,fsm_type,test_condition, trial_number,trial_type, trial_id, state_1:option_2_p)

ts_prediction <- ts_prediction %>% left_join(ts_prediction_rt)

n <- nrow(ts_prediction %>% group_by(participant_id) %>% count())
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_test_prediction_", "_an-exp2_n-",n, ".csv")
write.csv(ts_prediction, filename, row.names = F)


# Control Test Trials ####
ts_control_rt <- ts_raw %>% filter(test_control_counter >=0 & !(is.na(response))) %>% 
  select(participant_id, trial_number = test_control_counter, sender, duration) %>% 
  mutate(trial_number = trial_number + 1, 
         response_time = ifelse(sender == "control_screen_response", 
                                duration, duration + 15000)) %>% select(-duration, -sender)

ts_control <- ts_raw %>% filter(test_control_counter >=0 & 
                                 !(is.na(response))) %>% 
  select(participant_id, fsm_number, test_condition, trial_number = test_control_counter, trial_type = exp_type, 
         state_1, response_1, state_2, state_3, option_1, option_2,
         correctResponse, response, duration, response_correct = correct,
         option_1_p, option_2_p) %>% mutate(fsm_type = ifelse(fsm_number == 21, "easy", "hard"), 
                                            response_1 = ifelse(response_1 == "", "NA", response_1),
                                            trial_number = trial_number + 1,
                                            trial_id = ifelse(trial_type == "visible", 
                                                              paste0(fsm_type,"_c_vis_", state_1, response_1, state_3), 
                                                              paste0(fsm_type,"_c_hid_", state_1, state_3, "-", option_1,"-",option_2))) %>% 
  select(participant_id,fsm_type,test_condition, trial_number,trial_type, trial_id, state_1:option_2_p)

ts_control <- ts_control %>% left_join(ts_control_rt)

n <- nrow(ts_control %>% group_by(participant_id) %>% count())
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_test_control_", "_an-exp2_n-",n, ".csv")
write.csv(ts_control, filename, row.names = F)

# Explanation Test Trials ####
ts_explanation_rt <- ts_raw %>% filter(test_explanation_counter >= 0 & !(is.na(response))) %>% 
  select(participant_id, trial_number = test_explanation_counter, sender, duration) %>% 
  mutate(trial_number = trial_number + 1, 
         response_time = ifelse(sender == "explanation_screen_response", 
                                duration, duration + 15000)) %>% select(-duration, -sender)

ts_explanation <- ts_raw %>% filter(test_explanation_counter >= 0 & !(is.na(response))) %>% 
  select(participant_id, fsm_number, test_condition, trial_number = test_explanation_counter, trial_type = exp_type, 
         state_1, response_1, state_2, response_2, state_3, explanation_1_decrease, explanation_2_decrease,
         correctResponse, response, response_correct = correct, duration, option_1_p, option_2_p) %>% 
  mutate(fsm_type = ifelse(fsm_number == 21, "easy", "hard"),
         trial_number = trial_number + 1, 
         trial_id = ifelse(trial_type == "visible", 
                           paste0(fsm_type,"_e_vis_", state_1, response_1, state_2, response_2, state_3), 
                           paste0(fsm_type,"_e_hid_", state_1, response_1, response_2, state_3))) %>% 
  select(participant_id,fsm_type,test_condition, trial_number,trial_type, trial_id, state_1:response_correct)

ts_explanation <- ts_explanation %>% left_join(ts_explanation_rt)

n <- nrow(ts_explanation %>% group_by(participant_id) %>% count())
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_test_explanation_", "_an-exp2_n-",n, ".csv")

write.csv(ts_explanation, filename, row.names = F)


# Combine all trials into one dataframe ####
ts <- ts_learning %>% bind_rows(ts_prediction, ts_control, ts_explanation) %>% 
  mutate(trial_number = trial_number + 45, 
         response_time_lessThan300ms = ifelse(response_time <= 300, 1, 0)) %>% 
  unite(trial_order, c(learning_counter, trial_number), sep = "", remove = FALSE, na.rm = T) %>% 
  mutate(trial_order = as.integer(trial_order), 
         phase = ifelse(is.na(learning_counter), "test", "learning"), 
         response_time_lessThan1000ms = ifelse(response_time < 1000, 1, 0), 
         response_time_tooQuick = ifelse(phase == "learning" & response_time_lessThan300ms, 1, 
                                         ifelse(phase == "test" & response_time_lessThan1000ms, 1, 0))) %>% 
  arrange(participant_id, trial_order) %>% 
  select(-learning_counter, -trial_number) %>% relocate(phase, .after = 3)

n <- nrow(ts %>% group_by(participant_id) %>% count())
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_trials", "_an-exp2_n-",n, ".csv")
write.csv(ts, filename, row.names = F)


# Participants ####
tmp_ss <- ts_raw %>% 
  select(participant_id, 
         study_task = test_condition, 
         study_fsm = fsm_number, 
         q_estimate, q_strategy = q_stategy_scale, 
         txt_advice = q_strategy,
         meta_timestamp = timestamp)

ss <- tmp_ss %>% group_by(participant_id) %>% 
  summarise(study_task = first(na.omit(study_task)), 
            study_fsm = first(na.omit(study_fsm)),
            meta_timestamp = first(na.omit(meta_timestamp)),
            q_strategy = first(na.omit(q_strategy)),
            q_estimate = first(na.omit(q_estimate)),
            meta_timestamp = first(na.omit(meta_timestamp))) %>% 
  left_join(tmp_ss %>% select(participant_id, txt_advice) %>% filter(!txt_advice == "")) %>% mutate(study_fsm = ifelse(study_fsm == 21, "easy", "hard"))

# Add Prolific demographic data
ss_demogr <- read.csv("data_raw/2023-02-13_demographics_anonymized_an-exp2_n-24.csv") 
ss_demogr <- ss_demogr %>% select(participant_id, 
                                  subject_age = Age, subject_sex = Sex, 
                                  subject_ethnicity = Ethnicity.simplified, 
                                  subject_timeTaken = Time.taken)

ss <- ss %>% left_join(ss_demogr)
n <- nrow(ss)
filename <- paste0("data_preprocessed/", format(Sys.Date(),"%Y-%m-%d"), "_participants_", "_an-exp2_n-",n, ".csv")
write.csv(ss, filename, row.names = F)
