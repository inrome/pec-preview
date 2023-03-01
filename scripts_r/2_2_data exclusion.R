library(tidyverse)

ts <- read.csv("data_preprocessed/2023-02-17_trials_an-exp2_n-24.csv")

times <- ts %>% group_by(participant_id, phase) %>% summarise(sum = sum(response_time_tooQuick, na.rm = T))

to_exclude <- ts %>% group_by(participant_id) %>% summarise(sum = sum(response_time_tooQuick, na.rm = T)) %>% filter(sum > 5) %>% pull(participant_id) 

