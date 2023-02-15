library(tidyverse)
library(openxlsx)

ts_raw <- read.xlsx("../data_withIDs/2023-02-13 openlab data.xlsx")
tmp_prolific <- read.csv("../data_withIDs/prolific_export_63e69d806856401557af9a2f.csv")
# ts_raw %>% distinct(openLabId) %>% count()
# ts_raw %>% distinct(code) %>% count()

# select only participants from the new experiment
ts_raw <- ts_raw %>% filter(code %in% unique(tmp_prolific$Participant.id))

# check that all IDs match
setequal(tmp_prolific %>% distinct(code = Participant.id), ts_raw %>% distinct(code)) 

# generate unique IDs
ids <- ts_raw %>% distinct(code, openLabId) %>% mutate(participant_id = 1:nrow(.))
n <- nrow(ids)
filename <- paste0("../data_withIDs/", format(Sys.Date(),"%Y-%m-%d"), "_removed-ids", "_an-exp2_n-",n, ".csv")
write.csv(ids, filename, row.names = F) # write them in a csv just in case

# substitude Prolific ID with new code in trials 
ts_anonym <- ts_raw %>% left_join(ids) %>% select(-code, -openLabId)
n <- ts_anonym %>% distinct(participant_id) %>% nrow
filename <- paste0("data_raw/", format(Sys.Date(),"%Y-%m-%d"), "trials_anonymized", "_an-exp2_n-",n, ".csv")
write.csv(ts_anonym, filename, row.names = F) # write them in a csv just in case

# Get Prolific meta data
tmp_prolific <- tmp_prolific %>% rename(code = Participant.id) %>% mutate(Age = as.integer(Age))
tmp_prolific_anonym <- tmp_prolific %>% left_join(ids) %>% select(-code)

n <- tmp_prolific_anonym %>% distinct(participant_id) %>% nrow
filename <- paste0("data_raw/", format(Sys.Date(),"%Y-%m-%d"),"_","demographics_anonymized", "_","an-exp2_n-",n, ".csv")
write.csv(tmp_prolific_anonym, filename, row.names = F) # write them in a csv just in case
