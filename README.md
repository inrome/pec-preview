# Prediction, Explanation, and Control Under Free Exploration Learning: Experiment 2
This is a follow-up experiment with modified instruction at the learning phase and a preview of test tasks during the learning. During the learning phase, participants were occasionally presented with test questions that asked them to predict, control, or explain the chatbot's behavior (depending on their experimental condition) with an immediate feedback on their accuracy ("Correct" or "That's incorrect"). There were 5 questions in total. 

# Requirements
- Data collection: [Lab.JS](https://lab.js.org/) + [Open Lab](https://open-lab.online/)
- Preprocessing: RStudio (R version 4.2.1)
- Main analysis: Python 3.7

# Components 
- `data_raw` — contains anonymized data downloaded "as is" from a crowdsourcing platform Prolific (demographics) and a hosting for online studies Open Lab. Anonymization scipt: `scripts_r/1_anonymization.R`
- `data_preprocessed` — files prepared for data analysis 

## Demo Experiment
You can try the [Chatbot Interaction Task v3](https://open-lab.online/test/chatbot-interaction-task-v3/63bf3f75f260f774c963e5a4) hosted at Open Lab

# Participants
24 U.S. participants. Data was collected online via Prolific in February 2023 . See `data_raw/2023-02-13_demographics_anonymized_an-exp2_n-24.csv`, for a more detailed demographics. 

## Related Projects
[Prediction, Explanation, and Control Under Free Exploration Learning: Experiment 1](https://github.com/inrome/cogsci-2023)

## Authors
- Roman Tikhonov ([Google Scholar](https://scholar.google.ru/citations?user=4ag4R48AAAAJ&hl=ru))
- Simon DeDeo ([Google Scholar](https://scholar.google.com/citations?user=UW3tRn8AAAAJ&hl=en))
