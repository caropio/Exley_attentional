
Python code to analyse our behavioral data 

Each ```H*_data_analysis.py scripts``` script in the main folder begins by calling ```data_processing.py```.

Before running the ```H*_data_analysis.py``` scripts, ensure that the ```/data``` folder contains the outputs ```criterion info data.csv```, 
```dataset.csv``` and ```survey data.csv``` from running the ```data_brushing.py``` script.  

Here is a description of the different variables of the ```dataset.csv``` file which are of interest in the analysis

| Variable                  | Description                                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| id                        | id associated to each participant (letters and number                                                           |
| number                    | number id associated to each participant (one number)                                                           |
| session                   | code of session (deployment of experiment)                                                                      |
| charity_name              | name of charity chosen by participant in part 1                                                                 |
| charity_calibration       | participant specific X from part 2                                                                              |
| buffer_X                  | participant specific X from the buffer part of part 2                                                           |
| censored_calibration      | 1 if participant is censored or 0                                                                               |
| case                      | the case (YSPS/YCPC/YSPC/YCPS) of the presented price list of part 3                                            |
| round_order               | order of presentation of price list within the case (1 to 7)                                                    |
| option_A_vector           | non-zero amount of money involved in lottery (Option A) which is either 10 or X value                           |
| prob_option_A             | number of green and purple balls of associated probability urn                                                  |
| option_B_vector           | array of the different values of option B across the rows                                                       |
| choices                   | 21 choices (As and Bs) of valuation price list made across the rows                                             |
| switchpoint               | number of index (starting at 1) after switching between options                                                 |
| valuation                 | valuation of the price list (in %)                                                                              |
| nb_switchpoint            | number of switchpoints in the price list                                                                        |
| total_time_spent_s        | time took to complete the entire price list (from the page upload until they click on next button) in seconds   |
| watching_urn_ms           | array of different times spent revealing probability urn (array of times in ms)                                 |
| watching_urn_ms_corrected | above array without values below or equal to 200 ms (array of times in ms)                                      |
| frequency                 | number of times the probability urn was revealed                                                                |
| temporal_information      | temporal information (when urn is unmasked and choices are made) - exploratory                                  |
| order of cases            | ordered array of presentation of cases (randomized)                                                             |
| dwell_time_relative       | attention towards urn normalized by total time spent on price list (in %)                                       |
| dwell_time_absolute       | attention towards urn (in s)                                                                                    |
| charity                   | indicator for whether the lottery is for the charity (1) or not (0)                                             |
| tradeoff                  | indicator for whether we are in tradeoff context (1) or not (0)                                                 |
| interaction               | interaction term of charity and tradeoff dummies                                                                |


