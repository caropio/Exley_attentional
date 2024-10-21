
Python code to analyse our behavioral data 

Each ```H*_data_analysis.py scripts``` script in the main folder begins by calling ```data_processing.py```.

Before running the ```H*_data_analysis.py scripts```, ensure that the ```/data``` folder contains the outputs ```criterion info data.csv```, 
```dataset.csv``` and ```survey data.csv``` from running the ```data_brushing.py``` script. Additionally, 

| Variable             | Description                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
| id                   | number id associated to each participant                                                                        |
| session              | code of session (deployment of experiment)                                                                      |
| charity_name         | name of charity chosen by participant in part 1                                                                 |
| charity_calibration  | participant specific X from part 2                                                                              |
| case                 | the case (YSPS/YCPC/YSPC/YCPS) of the presented price list of part 3                                            |
| round_order          | order of presentation of price list within the case (1 to 7)                                                    |
| option_A_vector      | non-zero amount of money involved in lottery (Option A) which is either 10 or X value                           |
| prob_option_A        | number of green and purple balls of associated probability urn                                                  |
| option_B_vector      | array of the different values of option B across the rows                                                       |
| choices              | 21 choices (As and Bs) of valuation price list made across the rows                                             |
| total_time_spent_s   | time took to complete the entire price list (from the page upload until they click on next button) in seconds   |
| watching_urn_ms      | array of different times spent revealing probability urn (array of times in ms)                                 |
| temporal_information | temporal information (when urn is unmasked and choices are made) - exploratory                                  |

