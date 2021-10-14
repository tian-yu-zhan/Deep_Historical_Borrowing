# Deep_Historical_Borrowing
R code to replicate simulation studies in the article "Deep historical borrowing framework to prospectively and simultaneously synthesize control information in confirmatory clinical trials with multiple endpoints"

https://www.tandfonline.com/doi/full/10.1080/10543406.2021.1975128?src=

This is a help file for the R code accompanying the article “Deep Historical Borrowing Framework to Prospectively and Simultaneously Synthesize Control Information in Confirmatory Clinical Trials with Multiple Endpoints”.

Supplemental Materials include additional simulation results of Section 4 with an empirical correlation at 0.5.

The R code of reproducing simulation results is saved at the folder “sim_main_article” for Section 4, “case” for Section 5, “sim_supp” for the Supplemental Materials.

Training: Within a specific example folder, source “XX_train.r” to train DNNs from Algorithm 1 and 2 based on simulated current trial data, and a file for scaling parameters. Note that those files are already saved in each example folder.

Testing: One can directly source “XX_test.r” to generate observed current trial data to evaluate operating characteristics based on the pre-trained models.
