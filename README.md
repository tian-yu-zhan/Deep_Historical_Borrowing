# Deep_Historical_Borrowing
R code to replicate simulation studies in the article "Deep Historical Borrowing Framework to Prospectively and Simultaneously Synthesize Control Information in Confirmatory Clinical Trials with Multiple Endpoints"

This is a help file for the R code accompanying the article “Deep Historical Borrowing Framework to Prospectively and Simultaneously Synthesize Control Information in Confirmatory Clinical Trials with Multiple Endpoints”.

The R code of reproducing simulation results is saved at the folder “sim_main_article” for Section 4, “case” for Section 5, “sim_supp” for the Supplemental Materials.

Training: Within a specific example folder, source “XX_train.r” to train DNNs from Algorithm 1 and 2 based on simulated current trial data, and a file for scaling parameters. Note that those files are already saved in each example folder.

Testing: One can directly source “XX_test.r” to generate observed current trial data to evaluate operating characteristics based on the pre-trained models.
