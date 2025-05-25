#!/bin/bash

# Directory of Script
cd ~

# Run R script
Rscript /home/brian/research/website_download_automation/Ameriflux.R


# Remember to make it executable
# chmond +x Ameriflux.sh


# To have it weekly run the script, go into terminal and put:

# crontab - e

# add to your crontab

# 0 3 * * 0 /home/brian/research/website_download_automation/Ameriflux.sh >> /home/brian/research/data_files/Downloaded Data 2>&1
