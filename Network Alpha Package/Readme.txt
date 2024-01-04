version https://git-lfs.github.com/spec/v1
oid sha256:b2e270c7c5a3f3c0ab48ed6904ce6a10033731eaac0e5e6a84b674f25953970f
size 4874

Network Alpha - Readme Guide


1. Description

The package 'Network Alpha Package' consists of the following folders:

- data: 
    data folder includes the raw data files downloaded from Bloomberg terminal. These are named as cse6242_us_stock_<year>db.csv
    Jupyter notebook Clean_US_Stock_DB.ipynb for data wrangling and cleaning up the raw data files by NaN handling, outlier management, impute missing data.
    Cleaned csv data files names as <year>_clean.csv. These files were used for analysis.

-experiments
   experiments folder consists of the results of the hyperparameter tuning and algorithm comparison experiments
   Algo Compare.csv - results of KD tree and brute force algorithm
   Brute_force_NN_tuning - results of hyperparameter tuning of Brute force Nearest Neighbor Algorithm
   Kdtree_NN_tuning - results of hyperparameter tuning of Kdtree Nearest Neighbor Algorithm

- lib
   lib folder includes the following d3.js source files for import in the html code.
   
   d3.v5.min
   d3-dsv.min
   d3-legend.min
   d3-tip.min

- nearest_neighbor
   nearest_neighbor folder includes the csv files that are imported into the visualization.
   
   <month>_<year>.csv - The files for all 12 months and years 2016-2020 include the source and target values of the nodes that are used in the network graph visualization.
   <month>_<year>_info.csv - These files have the corresponding metrics for the different tickers and also connect the ticker symbols to the node ids. This is also used in the visualization.
   Jupyter Notebook Nearest Neighbor.ipynb is used to generate the source, target, and other metrics from the clean data files in the data folder.

-visualization
    visualization folder contains the front-end code and files

    top.jpg - Banner at the top of the visualization
    comp_desc.csv - Csv file consisting of stock tickers used in the drop down menu in the visualization
    index.html - html file for front-end layout
    main.js  - Javascript file with the d3.js code
    style.css - css style file


2. Installation

The application uses Python 3.8.8 for back end calculations. In this demo application, the local files have been pre-generated in Python and available for loading in the front-end display.
The Javascript source files are available in the folder. To run the application, download and unzip the Network Alpha Package and verify that the above folders are present in the package.

 - Create a simple http server at the Network Alpha Package level using Python. Navigate to the folder in Anaconda prompt and type in python -m http.server 8000. This step is crucial to run    the application.   Other forms of http server generation has not been tested and may not work. Refer to https://ryanblunden.com/create-a-http-server-with-one-command-thanks-to-python-   29fcfdcd240e for additional details.
 - Open a browser and navigate to http://localhost:8000/
 - Navigate to visualization folder and the dashboard should open in the browser window

3. Execution

Opening the visualization, would show a default network graph of all stocks on January, 2016 with 5 neighbors for each stocks (node). The supplementary bar graph will not be shown in the default visualization. The nodes are colored based on industry sector and sized based on market cap. Larger node signifies larger market cap. The demo data includes all 12 months from 2016-2019. The year 2020 was not included to enable performance calculations. 

- The user has the option to select month and year from drop down menu. The user also has the added option to select the stock ticker. The changes will ONLY take in effect, when the user   presses the RUN button.
- On clicking the RUN button, the visualization updates to display the selected stock in blue, the five nearest neighbors in black, and a filtered view showing only the sectors that   includes the selected stock and its neighbors. The number of stocks displayed in the 'Filtered' view is less to enable better exploration of the network.
- On clicking RUN, the visualization also includes annual performance comparison between the selected stock and the top 5 recommendations. This is shown in a bar graph with green showing positive returns and red showing negative returns.
- The user also has the option to select an 'All' view which includes all stocks and sectors. The network graph is more dense but provides user an option to explore all the stocks in the   market.
- Furthermore, the user can drag and pin the nodes to improve readability of the network graph. The pinned nodes will change color to green. Double clicking pinned nodes, will return them to the original state.

- The visualization is best displayed in Google Chrome and in a 21 inch monitor or above. A smaller monitor might compress the display.
