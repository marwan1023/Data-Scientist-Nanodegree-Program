### Data-Scientist-Nanodegree-Program
By Marwan Saeed Alsharabbi
## COVID-19 Analysis ,visualization & Prediction

### COVID-19 Analysis ,visualization & Prediction 
### Table of Contents

- <a href='#1'>Introduction</a>  
- <a href='#2'>Load packages Prerequisites</a> 
- <a href='#3'>Data and variables</a>   
- <a href='#4'>Questions and Problems</a>  
- <a href='#5'>Perform data preparation and Cleaning</a> 
- <a href='#6'>Evaluate the Results</a>
- <a href='#7'>Modeling</a>      
- <a href='#8'>Write A COVID-19 Analysis Blog Post</a>



## <a id="1">Introduction</a> 

Coronavirus is a family of viruses that are named after their spiky crown. The novel coronavirus, also known as SARS-CoV-2, is a contagious respiratory virus that first reported in Wuhan, China. On 2/11/2020, the World Health Organization designated the name COVID-19 for the disease caused by the novel coronavirus. This notebook aims at exploring COVID-19 through data analysis and projections.
The world is going through a difficult time and fighting with a deadly virus called COVID-19. Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). It was first identified in December 2019 in Wuhan, China, and has resulted in an ongoing pandemic. The first case may be traced back to 17 November 2019.As of 8 June 2020, more than 7.06 million cases have been reported across 188 countries and territories, resulting in more than 403,000 deaths. More than 3.16 million people have recovered.

### Select a real-world dataset 

I chose the Covid 19 data set from the following site(https://ourworldindata.org/coronavirus), and I will analyze the data, clean and perform some interesting processes and conclusions. I will strengthen the analysis and cleaning of global data. 
The data was downloaded from https://covid.ourworldindata.org/data/owid-covid-data.csv.


### Data Sources:

Confirmed cases and deaths: Data comes from the European Centre for Disease Prevention and Control (ECDC)
Testing for COVID-19: Data is collected by the Our World in Data team from official reports; you can find the source information for every country and further details in the post on COVID-19 testing. The testing dataset is updated around twice a week.
Confirmed cases and deaths: Data is collected from a variety of sources (United Nations, World Bank, Global Burden of Disease, etc.)

### License:
The information on this page is summarized from OWID's COVID-19 github page. All of Our World in Data is completely open access and all work is licensed under the Creative Commons BY license. More information about the usage of content can be found OWID github page.https://github.com/owid/covid-19-data/tree/master/public/data

### Authors:
OWID's COVID19 github page the data has been collected, aggregated, and documented by Diana Beltekian, Daniel Gavrilov, Joe Hasell, Bobbie Macdonald, Edouard Mathieu, Esteban Ortiz-Ospina, Hannah Ritchie, Max Roser.


## <a id="2">Load packages Prerequisites</a>
* Numpy
* Pandas
* seaborn
* Matplotlib
* math
* IPython import display
* xgboost
* sklearn
* plotly
* datetime
* scipy
* zscore


## <a id="3">Data and variables</a>
For the analysis and Machine Learning I have used The complete _Our World in Data_ COVID-19 dataset

**Our complete COVID-19 dataset is available in [CSV](https://covid.ourworldindata.org/data/owid-covid-data.csv), [XLSX](https://covid.ourworldindata.org/data/owid-covid-data.xlsx), and [JSON](https://covid.ourworldindata.org/data/owid-covid-data.json) formats, and includes all of our historical data on the pandemic up to the date of publication.**

The CSV and XLSX files follow a format of 1 row per location and date. The JSON version is split by country ISO code, with static variables and an array of daily records.

The variables represent all of our main data related to confirmed cases, deaths, hospitalizations, and testing, as well as other variables of potential interest.

As of 26 January 2021, the columns are: `iso_code`, `continent`, `location`, `date`, `total_cases`, `new_cases`, `new_cases_smoothed`, `total_deaths`, `new_deaths`, `new_deaths_smoothed`, `total_cases_per_million`, `new_cases_per_million`, `new_cases_smoothed_per_million`, `total_deaths_per_million`, `new_deaths_per_million`, `new_deaths_smoothed_per_million`, `reproduction_rate`, `icu_patients`, `icu_patients_per_million`, `hosp_patients`, `hosp_patients_per_million`, `weekly_icu_admissions`, `weekly_icu_admissions_per_million`, `weekly_hosp_admissions`, `weekly_hosp_admissions_per_million`, `total_tests`, `new_tests`, `total_tests_per_thousand`, `new_tests_per_thousand`, `new_tests_smoothed`, `new_tests_smoothed_per_thousand`, `positive_rate`, `tests_per_case`, `tests_units`, `total_vaccinations`, `people_vaccinated`, `people_fully_vaccinated`, `new_vaccinations`, `new_vaccinations_smoothed`, `total_vaccinations_per_hundred`, `people_vaccinated_per_hundred`, `people_fully_vaccinated_per_hundred`, `new_vaccinations_smoothed_per_million`, `stringency_index`, `population`, `population_density`, `median_age`, `aged_65_older`, `aged_70_older`, `gdp_per_capita`, `extreme_poverty`, `cardiovasc_death_rate`, `diabetes_prevalence`, `female_smokers`, `male_smokers`, `handwashing_facilities`, `hospital_beds_per_thousand`, `life_expectancy`, `human_development_index`

A [full codebook](https://github.com/owid/covid-19-data/tree/master/public/data/owid-covid-codebook.csv) is made available, with a description and source for each variable in the dataset.

## <a id="4">Questions and Problems</a>
Questions

 - How many total population in each location by continents from our dataset?
- The 10 top population total in each location by continents from our dataset?
- Show countries in Asia, Europe and North America the total_cases and total_deaths,new_cases,total_tests, total_vaccinations by mean, and max?
- Let's see the speed of transmission of the Corona virus between countries on the map ?
- Let's see number of total_cases,total_deaths,total_deaths_per_million,test per confirmed(%) on map ?
- Top 15 countries for the total_cases,total_deaths,total_deaths_per_million,total_tests ,people_fully_vaccinated and total_vaccinations on plot_hbar and Visulizing Treemaps?
- How many the New Deaths Smoothed day by day in continents ?
- How many the New Tests Smoothed day by day in continents ?
- find some gdp_per_capita and new_cases clusters over countries ?
- find some new_deaths_smoothed_per_million, life_expectancy and hospital_beds_per_thousand clusters over countries ?
- find some new_deaths_smoothed_per_million, handwashing_facilities and extreme_poverty clusters over countries ?
- find some new_deaths_smoothed_per_million, life_expectancy and hospital_beds_per_thousand clusters over countries?
- Which vaccination scheme is used most?

### 1- Problem Question : 

Created a Linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 days. In this project I have used sklearn for creating Linear Regression model and created training split with 80 to 20%. The trained the model and predicted the death for next 30 days. Also created model using XGBoost for improving the linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 days.

### 2- Problem Question:
I will create a model that can predict the risk for the Case Mortality Ratio of a Country utilizing its Life Expectancy, Percentage of Population over 65, and Percentage of diabetes_prevalence and cardiovasc_death_rate ?

It decided on using Population Over Age 65 and Obesity because in the world, over 80% of the deaths were in the population 65 and over, and the CDC has stated that 94% of deaths had some underlying health condition. We also used Life Expectency per country to account for possible deficiencies in the health care system. John Hopkins University has listed several diseases such as heart disease and Diabetes which are known to be exacerbated by Obesity. Our idea is that we can more accurately predict the Mortality Ratio of COVID-19 by using both population 65 and over and Obesity rather than just population 65 and over. This may show that creating a healthier population is the best way to prevent the devastation in future pandemics that the world is currently facing

## <a id="5">Perform data preparation and Cleaning</a>

For now, let's assume this was indeed a data entry error. We can use one of the following approaches for dealing with the missing or faulty value:

- Replace it with 0.
- Replace it with the average of the entire column
- Replace it with the average of the values on the previous & next date
- Discard the row entirely
  Which approach you pick requires some context about the data and the problem. In this case, since we are dealing with data ordered by date, we can go ahead with the one approach

   It is not really logical to delete Nan values but replace with 0, because that would confirm that the result was static because the data is historical and adopts high time series, we cannot replace or delete even the most data in the rows because it is data historical 
 
## <a id="6">Evaluate the Results</a>
 
 For the EDA and Visualization I have used Our World in Data. Based on the research, I decided to use packages like Plotly and seaborn for better visualization, also it needed jupyter extensions to display the graph. I have created couple of plots and maps, created bar plot to display top 15 countries with highest number of death rates created sub bar plots for displaying Top 15 Countries with Confirmed Cases, Top 15 Countries with total_deaths , Top 15 Countries with new Cases, and Top 15 Countries with people_fully_vaccinated and total_vaccinations . Also created a bar chart for comparing confirmed, death and total cases. Created correlation heat map to find out the highly correlated variables. Created Scatter geo map for displaying the trend over the time and created including variables like gdp_per_capita , diabetes_prevalence ,female_smokers, male_smokers, hospital_beds_per_100k, cvd_death_rate. visualizations of the analysis are included in the attached and answer for some questions
 
see the notbook or html file [Capston-dsnd-project-Covid19](https://github.com/marwan1023/Data-Scientist-Nanodegree-Program/blob/master/Capston-dsnd-project-Covid19/Capston-dsnd-project.ipynb)
 
## <a id="7">Modeling</a>

### 1- Linear Regression-Forecast
Created a Linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 days. In this project I have used sklearn for creating Linear Regression model and created training split with 80 to 20%. The trained the model and predicted the death for next 30 days. Also created model using XGBoost for improving the linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 day.

 #### Linear Regression Model: used to predict future values from past values
- Used OneWorld.org data for modeling.
- Predicted the Death rate for next 30 days with independent variable like age, gdp, diabetes, smokers, hospital beds etc.
- Split into 80% train and 20% test data.
- Linear Regression Fitted with Accuracy : 0.9976567640125802
- Used sklearn.model_selection for train_test_split
- Used sklearn.linear_model for LinearRegression
- Orange line indicates the future forecast of Death.
- Predicted the Death rate for next 30 days in the forecast column

#### XGboost algorithm to see if we can get better results

- XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Kubernetes, Hadoop, SGE, MPI, Dask) and can solve problems beyond billions of examples.

- Used OneWorld.org data for modeling.
- XGBoost will generally fit training data much better than linear regression.
- Orange line indicates the death prediction
 
### 2- k-nearest neighbors(KNN) algorithm

   will create a model that can predict the risk for the Case Mortality Ratio of a Country utilizing its Life Expectancy, Percentage of Population over 65, and Percentage of     diabetes_prevalence and cardiovasc_death_rate 
  It decided on using Population Over Age 65 and diabetes_prevalence cardiovasc_death_rate because in the world, over 80% of the deaths were in the population 65 and over, and   the CDC has stated that 94% of deaths had some underlying health condition. We also used Life Expectancy per country to account for possible deficiencies in the health care  system. John Hopkins University has listed several diseases such as heart disease and Diabetes which are known to be exacerbated by cardiovasc_death_rate and Obesity. Our idea  is that we can more accurately predict the Mortality Ratio of COVID-19 by using both population 65 and over and Obesity rather than just population 65 and over. This may show  that creating a healthier population is the best way to prevent the devastation in future pandemics that the world is currently facing.
After viewing the graphs in Linear Regression-Forecast we the accuracy that XGboost algorithms can achieve with this data. . We will continue and see if our ML Algorithm can do better than we are expecting. We have initially chosen to use categorization with the HighRisk category as that may be more accurate than regression.
Or can we use more precise algorithms to build a data-appropriate learning model?

- We then implemented a cross_validation scoring utilizing a cv of 8 to make sure the randomness of our train/test data was not being affected and got something similar, but with a nterestingly a slightly better classification of 90.518% with k =7  
- The model using the extra features especially the human_development_index, smoker data and more recent target data has gotten better at predicting a countries rate of mortality vs population going from 90% to 93% accuracy depending on the randomization.

 - This may be due to different reporting systems for what is and is not a covid death and overall accuracy of the inputs.

- Our original Hypotheses that Age and Obesity would be factors seem to have been proven true through the data, one might even be able to try regression on the normalized mortality / population and if we had the One World Data originally we may have even gone further and tried that as the correlation seems to be stronger


 ## <a id="8">Write A COVID-19 Analysis Post</a>
 
This project is also featured on the Medium website @ [Analyzing Data on COVID-19 (coronavirus) by Our World in Data.](https://marwanalsharabbi.medium.com/analyzing-data-on-covid-19-coronavirus-by-our-world-in-data-d61676b11586)
 
 ### Recommendation
 - Assuming the published data are reliable, the SIR model (1) can be applied to assess the spread of the COVID-19 disease and predict the number of infected, removed and recovered populations and deaths in the communities, accommodating at the same time possible surges in the number of susceptible individuals.
 - Preferably exclude vaccine columns, analyze and analyze data for example @ [covid-world-vaccination-progress](https://www.kaggle.com/gpreda/covid-world-vaccination-progress)
 -  It collects separate vaccine data, analyzes it, and creates a prediction model for daily and monthly vaccination.

#### References and Future Work
    
- https://www.geeksforgeeks.org/python-programming-language/?ref=leftbar

- https://www.python-course.eu/python3_class_and_instance_attributes.php

- https://thispointer.com/data-analysis-in-python-using-pandas/

- https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas

- https://ourworldindata.org/coronavirus

- https://covid19.moh.gov.sa/

- https://github.com/

- https://www.kaggle.com/
 
 
