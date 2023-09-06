# DataScience_Churn-Analysis-ML
## Description
A fictitious telecom company providing home phone and Internet services in California wants to develop a machine learning model that can predict customers who will leave. The aim of this project is to create a system that will predict lost customers with different machine learning models. Random Forest, XGBoost, LightGBM and CatBoost algorithms were used for this purpose.
## Content
Telco customer churn data contains information about a fictitious telecom company that provided home phone and Internet services to 7043 customers in California in the third quarter. Which customers left, stayed or signed up for their service shows.
## Columns
    CustomerId : Customer Id
    Gender : Gender
    SeniorCitizen : Whether the customer is elderly (1, 0)
    Partner : Whether the client has a partner (Yes, No)
    Dependents : Whether the client has dependents (Yes, No
    tenure : Number of months the customer stays with the company
    PhoneService : Whether the customer has telephone service (Yes, No)
    MultipleLines : Whether the customer has more than one line (Yes, No, No telephone service)
    InternetService : Customer's internet service provider (DSL, Fiber optic, No)
    OnlineSecurity : Whether the customer has online security (Yes, No, No Internet service)
    OnlineBackup : Whether the customer has an online backup (Yes, No, No Internet service)
    DeviceProtection : Whether the customer has device protection (Yes, No, No Internet service)
    TechSupport : Whether the customer receives technical support (Yes, No, No Internet service)
    StreamingTV : Whether the customer has streaming TV (Yes, No, No Internet service)
    StreamingMovies : Whether the customer has streaming movies (Yes, No, No Internet service)
    Contract : Customer's contract period (Month to month, One year, Two years)
    PaperlessBilling : Whether the customer has a paperless invoice (Yes, No)
    PaymentMethod : Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit c(automatic))
    MonthlyCharges : Amount charged to the customer monthly
    TotalCharges : Total amount charged to the customer
    Churn Whether the customer used it or not (Yes or No)

## Flow
- The data was uploaded and generally analyzed and some structural changes were made.
- Categorical, numeric columns captured.
- Outlier analysis was performed. Outliers were replaced with upper and lower limits by suppression.
- Missing value analysis was done and problems were solved.
- Various new columns have been added through feature engineering.
- Label endocing was applied to binary variables.
- One-hot encoding was applied to multi-class categorical variables.
- Standardization was applied to numeric variables.
- Dependent and independent variables were created over the train data.
- Models were built with different algorithms for model selection.
- Random Forest, XGBoost, LightGBM and CatBoost algorithms were used for models.
- We looked at the importance of features. 
