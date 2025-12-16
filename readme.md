# Customer-Churn-Prediction

## Build a system that predicts if a customer will stop using a service (churn). You’ll clean and analyze data, create useful features, train models (Logistic Regression, Random Forest, XGBoost), and deploy your model with Streamlit so users can test churn predictions.

## Telco Customer Churn Analysis Report

1.	Introduction
This report presents a full analysis of customer churn for a telecommunications provider. The goal is to identify why customers leave and to build a predictive model that flags at-risk customers early. Insights from this report enable more effective retention strategies, improved customer experience, and stronger revenue stability.

2.	Data Overview
The dataset contains demographic information, contract details, billing data, and service usage. The target variable is Churn, indicating whether a customer has discontinued service. Before modeling, the data was cleaned, encoded, and structured to ensure accuracy and consistency.

   The dataset utilized for this project was "/content/WA_Fn-UseC_-Telco-Customer-Churn.csv", containing 7043 rows (customers) and 21 columns (features) with 'Churn' as the    target variable.

## Analysis Steps:

1.	Data Cleaning and Preprocessing:
- The TotalCharges column, initially an object type, was converted to a numeric (float) type. During this process, null values introduced by coercion were implicitly        handled as NaN. (However, it was observed that no nulls were initially present based on df.isna().sum() and df.info() output, but pd.to_numeric can introduce them).
- 	Checked for and confirmed no duplicate rows were present.
-	No outliers were explicitly checked or handled.
  
2.	Exploratory Data Analysis (EDA) and Feature Engineering:
-	Correlation Analysis: Explored the relationships between numerical features (SeniorCitizen, tenure, MonthlyCharges, TotalCharges). Key findings included a strong positive correlation between tenure and TotalCharges (0.83), and a moderate positive correlation between MonthlyCharges and TotalCharges (0.65).
-	Churn Distribution: Examined the overall distribution of churn, revealing an imbalanced dataset with more 'No' churn instances than 'Yes' churn.
  
	Impact Analysis: Analyzed the impact of various features on churn:
-	MonthlyCharges: Customers who churned had higher average monthly charges. Further binned MonthlyCharges into custom categories (18-40, 41-60, 61-80, 81-100, 101+) to    observe churn rates, noting higher churn in the 61-80 and 81-100 categories.
-	gender: No significant difference in churn rates between genders when considering MonthlyCharges.
-	tenure: Customers who churned had a significantly lower average tenure (approx. 18 months) compared to those who did not churn (approx. 37.5 months).
-	Contract: Longer contract terms were associated with lower average monthly charges and lower churn rates.
-	Categorical Features: Conducted crosstab analysis for SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, and PaymentMethod. Key observations included higher churn among customers without online security, tech support, device protection, senior citizens, those with month-to-month contracts, and those using electronic checks.
  
	Feature Engineering: Created a new categorical feature MonthlyCharges_Category_Custom by binning the MonthlyCharges column to analyze churn patterns within specific charge ranges.

## Model Training and Evaluation:

Three different machine learning models were trained and evaluated for churn prediction:
1.	Random Forest Classifier:
- 	Configuration: n_estimators=200, max_depth=10, class_weight='balanced', random_state=42.
- 	Evaluation Metrics: Accuracy, Classification Report (precision, recall, f1-score for both classes), and Confusion Matrix.
-	Performance: Achieved an accuracy of 0.769, with a recall for churn (class 1) of 0.71 and precision of 0.55.
- 	Feature Importance: Identified Contract, tenure, MonthlyCharges, OnlineSecurity, TechSupport, and InternetService as key drivers of churn.
  
2.	Logistic Regression:
-	Configuration: max_iter=1000, class_weight='balanced', random_state=42.
-	Evaluation Metrics: Accuracy, Classification Report, and Confusion Matrix.
-	Performance: Achieved an accuracy of 0.736, with a recall for churn (class 1) of 0.80 and precision of 0.50.
-	Feature Importance (Coefficients): Highlighted Contract (negative coefficient, indicating retention for longer terms) and PaperlessBilling (positive coefficient, indicating churn driver) as most impactful.
  
3.	XGBoost Classifier:
-	Configuration: n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight for class imbalance, random_state=42, eval_metric='logloss'.
-	Evaluation Metrics: Accuracy, Classification Report, and Confusion Matrix.
-	Performance: Achieved an accuracy of 0.748, with a recall for churn (class 1) of 0.76 and precision of 0.52.
-	Feature Importance: Similar to Random Forest, Contract, InternetService, OnlineSecurity, and TechSupport were important.

4.	Exploratory Insights
Several behavioral and service-related patterns were observed:
   -  Customers on month-to-month contracts leave more frequently.
   -  New customers with lower tenure show a higher churn rate.
   -  Higher monthly charges correlate with dissatisfaction and churn.
   -  Customers lacking online security, tech support, or device protection are more likely to leave.
   -  Fiber optic customers exhibited higher churn than DSL customers.

5.	Predictive Modeling
Three models were trained: Random Forest, Logistic Regression, and XGBoost. Random Forest achieved the strongest balance of accuracy, recall, and robustness, making it the recommended model for deployment. Logistic Regression offered the best explainability, while XGBoost performed competitively across metrics.

6.	Key Drivers of Churn
The analysis revealed major churn drivers:
-  Month-to-month contract type
-  Low tenure
-  High monthly charges
-  Lack of support services (Tech Support, Online Security)
-  Fiber optic internet service
-  Paperless billing
-  Senior citizen customers
-  Lack of device protection These factors provide the foundation for targeted retention efforts.

7.	Model Deployment
The Random Forest model has been deployed using Streamlit. All encoders, the model file, and preprocessing steps have been packaged for real-time predictions. The app allows business teams to input customer attributes and instantly receive churn predictions, supporting fast, data-driven decision-making.

### Model Recommendations:

Based on the comparative analysis, Random Forest and XGBoost models demonstrate superior predictive performance for identifying customer churn. Therefore, Random Forest or XGBoost are recommended for deployment in production environments where accurate churn prediction is paramount. Their higher accuracy and robust handling of complex relationships make them ideal for predictive tasks.

For explaining churn drivers to stakeholders and gaining interpretable insights, Logistic Regression is the preferred model. Its coefficients directly indicate the direction and magnitude of a feature's impact on churn, making it excellent for communicating 'why' a customer might churn.

### Key Churn Drivers:

The analysis consistently highlighted several significant factors driving customer churn across all models:
•	Contract Type: Month-to-month contracts are a very strong indicator of churn.
•	Tenure: Customers with shorter tenures are significantly more likely to churn.
•	Monthly Charges: Higher monthly charges are associated with increased churn, especially in the 61-100+ range.
•	Online Security: Lack of online security services is a major churn driver.
•	Tech Support: Absence of tech support services correlates with higher churn.
•	Internet Service Type: Fiber optic internet service users show a higher propensity to churn compared to DSL users.
•	Paperless Billing: Customers opted for paperless billing tend to churn more.
•	Senior Citizen: Senior citizens have a higher churn rate.
•	Device Protection: Lack of device protection also contributes to churn.

## Actionable Retention Strategies:

To mitigate customer churn, the telecom company should implement targeted strategies focusing on the identified drivers:

1.	Target Month-to-Month Contract Customers: Offer incentives (e.g., discounts, bundled services, loyalty bonuses) to encourage customers on month-to-month contracts to switch to longer-term contracts (one-year or two-year), thereby increasing their commitment.
2.	Early Tenure Engagement: Implement proactive engagement programs for new customers, particularly within their first year, to ensure satisfaction and address any early issues that might lead to churn.
3.	Bundle Security & Support Services: Actively promote and bundle online security and tech support services, especially for customers currently without them. Highlight the value and benefits of these services to improve retention.
4.	Address Fiber Optic Customer Satisfaction: Investigate the reasons for higher churn among fiber optic users. This could involve improving service reliability, offering better customer support for technical issues specific to fiber optic, or re-evaluating pricing strategies for this segment.
5.	Review Paperless Billing Experience: Understand why paperless billing is associated with higher churn. It might indicate a segment that is less engaged or more price-sensitive. Consider offering personalized communications or benefits to these customers.
6.	Senior Citizen Specific Programs: Develop tailored plans, support services, or educational resources for senior citizens, addressing their specific needs and concerns to improve their loyalty.
7.	Optimize Pricing for High Monthly Charges: For customers with high monthly charges, especially those in the 61−100+ range, consider offering personalized plans, loyalty discounts, or value-added services to prevent them from seeking alternatives.
8.	Offer Device Protection Incentives: Promote device protection plans more actively, perhaps by bundling them with other services or offering initial trial periods.

9.	Business Impact
    Implementing churn prediction and retention strategies will:
-  Reduce customer loss and stabilize recurring revenue.
-  Improve targeting for marketing and retention campaigns.
-  Enhance customer satisfaction and lifetime value.
-  Support strategic service improvements based on data-driven insights.
  Overall, this system empowers the business with proactive churn management capabilities.

6.	Strategic Recommendations
Based on the insights, the following strategies are recommended:
  -   Reduce month-to-month churn through incentives and loyalty programs.
  -  Improve early customer experience with onboarding support.
  -  Increase adoption of service bundles such as security and tech support.
  -  Diagnose and address dissatisfaction among fiber optic users.
  -  Offer value-based pricing to high-charge customers.
  -  Improve engagement and communication for paperless billing users.
  -  Provide simplified plans and dedicated support for senior citizens.
  -  Promote device protection bundles to reduce service frustration.


  Conclusion
The churn analysis and predictive model provide a clear understanding of why customers leave and how to prevent it. Using the Random Forest model, the company can identify at-risk customers early and take effective action. This report lays the groundwork for stronger retention strategies, improved customer experience, and long-term revenue growth.

View Deployment of Analysis
https://customer-churn-prediction-tele-app.streamlit.app/
