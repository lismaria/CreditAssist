The Credit Card Classification project provides a web-based solution for predicting credit classification based on financial and personal attributes. It automates the credit assessment process using machine learning, offering quick insights into creditworthiness. By leveraging historical credit data, the project trains models such as Support Vector Machines (SVM), Random Forests (RF), Gradient Boosting (GB), and Logistic Regression (LR)  to make predictions. Users input their information through a user-friendly interface, and the system generates predictions, aiding both lenders and borrowers in making informed decisions efficiently.

## Website Snapshots
![landing_page](https://github.com/lismaria/CreditAssist/assets/69604870/9e2e1318-b2de-4bd1-99db-75772820dc3a)
![about_us](https://github.com/lismaria/CreditAssist/assets/69604870/2c512654-e0ef-48dc-a5dc-314e12baaad2)
![predict_1](https://github.com/lismaria/CreditAssist/assets/69604870/707e275f-2e2c-4178-bd57-eb0f445e98bd)
![predict_2](https://github.com/lismaria/CreditAssist/assets/69604870/56c2cf8f-a719-49ad-a8ee-6ad798952610)
![predict_3](https://github.com/lismaria/CreditAssist/assets/69604870/e4aaac62-4243-4ef9-995c-01ad76d6bfff)
![result_good](https://github.com/lismaria/CreditAssist/assets/69604870/611d1dad-d331-4722-8746-c3d0a55ccdb2)
![result_bad](https://github.com/lismaria/CreditAssist/assets/69604870/1d40009e-dcd9-47d6-ac53-e648be364dfb)

## Dataset Description
Dataset Description
The dataset comprises 1000 samples, each containing various features related to individuals applying for credit.
Following are the features :
1.	CHK_ACCT: Checking account status, categorized into different levels such as "0DM" (no checking account), "less-200DM" (less than 200 DM balance), or "no-account" (no account exists).
2.	Duration: Duration of the credit in months.
3.	History: Credit history status, categorized into levels such as "critical", "duly-till-now", or "delay".
4.	Purpose of credit: Purpose for which the credit is taken, such as "radio-tv", "education", "furniture", etc.
5.	Credit Amount: Amount of credit in DM (Deutschmark).
6.	Balance in Savings A/C: Balance in the savings account, categorized into different levels like "unknown", "less100DM", or "over1000DM".
7.	Employment: Duration of employment, categorized into different levels such as "over-seven" (more than seven years), "four-years", etc.
8.	Install_rate: Installment rate in percentage of disposable income.
9.	Marital status: Marital status of the applicant, categorized into different levels such as "single-male", "female-divorced", "married-male", etc.
10.	Co-applicant: Presence of a co-applicant, categorized as "none" or otherwise.
11.	Present Resident: Duration of present residence, categorized into different levels such as "1" (less than one year) or "4" (more than four years).
12.	Real Estate: Presence of real estate, categorized as "car", "building-society", etc.
13.	Age: Age of the applicant.
14.	Other installment: Presence of other installments, categorized as "none" or otherwise.
15.	Residence: Type of residence, categorized as "own", "rent", etc.
16.	Num_Credits: Number of existing credits at the bank.
17.	Job: Job status of the applicant, categorized as "skilled", "unskilled-resident", "management", etc.
18.	No. dependents: Number of dependents.
19.	Phone: Availability of a phone, categorized as "yes" or "no".
20.	Foreign: Foreign worker status, categorized as "yes" or "no".
21.	Credit classification (Target Variable): Classification of credit risk, labeled as "good" or "bad".
