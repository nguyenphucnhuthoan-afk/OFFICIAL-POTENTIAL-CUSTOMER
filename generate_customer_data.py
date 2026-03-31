import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed(42)
n = 50000

# 1. Generate numerical features
data = pd.DataFrame()
data['Age'] = np.random.randint(18, 71, n)
data['AnnualIncome'] = np.round(np.random.normal(60000, 30000, n))
data['AnnualIncome'] = data['AnnualIncome'].clip(15000, 200000)
data['NumberOfPurchases'] = np.random.poisson(20, n).clip(1, 100)
data['TimeSpentOnWebsite'] = np.round(np.random.normal(30, 15, n)).clip(1, 180)
data['CustomerTenureYears'] = np.random.randint(1, 16, n)
data['LastPurchaseDaysAgo'] = np.random.randint(0, 366, n)
data['SessionCount'] = np.random.poisson(30, n).clip(1, 200)
data['CustomerSatisfaction'] = np.random.choice([1,2,3,4,5], size=n, p=[0.1,0.15,0.25,0.3,0.2])

# 2. Generate categorical features
genders = ['Male', 'Female', 'Other']
product_categories = ['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports']
devices = ['Mobile', 'Desktop', 'Tablet']
regions = ['North', 'South', 'East', 'West']
referral_sources = ['Social Media', 'Search Engine', 'Referral', 'Direct']
segments = ['Low Value', 'Medium Value', 'High Value']

data['Gender'] = np.random.choice(genders, n, p=[0.48, 0.48, 0.04])
data['ProductCategory'] = np.random.choice(product_categories, n)
data['PreferredDevice'] = np.random.choice(devices, n, p=[0.7, 0.25, 0.05])
data['Region'] = np.random.choice(regions, n)
data['ReferralSource'] = np.random.choice(referral_sources, n)

# 3. CustomerSegment logic
def assign_segment(row):
    if row['AnnualIncome'] > 70000 and row['NumberOfPurchases'] > 50:
        return 'High Value'
    elif row['AnnualIncome'] > 50000 and row['NumberOfPurchases'] > 20:
        return 'Medium Value'
    else:
        return 'Low Value'

data['CustomerSegment'] = data.apply(assign_segment, axis=1)

# 4. LoyaltyProgram logic
data['LoyaltyProgram'] = np.where(data['CustomerSatisfaction'] > 3, 'Yes', 'No')

# 5. DiscountsAvailed logic
data['DiscountsAvailed'] = np.where(np.random.rand(n) < 0.5, 'Yes', 'No')

# 6. PurchaseStatus logic
def purchase_status(row):
    base = 0.2
    if row['AnnualIncome'] > 70000: base += 0.25
    if row['TimeSpentOnWebsite'] > 40: base += 0.2
    if row['CustomerSegment'] == 'High Value' : base += 0.2
    if row['LoyaltyProgram'] == 'Yes': base += 0.05
    if row['DiscountsAvailed'] == 'Yes': base += 0.1
    # Add noise
    base += np.random.normal(0, 0.08)
    return 1 if base > 0.5 else 0

data['PurchaseStatus'] = data.apply(purchase_status, axis=1)

# 7. Add noise to 10% of rows
def add_noise(row):
    if np.random.rand() < 0.1:
        # Flip PurchaseStatus
        row['PurchaseStatus'] = 1 - row['PurchaseStatus']
        # Randomize CustomerSatisfaction
        row['CustomerSatisfaction'] = np.random.choice([1,2,3,4,5])
        # Randomize LoyaltyProgram
        row['LoyaltyProgram'] = np.random.choice(['Yes', 'No'])
    return row

data = data.apply(add_noise, axis=1)

# 8. Save to CSV
data.to_csv('customerData_50k.csv', index=False)

# 9. Correlation matrix
print('Correlation Matrix:')
print(data[[
    'Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite',
    'CustomerTenureYears', 'LastPurchaseDaysAgo', 'SessionCount', 'CustomerSatisfaction', 'PurchaseStatus'
]].corr())

# 10. Describe statistics
print('\nDescriptive Statistics:')
print(data.describe())

# 11. Target distribution
print('\nPurchaseStatus Distribution:')
print(data['PurchaseStatus'].value_counts(normalize=True))