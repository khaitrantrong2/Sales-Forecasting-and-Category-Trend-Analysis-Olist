## Sales-Forecasting-and-Category-Trend-Analysis-Olist
This project analyzes the Olist Brazilian E‑commerce dataset to explore sales trends, customer behavior, product categories, and to build a short‑term sales forecast.
The goal is to support high‑level inventory and marketing planning using historical data.

## 1. Project Objectives
- Clean and prepare the Olist dataset for analysis
- Explore sales patterns (daily, weekly, monthly)
- Identify top‑performing product categories
- Analyze customer reviews and delivery performance
- Build a 60‑day revenue forecast using time‑series modeling
- Provide high‑level inventory & marketing recommendations

## 2. Dataset, Tools & Technologies
Dataset
- Source: Olist Brazilian E‑commerce public dataset - https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Key tables used:

| Table | Purpose |
|---|---|
| orders | Order-level info (status & timestamps) |
| order_items | Item lines, price & freight |
| products | Product attributes & category |
| order_reviews | Review score & comments |
| order_payments | Payment type & value |
| customers | Customer location (city/state) |
| sellers | Seller location (city/state) |
| geolocation | Zip code → lat/long mapping |
| product_category_name_translation | PT → EN category mapping |


Tools & Technologies
- Python (Pandas, NumPy, Matplotlib, Seaborn, Statsmodels)
- Google Colab (Notebook)
- GitHub (Markdown - Readme)
- Word (Summary Report) 
- PowerPoint (Presentation Slide)
  
## 3. Data Preparation
The notebook performs:

Step 1: Set-up and Install dependencies
- Open the notebook in Colab.
- Install dependencies

Use code below:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```
Step 2: Load and read data.

- Find dataset - Example: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download

- Use 9 files download from Kaggle and upload the downloaded CSV files from the dataset.

Use code below:
```python
from google.colab import files
uploaded = files.upload()
```

- Read data:

Use code below:
```python
customers = pd.read_csv('olist_customers_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
order_payments = pd.read_csv('olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
product_category = pd.read_csv('product_category_name_translation.csv')
```
Step 3: Data type standardization

- To see sample of each source

Use code below:
```python
geolocation.head()
order_items.head()
order_payments.head()
order_reviews.head()
orders.head()
products.head()
sellers.head()
product_category.head()
customers.head()
```
- To check data type of each source

Use code below:
```python
geolocation.info()
order_items.info()
order_payments.info()
order_reviews.info()
orders.info()
products.info()
sellers.info()
product_category.info()
customers.info()
```

From result of checking sample and data type we converted:
- IDs → string
- Dates → datetime
- Prices & freight → numeric
- Ensure coerced invalid timestamps.


Use code below for string convert:
```python
geolocation["geolocation_city"] = geolocation["geolocation_city"].astype("string")
geolocation["geolocation_state"] = geolocation["geolocation_state"].astype("string")
geolocation["geolocation_zip_code_prefix"] = geolocation["geolocation_zip_code_prefix"].astype("string")
order_items["order_id"] = order_items["order_id"].astype("string")
order_items["product_id"] = order_items["product_id"].astype("string")
order_items["seller_id"] = order_items["seller_id"].astype("string")
order_payments["order_id"] = order_payments["order_id"].astype("string")
order_payments["payment_type"] = order_payments["payment_type"].astype("string")
order_reviews["review_id"] = order_reviews["review_id"].astype("string")
order_reviews["order_id"] = order_reviews["order_id"].astype("string")
order_reviews["review_comment_title"] = order_reviews["review_comment_title"].astype("string")
order_reviews["review_comment_message"] = order_reviews["review_comment_message"].astype("string")
orders["order_id"] = orders["order_id"].astype("string")
orders["customer_id"] = orders["customer_id"].astype("string")
orders["order_status"] = orders["order_status"].astype("string")
products["product_id"] = products["product_id"].astype("string")
products["product_category_name"] = products["product_category_name"].astype("string")
sellers["seller_id"] = sellers["seller_id"].astype("string")
sellers["seller_city"] = sellers["seller_city"].astype("string")
sellers["seller_state"] = sellers["seller_state"].astype("string")
sellers["seller_zip_code_prefix"] = sellers["seller_zip_code_prefix"].astype("string")
product_category["product_category_name"] = product_category["product_category_name"].astype("string")
product_category["product_category_name_english"] = product_category["product_category_name_english"].astype("string")
customers["customer_id"] = customers["customer_id"].astype("string")
customers["customer_unique_id"] = customers["customer_unique_id"].astype("string")
customers["customer_city"] = customers["customer_city"].astype("string")
customers["customer_state"] = customers["customer_state"].astype("string")
customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].astype("string")
```

Use code below for datetime convert:
```python
order_items["shipping_limit_date"] = pd.to_datetime(order_items["shipping_limit_date"], errors="coerce")
order_reviews["review_answer_timestamp"] = pd.to_datetime(order_reviews["review_answer_timestamp"], errors="coerce")
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
orders["order_approved_at"] = pd.to_datetime(orders["order_approved_at"], errors="coerce")
orders["order_delivered_carrier_date"] = pd.to_datetime(orders["order_delivered_carrier_date"], errors="coerce")
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
```

Use code below for numeric convert:
```python
order_payments["payment_value"] = pd.to_numeric(order_payments["payment_value"], errors="coerce")
order_items["price"] = pd.to_numeric(order_items["price"], errors="coerce")
order_items["freight_value"] = pd.to_numeric(order_items["freight_value"], errors="coerce")
```

Check result data type after convert:

Use code below (replicate respectively 9 times for 9 files):
```python
geolocation.info()
```

1. geolocation table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000163 entries, 0 to 1000162
Data columns (total 5 columns):
 #   Column                       Non-Null Count    Dtype  
---  ------                       --------------    -----  
 0   geolocation_zip_code_prefix  1000163 non-null  string 
 1   geolocation_lat              1000163 non-null  float64
 2   geolocation_lng              1000163 non-null  float64
 3   geolocation_city             1000163 non-null  string 
 4   geolocation_state            1000163 non-null  string 
dtypes: float64(2), string(3)
memory usage: 38.2 MB
```
2. order_items table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 112650 entries, 0 to 112649
Data columns (total 7 columns):
 #   Column               Non-Null Count   Dtype         
---  ------               --------------   -----         
 0   order_id             112650 non-null  string        
 1   order_item_id        112650 non-null  int64         
 2   product_id           112650 non-null  string        
 3   seller_id            112650 non-null  string        
 4   shipping_limit_date  112650 non-null  datetime64[ns]
 5   price                112650 non-null  float64       
 6   freight_value        112650 non-null  float64       
dtypes: datetime64[ns](1), float64(2), int64(1), string(3)
memory usage: 6.0 MB
```
3. order_payments table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 103886 entries, 0 to 103885
Data columns (total 5 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   order_id              103886 non-null  string 
 1   payment_sequential    103886 non-null  int64  
 2   payment_type          103886 non-null  string 
 3   payment_installments  103886 non-null  int64  
 4   payment_value         103886 non-null  float64
dtypes: float64(1), int64(2), string(2)
memory usage: 4.0 MB
```
4. Order_review
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 99224 entries, 0 to 99223
Data columns (total 7 columns):
 #   Column                   Non-Null Count  Dtype         
---  ------                   --------------  -----         
 0   review_id                99224 non-null  string        
 1   order_id                 99224 non-null  string        
 2   review_score             99224 non-null  int64         
 3   review_comment_title     11568 non-null  string        
 4   review_comment_message   40977 non-null  string        
 5   review_creation_date     99224 non-null  datetime64[ns]
 6   review_answer_timestamp  99224 non-null  datetime64[ns]
dtypes: datetime64[ns](2), int64(1), string(4)
memory usage: 5.3 MB
```
5.Order table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 99441 entries, 0 to 99440
Data columns (total 8 columns):
 #   Column                         Non-Null Count  Dtype         
---  ------                         --------------  -----         
 0   order_id                       99441 non-null  string        
 1   customer_id                    99441 non-null  string        
 2   order_status                   99441 non-null  string        
 3   order_purchase_timestamp       99441 non-null  datetime64[ns]
 4   order_approved_at              99281 non-null  datetime64[ns]
 5   order_delivered_carrier_date   97658 non-null  datetime64[ns]
 6   order_delivered_customer_date  96476 non-null  datetime64[ns]
 7   order_estimated_delivery_date  99441 non-null  datetime64[ns]
dtypes: datetime64[ns](5), string(3)
memory usage: 6.1 MB
```
6.Product table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32951 entries, 0 to 32950
Data columns (total 9 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   product_id                  32951 non-null  string 
 1   product_category_name       32341 non-null  string 
 2   product_name_lenght         32341 non-null  float64
 3   product_description_lenght  32341 non-null  float64
 4   product_photos_qty          32341 non-null  float64
 5   product_weight_g            32949 non-null  float64
 6   product_length_cm           32949 non-null  float64
 7   product_height_cm           32949 non-null  float64
 8   product_width_cm            32949 non-null  float64
dtypes: float64(7), string(2)
memory usage: 2.3 MB
```
7. Seller table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3095 entries, 0 to 3094
Data columns (total 4 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   seller_id               3095 non-null   string
 1   seller_zip_code_prefix  3095 non-null   string
 2   seller_city             3095 non-null   string
 3   seller_state            3095 non-null   string
dtypes: string(4)
memory usage: 96.8 KB
```
8. Product_category table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 71 entries, 0 to 70
Data columns (total 2 columns):
 #   Column                         Non-Null Count  Dtype 
---  ------                         --------------  ----- 
 0   product_category_name          71 non-null     string
 1   product_category_name_english  71 non-null     string
dtypes: string(2)
memory usage: 1.2 KB
```
9. Customer table
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 99441 entries, 0 to 99440
Data columns (total 5 columns):
 #   Column                    Non-Null Count  Dtype 
---  ------                    --------------  ----- 
 0   customer_id               99441 non-null  string
 1   customer_unique_id        99441 non-null  string
 2   customer_zip_code_prefix  99441 non-null  string
 3   customer_city             99441 non-null  string
 4   customer_state            99441 non-null  string
dtypes: string(5)
memory usage: 3.8 MB
```
Data valid guaranteed:
- IDs → must be unique

Use code below for checking duplicate:
```python
orders.duplicated("order_id").sum()
order_items.duplicated(["order_id","order_item_id"]).sum()
order_payments.duplicated(["order_id","payment_sequential"]).sum()
order_reviews.duplicated("review_id").sum()
products.duplicated("product_id").sum()
customers.duplicated("customer_id").sum()
sellers.duplicated("seller_id").sum()
product_category.duplicated("product_category_name").sum()
```

Result:
Except order_reviews shown duplicate result - np.int64(814), acceptable as review bundle cases.  
The rest shown no duplicate - np.int64(0)

## 4. Feature engineering
- Merged product categories (Portuguese → English)

Use code below:
```python
products2 = products.merge(
    product_category,
    on="product_category_name",
    how="left")
```

- Check increase row after merge

Use code below:  
```python
len(order_items), len(items_cat), len(items_with_date)
```
Result:
```python
(112650, 112650, 110197)
```

- Create order_items + category + date
Merge order_items and products to have category use code below:

Use code below:
```python
items_cat = order_items.merge(
    products2[["product_id", "product_category_name_english"]],
    on="product_id",
    how="left")
```

- Extract purchase date from orders 

Use code below:
```python
orders_clean = orders[["order_id", "order_purchase_timestamp", "order_status"]].copy()
orders_clean = orders_clean.rename(columns={"order_purchase_timestamp": "order_date"})
```

- Filter delivered orders:

Use code below:
```python
orders_clean = orders_clean[orders_clean["order_status"] == "delivered"]
```

- Merge date into items:

Use code below:
```python
items_with_date = items_cat.merge(
    orders_clean[["order_id", "order_date"]],
    on="order_id",
    how="inner")

items_with_date["order_date"] = items_with_date["order_date"].dt.date
items_with_date["order_date"] = pd.to_datetime(items_with_date["order_date"])
```

- Check increase row after merge:

Use code below:
```python
len(order_items), len(items_cat), len(items_with_date)
```

Result:
```python
(112650, 112650, 110197)
```

- Create daily revenue (base for forecast)

Use code below:
```python
items_with_date["revenue"] = items_with_date["price"] + items_with_date["freight_value"]
daily_revenue = (
    items_with_date
    .groupby("order_date")["revenue"]
    .sum()
    .sort_index())
daily_revenue = daily_revenue.asfreq("D").fillna(0)
```

- Check duplicate revenue line

Use code below:

```python
daily_revenue.index.duplicated().sum()
```

Result:
```python
np.int64(0)
```

- Check increase row after merge:

Use code below:
```python
len(order_items), len(items_cat), len(items_with_date)
```

Result:
```python
(112650, 112650, 110197)
```

**Data quality guaranteed for final cleaned datasets:**
- Verified joins did not create many-to-many row explosions by comparing row counts before vs after merges.
- Confirmed the aggregated daily time series has no duplicate dates, no missing revenue, and no negative values after reindexing to a full daily calendar.

## 5. Exploratory Data Analysis (EDA)

**Modeling Approach**
- Aggregated daily/monthly/weekly revenue
- Category Contribution Analysis & Related factors impact (Review score & Delivery time)
- Train/test split: last 60 days as test
- Baseline model: Naive (last value)
- Time-series model: SARIMA (weekly seasonality)
- Evaluation metric: MAPE

5.1 Daily revenue over time

Use code below:
```python
daily_revenue = (
    items_with_date
    .groupby("order_date")["revenue"]
    .sum()
    .asfreq("D")
    .fillna(0))

plt.figure(figsize=(14,5))
plt.plot(daily_revenue.index, daily_revenue.values)
plt.title("Daily Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()
```
- Rolling mean

Use code below:
```python
plt.figure(figsize=(14,5))
plt.plot(daily_revenue.index, daily_revenue.rolling(7).mean(), label="7-day MA")
plt.plot(daily_revenue.index, daily_revenue.rolling(30).mean(), label="30-day MA")
plt.title("Rolling Average Revenue")
plt.legend()
plt.tight_layout()
plt.show()
```
Result:
<img width="1389" height="490" alt="Rolling Average Revenue" src="https://github.com/user-attachments/assets/e42097b1-e89c-4c54-9784-11ac990e47df" />
<img width="1389" height="490" alt="Daily revenue Over Time" src="https://github.com/user-attachments/assets/ced9fc01-f04e-49fb-8adb-9d507ef39612" />

**Assessment:**
- Daily revenue is highly volatile, suggesting demand is not evenly distributed across days.
- There is a significant spike, which may be driven by a special event such as a flash sale or a marketing campaign.
- The 7-day and 30-day moving averages indicate an overall upward trend and then a more stable level afterward.

**Recommendation:**
- Use moving averages (7D/30D) as planning signals for inventory buffer and fulfillment capacity.
- Flag extreme spike days for root-cause analysis (campaign, promotion, operational issues) and treat them separately in forecasting.

5.2 Revenue by month

Use code below:
```python
monthly_revenue = daily_revenue.resample("ME").sum()

plt.figure(figsize=(12,5))
plt.plot(monthly_revenue.index, monthly_revenue.values, marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()
```
Result:
<img width="1189" height="490" alt="Monthly Revenue Trend" src="https://github.com/user-attachments/assets/ec67390e-e8d7-4ff6-ab5d-6222f2d1e410" />

**Assessment:**
- Revenue increases strongly toward year-end (Q4), especially in November–December.
- The months after the peak (e.g., January–February) tend to be lower, consistent with post-holiday demand normalization.
- Overall, the platform shows growth over time with seasonal fluctuations.

**Recommendation:**
- Build a seasonal inventory and marketing plan: ramp up ahead of Q4 and reduce inventory after the peak to avoid dead stock.
- Pre-book fulfillment capacity and logistics resources before major shopping events (e.g., Black Friday, Christmas, YE flash-sales).

5.3 Weekly Seasonality

Use code below:
```python
import seaborn as sns
items_with_date["weekday"] = items_with_date["order_date"].dt.day_name()

weekday_rev = (
    items_with_date
    .groupby("weekday")["revenue"]
    .sum()
    .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))

plt.figure(figsize=(10,5))
sns.barplot(x=weekday_rev.index, y=weekday_rev.values)
plt.title("Revenue by Day of Week")
plt.xlabel("Day")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()
```
Result:
<img width="989" height="490" alt="Revenue by Day of Week" src="https://github.com/user-attachments/assets/6bad57d3-396c-49dd-8623-314d86b5edb3" />

**Assessment:**
- Revenue is generally higher on weekdays, especially early week-day and lower on weekends.
- This suggests customer purchasing behavior is more active during working days.

**Recommendation:**
- Concentrate marketing spend and promotions on peak weekdays to maximize conversion efficiency.
- Align operational staffing (picking/packing/support) with early weekday demand patterns.

5.4 Category Contribution Analysis

Use code below:
```python
cat_revenue = (
    items_with_date
    .groupby("product_category_name_english")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(15))

plt.figure(figsize=(10,6))
sns.barplot(y=cat_revenue.index, x=cat_revenue.values)
plt.title("Top 15 Categories by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Category")
plt.tight_layout()
plt.show()
```

Result:
<img width="990" height="590" alt="Top 15 Categories by Revenue" src="https://github.com/user-attachments/assets/62cc880c-971f-4314-8563-6742d8b7cca5" />

Calculate ad-hoc information of % Top sales contribution total revenue:

Use code below:
```python
total_rev = items_with_date["revenue"].sum()

top5_rev = (
    items_with_date.groupby("product_category_name_english")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .sum())

top5_share = top5_rev / total_rev * 100
print("Top 5 share (%):", top5_share)

total_rev = items_with_date["revenue"].sum()

top15_rev = (
    items_with_date.groupby("product_category_name_english")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .sum())

top15_share = top15_rev / total_rev * 100
print("Top 15 share (%):", top15_share)
```

**Assessment:**
- Revenue is concentrated across a subset of leading categories; the Top 5 categories account for ~39% (Top 15 account for ~76%) of total revenue.
- Many long-tail categories contribute relatively little revenue.

**Recommendation:**
- Prioritize inventory availability and marketing budget for the highest-revenue categories to improve ROI.
- For low-revenue long-tail categories, consider leaner stocking strategies (e.g., limited inventory / make-to-order / reduced assortment).

5.5 Review Score & Delivery Time Impact

Use code below:
```python
orders_delivery = orders.copy()
orders_delivery = orders_delivery[
    orders_delivery["order_delivered_customer_date"].notna()]

orders_delivery["delivery_days"] = (
    (orders_delivery["order_delivered_customer_date"] -
     orders_delivery["order_purchase_timestamp"]).dt.days)

reviews_join = order_reviews.merge(
    orders_delivery[["order_id", "delivery_days"]],
    on="order_id",
    how="inner")

plt.figure(figsize=(8,5))
sns.boxplot(x="review_score", y="delivery_days", data=reviews_join)
plt.title("Delivery Time vs Review Score")
plt.tight_layout()
plt.show()
```
Result:
<img width="790" height="490" alt="Delivery Time vs Reviewe Score" src="https://github.com/user-attachments/assets/8d51490e-dee3-41d0-8fe3-0582c64babd3" />

**Assessment:**
- The boxplot indicates longer delivery time is associated with lower review scores (1–2 star reviews show noticeably higher delivery days).
- Logistics performance is a key driver of customer satisfaction (and potentially conversion).

**Recommendation:**
- Set and monitor delivery SLA (service-level agreement) by seller/region and track breach rates.
- Prioritize faster fulfillment (closer warehouses) for top categories/high-volume items.
- Rank sellers by delivery_days + low review rate.
- Apply improvement actions (process fix, visibility reduction, or penalties depending on policy).

5.6 Train–test split

Use code below:
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

y = daily_revenue.copy()
h = 60
y_train = y.iloc[:-h]
y_test = y.iloc[-h:]
```
5.7 Baseline: Naive forecast

Use code below:
```python
y_naive = pd.Series(index=y_test.index, data=y_train.iloc[-1])
mape_naive = mean_absolute_percentage_error(y_test, y_naive)
print("Naive MAPE:", mape_naive)
```
Result:
```python
Naive MAPE: 0.7014361267624021
```

5.8 SARIMA model

Use code below:
```python
model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,7))
res = model.fit(disp=False)
pred = res.predict(start=y_test.index[0], end=y_test.index[-1])
mape_sarima = mean_absolute_percentage_error(y_test, pred)
print("SARIMA MAPE:", mape_sarima)
```
Result:
```python
SARIMA MAPE: 0.796364798097603
```

5.9 Compare forecast vs actual

Use code below:
```python
plt.figure(figsize=(14,5))
plt.plot(y_train.index, y_train.values, label="Train")
plt.plot(y_test.index, y_test.values, label="Test - Actual")
plt.plot(pred.index, pred.values, label="SARIMA Forecast")
plt.plot(y_naive.index, y_naive.values, label="Naive Forecast", linestyle="--")
plt.legend()
plt.title("Model Comparison")
plt.tight_layout()
plt.show()
```
Result:
<img width="1389" height="490" alt="Model Comparision" src="https://github.com/user-attachments/assets/4a62d106-e389-452a-ab66-27eb33aa4047" />

**Assessment:**
- Results show Naive MAPE is lower than SARIMA → the series is highly volatile, seasonal signal may be weak, or models are affected by outliers/spikes.
- This does not necessarily mean SARIMA is “bad”; it often means baseline is hard to beat without adding features (events/holidays/promos) or tuning.

**Recommendation:**
- Keep Naive as a strong benchmark baseline and explicitly state that “baseline is strong”.
- Keep SARIMA as an “advanced attempt”, but it needs parameter tuning and/or exogenous variables (event/holiday/promo flags) to outperform the baseline.

5.10 Forecast 60 days

Use code below:
```python
model_full = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,7))
res_full = model_full.fit(disp=False)

future_index = pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=60, freq="D")
future_forecast = res_full.predict(start=future_index[0], end=future_index[-1])

plt.figure(figsize=(14,5))
plt.plot(y.index, y.values, label="History")
plt.plot(future_index, future_forecast.values, label="60-Day Forecast")
plt.legend()
```

**Assessment:**
- The 60-day forecast suggests relatively stable expected revenue without major spikes (unless a campaign/event occurs).
- Given volatility and outlier sensitivity, the forecast should be treated as a planning reference rather than an exact target.

**Recommendation:**
- Use the 60-day forecast to set:
  + Base inventory levels for top categories (avoid overstocking).
  + Capacity planning for warehouse and delivery based on baseline demand.
- For upcoming campaigns, use scenario planning (Base / Promo / Peak) instead of relying on a single forecast line.

## 6. Hypotheses
- Q4 revenue increases due to seasonal shopping.
- Longer delivery time reduces review score.
- A few categories generate most revenue (Pareto).
- Customers shop more on weekdays.

## 7. Key Insights
- Revenue is seasonal and weekday-driven
- Category concentration is high
- Delivery performance affects satisfaction
- Forecasting supports planning

## 8. Conclusion

**Inventory**
- Increase stock 10–20% for top categories before Q4
- Reduce stock in low‑season months (Jan–Feb)
- Use make‑to‑order for long‑tail categories
- Prepare inventory before Tue–Wed weekly peaks
  
**Marketing**
- Focus ad spend on Q4 and early week (Mon–Wed)
- Promote top categories during peak windows
- Highlight products with high review scores
  
**Logistics**
- Reduce delivery delays to improve review scores
- Prioritize sellers with faster lead times
- Consider fulfillment centers closer to customer cluster

## 9. Limitations & Future Work

**Limitations:**
- Data timeframe (2016–2018): findings may not reflect current market behavior, pricing, competition, or platform dynamics.
- Revenue proxy: revenue is computed from item prices (and may exclude cancellations, refunds, discounts, and full shipping economics), which may bias absolute levels.
- Event/outlier effects: large spikes likely reflect promotions or anomalies; without explicit event flags, models can mislearn patterns.
- Model scope: SARIMAX was used with limited tuning and without external drivers; results may understate the potential of more feature-rich models.

**Future deep dive analysis:**
- Feature engineering for forecasting: add calendar features (weekday/month), holiday & campaign flags, lag/rolling features; test Prophet or tree-based models (XGBoost/LightGBM) to capture nonlinear patterns.
- Robust evaluation: use rolling-origin / walk-forward validation and compare multiple horizons (7/14/30/60 days).
- Outlier handling: apply winsorization/capping or separate “event vs non-event” modeling to stabilize predictions.

**Future Wide-scope analysis:**
- Seller-level analysis: identify sellers driving long delivery times and low ratings; build seller scorecards (delivery_days, review_score, on-time rate).
- Category-level forecasting: forecast at top-category level and aggregate upward for more actionable inventory planning.
- Inventory simulation: translate forecasts into inventory decisions (safety stock, reorder point, service level) and test policies under peak-season scenarios.
- Customer & cohort insights (optional): segment customers by frequency/value and evaluate retention impact from delivery performance and review outcomes.
- Apply time-series cross-validation to reduce bias from a single train–test split and to better assess model stability across different forecast windows (e.g., 7/14/30/60 days).

## 10. Tools & Libraries
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- prophet (optional for advanced forecasting)
Add these to requirements.txt for reproducibility.

**Contact and attribution
Author - Khai Tran
Contact - khai.trantrong2@gmail.com - https://www.linkedin.com/in/khaitran22297/.**

