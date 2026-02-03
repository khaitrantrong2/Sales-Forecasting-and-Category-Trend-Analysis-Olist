## Sales-Forecasting-and-Category-Trend-Analysis-Olist
A 60‑day forecast, and high‑level inventory and marketing recommendations.
## Dataset and repository structure
Dataset
- Source: Olist Brazilian E‑commerce public dataset (Kaggle).
- Key tables used: orders, order_items, products, order_reviews, order_payments, sellers, geolocation, product_category_name_translation, customers.
Repository layout
- Google collab notebook - main Colab notebook: data prep, EDA, forecasting, recommendations - https://colab.research.google.com/drive/1cm-cTILbDeIh0K9k-TdQh3sh1EGSprYx#scrollTo=o1sxtUIl6KTB
- charts/ - exported PNG charts used in slides (daily_revenue.png; rolling_ma.png; monthly_revenue.png; weekday_revenue.png; top_categories.png; delivery_vs_review.png; forecast_60day.png).
- slides/ - presentation file and exported PDF.
- data/ - raw CSV files.
- README.md - this file.
  
## How to run and execute notebook analysis in Colab

1. Step 1: Set-up and Install dependencies
- Open the notebook in Colab.
- Install dependencies
Use code below:
```code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```
2. Step 2: Load and read data.

- Find dataset - Example: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download
- Use 9 files download from Kaggle and upload the downloaded CSV files from the dataset.
Use code below:
```code
from google.colab import files
uploaded = files.upload()
```
- Read data:
Use code below:
```code
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
3. Data exploration
- To see sample of each source (replicate 9 times for 9 files)
Use code below:
```code
geolocation.head()
```
- To check data type of each source (replicate 9 times for 9 files)
Use code below:
```code
geolocation.info()
```
- Result:

Below field need to convert to string for consistency and prevent errors when join or calculate:
geolocation["geolocation_city"]
geolocation["geolocation_state"] 
geolocation["geolocation_zip_code_prefix"] 
order_items["order_id"] 
order_items["product_id"] 
order_items["seller_id"] 
order_payments["order_id"] 
order_payments["payment_type"] 
order_reviews["review_id"]
order_reviews["order_id"] 
order_reviews["review_comment_title"] 
order_reviews["review_comment_message"] 
order_reviews["review_creation_date"]
orders["order_id"] 
orders["customer_id"] 
orders["order_status"] 
products["product_id"] 
products["product_category_name"] 
sellers["seller_id"] 
sellers["seller_city"] 
sellers["seller_state"]
sellers["seller_zip_code_prefix"]
product_category["product_category_name"] 
product_category["product_category_name_english"] 
customers["customer_id"] 
customers["customer_unique_id"] 
customers["customer_city"] 
customers["customer_state"]
customers["customer_zip_code_prefix"] 

Use code below:
```code
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
Below fields need to convert to date time for date series:
order_items["shipping_limit_date"]
order_reviews["review_answer_timestamp"]
orders["order_purchase_timestamp"]
orders["order_approved_at"]
orders["order_delivered_carrier_date"]
orders["order_delivered_customer_date"]
orders["order_estimated_delivery_date"]

Use code below:
```code
order_items["shipping_limit_date"] = pd.to_datetime(order_items["shipping_limit_date"], errors="coerce")
order_reviews["review_answer_timestamp"] = pd.to_datetime(order_reviews["review_answer_timestamp"], errors="coerce")
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
orders["order_approved_at"] = pd.to_datetime(orders["order_approved_at"], errors="coerce")
orders["order_delivered_carrier_date"] = pd.to_datetime(orders["order_delivered_carrier_date"], errors="coerce")
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
```
Below fields need to convert to numuric for calculation:
order_payments["payment_value"]
order_items["price"]
order_items["freight_value"]

Use code below:
```code
order_payments["payment_value"] = pd.to_numeric(order_payments["payment_value"], errors="coerce")
order_items["price"] = pd.to_numeric(order_items["price"], errors="coerce")
order_items["freight_value"] = pd.to_numeric(order_items["freight_value"], errors="coerce")
```
Result:

1. geolocation table
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

2. order_items table
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

3. order_payments table
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

4. Order_review
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

5.Order table

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

6.Product table

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

7. Seller table

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

8. Product_category table

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 71 entries, 0 to 70
Data columns (total 2 columns):
 #   Column                         Non-Null Count  Dtype 
---  ------                         --------------  ----- 
 0   product_category_name          71 non-null     string
 1   product_category_name_english  71 non-null     string
dtypes: string(2)
memory usage: 1.2 KB

9. Customer table

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

4. Step 4: Feature engineering.
- Join product category name

- Create order_items + category + date
Merge order_items and products to have category use code below:
```code
items_cat = order_items.merge(
    products2[["product_id", "product_category_name_english"]],
    on="product_id",
    how="left")
```
Extract purchase date from orders use code below:
```code
orders_clean = orders[["order_id", "order_purchase_timestamp", "order_status"]].copy()
orders_clean = orders_clean.rename(columns={"order_purchase_timestamp": "order_date"})
```
Filter delivered orders use code below:
```code
orders_clean = orders_clean[orders_clean["order_status"] == "delivered"]
```
Merge date into items use code below:
```code
items_with_date = items_cat.merge(
    orders_clean[["order_id", "order_date"]],
    on="order_id",
    how="inner")

items_with_date["order_date"] = items_with_date["order_date"].dt.date
items_with_date["order_date"] = pd.to_datetime(items_with_date["order_date"])
```
- Create daily revenue (base for forecast)
Use code below:
```code
items_with_date["revenue"] = items_with_date["price"] + items_with_date["freight_value"]
daily_revenue = (
    items_with_date
    .groupby("order_date")["revenue"]
    .sum()
    .sort_index())
daily_revenue = daily_revenue.asfreq("D").fillna(0)
```
 
5. Step 5: EDA, forecasting (train/test, baseline, SARIMAX), charts, evaluate. Export charts to charts/ for slides.
- Daily revenue over time
Use code below:
```code
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
```code
plt.figure(figsize=(14,5))
plt.plot(daily_revenue.index, daily_revenue.rolling(7).mean(), label="7-day MA")
plt.plot(daily_revenue.index, daily_revenue.rolling(30).mean(), label="30-day MA")
plt.title("Rolling Average Revenue")
plt.legend()
plt.tight_layout()
plt.show()
```
Result:
<img width="1389" height="490" alt="Daily revenue Over Time and Rolling Average Revenue" src="https://github.com/user-attachments/assets/0dc7a18d-6e1c-4952-bd35-57d92c7deb22" />

Assessment:

Doanh thu theo ngày biến động mạnh, thể hiện hành vi mua hàng không đều.
Xu hướng tổng thể tăng nhẹ theo thời gian, thể hiện tăng trưởng tự nhiên của nền tảng.
Một số spike lớn có thể liên quan đến flash sale hoặc chiến dịch marketing.
Rolling 7-day và 30-day cho thấy xu hướng tăng ổn định, không có dấu hiệu suy giảm dài hạn.

7. Step 6: Recommendations and summary.

8. Step 7: Conclusion


9. Step 8: Limitations & Future Work.

#Save outputs
- Export charts as PNG using plt.savefig("charts/filename.png", dpi=150, bbox_inches="tight").
- Export the final notebook as PDF.

Deliverables
1. Notebook: Full Colab notebook with code and narrative
2. Charts: Daily revenue, rolling MA, monthly trend, weekday pattern, top categories, delivery vs review, 60‑day forecast
3. Presentation slide with forecast + insight + recommendation, ready for export to PDF
4. Final report summarizing purpose, methodologies, findings and recommendations
5. Detailed explanations in Readme.md

Results summary and recommended actions
Findings
- Platform revenue shows a mild upward trend with high daily volatility.
- Strong seasonal peak in Q4 (Nov–Dec).
- Weekly peak on Tuesday and Wednesday.
- Top 5 categories contribute the majority of revenue; long tail contributes little.
- Longer delivery times correlate with lower review scores.
High‑level recommendations
- Increase stock for top categories by 10–20% ahead of Q4.
- Use make‑to‑order or minimal stock for long‑tail categories.
- Align replenishment to prepare inventory before Tue–Wed peaks.
- Prioritize fulfillment from warehouses closer to demand centers to reduce lead time.
- Focus marketing spend in Q4 and early week to maximize conversion.

Notes, limitations, and next steps
Limitations
- Dataset covers historical years (2016–2018); results are historical and may not reflect current market conditions.
- SARIMAX is used as a baseline; more advanced models (Prophet, XGBoost, LightGBM) may improve accuracy.
- Inventory recommendations are high level and do not include safety stock or reorder point calculations.
Next steps
- Refit models at category level and compare Prophet/ML models.
- Build a lightweight dashboard to monitor daily actuals vs forecast and seller lead times.
- Add seller‑level SLA monitoring and alerts to reduce delivery delays.

Requirements
Minimum packages
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- prophet (optional for advanced forecasting)
Add these to requirements.txt for reproducibility.

Contact and attribution
Author - Khai Tran
Contact - khai.trantrong2@gmail.com - https://www.linkedin.com/in/khaitran22297/.

