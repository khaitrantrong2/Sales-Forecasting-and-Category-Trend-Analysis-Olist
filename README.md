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
  
## How to run the analysis in Colab
1. Upload data
- Open the notebook in Colab.
- Use the upload cell to upload the required CSV files from the dataset.
2. Install dependencies
Use query below to view data:
```code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```
3. Execute notebook cells in order
- Section 1: Setup.
- Section 2: Load and read data.
- Section 3: Data exploration
- Section 4: Feature engineering.
- Section 5: EDA, forecasting (train/test, baseline, SARIMAX), charts, evaluate. Export charts to charts/ for slides.
- Section 6: Recommendations and summary.
- Section 7: Conclusion
- Section 8: Limitations & Future Work.
4. Save outputs
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

