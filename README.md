# Store_Sales_Project: A Deep Dive into Retail Analytics

In our project focused on forecasting store sales for Rossmann, our goal was to develop an accurate sales prediction model that could optimize inventory management and enhance customer satisfaction across all stores. The project aimed to improve efficiency and effectiveness within the retail chain through precise and actionable forecasts.

Key Steps in the Project

1.🔍 Problem Identification:

•Rossmann’s challenge was to optimize inventory management by accurately forecasting sales, which would reduce excess stock and minimize stock shortages across its stores.

2.🛠️ Feature Engineering:

•We introduced new features such as lagged sales data, rolling averages, and interaction terms between promotions and holidays to enhance the prediction accuracy.

Key Steps in the Project

🔍 Problem Identification:
• Rossmann faced the challenge of optimizing inventory management by accurately forecasting sales. This approach aimed to reduce excess stock and mitigate stock shortages across its extensive network of stores.

🛠️ Feature Engineering:
• To improve prediction accuracy, we introduced new features, such as lagged sales data, rolling averages, and interaction terms between promotions and holidays.

• Key Features Added:

📅 Temporal Variables: Day, Month, Year, IsWeekend, NEW_Day,NEW_Month, NEW_Year etc.

⏳ Lagged Features: Sales values lagged by 1, 7, and 30 days.

🔄 Rolling Features: 7 and 30-day rolling means, sums, and standard deviations of sales.

📈 Exponential Moving Averages: For a more responsive trend analysis.

3. 📍 Modeling:
   
• We tested various models including ARIMA, SARIMAX, and LGBM. The LGBM model outperformed the others by capturing complex relationships and trends in the data, providing the most accurate forecasts.

5. 📊 Analysis and Results:
   
• The LGBM model proved exceptional in predicting sales, offering crucial insights into future sales trends that are essential for strategic planning.

• Key Insights:

📉 Sales Trends: Showed significant variation across different store types and were influenced by seasonal factors like the Christmas season.

📆 Interaction Terms: Promotions during holidays had a marked impact on sales, effectively captured by our model.

5. 📈 Visualization and Presentation:
• We utilized Power BI to craft interactive dashboards, delivering a clear and dynamic presentation of our findings and model predictions. The visualizations, which are included in the project, facilitate a deeper understanding of temporal sales patterns and the effectiveness of our forecasting model.
