import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

file_path = "ecommerce_customer_analysis_clean.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# -----------------------------------------
# 2. Prepare the data
# -----------------------------------------

# Calculate total sales
df["Total_Sales"] = df["Product_Price"] * df["Quantity"]

# Convert Order_Date into date format
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

# Create month column
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)


# Create folder to save graphs
os.makedirs("visualizations", exist_ok=True)


# -----------------------------------------
# 3. Product Category vs Total Sales
# -----------------------------------------

category_sales = df.groupby("Product_Category")["Total_Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar")
plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/01_category_sales.png")
plt.show()


# -----------------------------------------
# 4. Orders by City
# -----------------------------------------

city_orders = df["City"].value_counts().head(10)

plt.figure(figsize=(10, 6))
city_orders.plot(kind="bar")
plt.title("Top 10 Cities by Number of Orders")
plt.xlabel("City")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/02_city_orders.png")
plt.show()


# -----------------------------------------
# 5. Customer Age Distribution
# -----------------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df["Customer_Age"].dropna(), bins=10)
plt.title("Customer Age Distribution")
plt.xlabel("Customer Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("visualizations/03_age_distribution.png")
plt.show()


# -----------------------------------------
# 6. Product Price vs Quantity
# -----------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(df["Product_Price"], df["Quantity"], alpha=0.6)
plt.title("Product Price vs Quantity")
plt.xlabel("Product Price")
plt.ylabel("Quantity")
plt.tight_layout()
plt.savefig("visualizations/04_price_vs_quantity.png")
plt.show()


# -----------------------------------------
# 7. Rating Distribution
# -----------------------------------------

plt.figure(figsize=(10, 6))
sns.histplot(df["Rating"].dropna(), bins=10, kde=True)
plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("visualizations/05_rating_distribution.png")
plt.show()


# -----------------------------------------
# 8. Monthly Sales Trend
# -----------------------------------------

monthly_sales = df.groupby("Month")["Total_Sales"].sum()

plt.figure(figsize=(10, 6))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/06_monthly_sales.png")
plt.show()


# -----------------------------------------
# 9. Correlation Heatmap
# -----------------------------------------

numeric_columns = [
    "Customer_Age",
    "Product_Price",
    "Quantity",
    "Rating",
    "Total_Sales"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/07_correlation_heatmap.png")
plt.show()


# -----------------------------------------
# 10. Final message
# -----------------------------------------

print("\nData Visualization completed successfully!")
print("All graphs have been saved inside the 'visualizations' folder.")