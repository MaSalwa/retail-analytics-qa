# Data

## File

- `online_retail_II.xlsx`
- Use sheet `Year 2009-2010` for this project.

## Load

```python
import pandas as pd

df = pd.read_excel("data/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df.head()
```

Columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.

## Source

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Also discussed in [this Kaggle CRM notebook](https://www.kaggle.com/code/nihandincer/customer-relationship-management-crm/notebook).

Cancellations are typically invoices whose id starts with `C`. `Customer ID` and `Description` have missing values; we will decide how to treat them in `notebooks/01_explore_and_clean.ipynb`.
