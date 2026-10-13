import pandas as pd 

def get_month_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Conversion rate by month, sorted calendar order."""
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "June",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    result = (
        df.groupby("Month")["Converted"]
        .mean()
        .reset_index()
        .rename(columns={"Converted": "ConversionRate"})
    )
    result["ConversionRate"] = result["ConversionRate"] * 100

    result["Month"] = pd.Categorical(result["Month"], categories=month_order, ordered=True)
    result = result.sort_values("Month").reset_index(drop=True)

    return result 

def get_visitor_type_conversion(df: pd.DataFrame) -> pd.DataFrame: 
    """ Conversion rate by visitor type.""" 
    result = ( 
        df.groupby("VisitorType")["Converted"] 
        .mean() 
        .reset_index() 
        .rename(columns = {"Converted": "ConversionRate"})
    ) 
    result["ConversionRate"] = result["ConversionRate"] * 100 
    return result 
