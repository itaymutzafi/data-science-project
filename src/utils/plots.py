import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import pandas as pd
import src.config as config

def date_groupby_line_plot(df, yname, title):
    per_day = df.groupby(df["date"].dt.date).size()

    per_day.plot(kind="line")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(yname)
    plt.grid(True)
    plt.show()

def articles_over_time_by_dataset_plot(df, is_log, specific_years):
    grouped = (
        df
        .set_index("date")
        .groupby("dataset")
        .resample("ME", include_groups=False)
        .size()
        .unstack(level=0)
    )

    plt.figure()

    for column in grouped.columns:
        plt.plot(grouped.index, grouped[column], label=column)

    plt.xlabel("Time")
    if is_log:
        plt.ylabel("Log Number of Articles")
        plt.yscale("log")
    else:
        plt.ylabel("Number of Articles")

    title = "Articles Over Time by Dataset"
    if specific_years is not None:
        title += f" {specific_years}"
    plt.title(title)
    
    plt.legend()
    plt.show()

def article_volume_per_company_plot(df):
    for company, group in df.groupby("company"):
        monthly_counts = (
            group.set_index("date")
                .resample("ME")
                .size()
        )
        
        plt.plot(
            monthly_counts.index, 
            monthly_counts.values, 
            label=company
        )

    plt.xlabel("Time")
    plt.ylabel("Number of Articles")
    plt.title("Monthly News Volume per Company")
    plt.legend(title="Company")
    plt.tight_layout()
    plt.show()

def pie_plot(counts, subject):
    plt.figure()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title(f"Distribution of {subject}")
    plt.show()

def table_visualize(df, groupby):
    return (
        df
        .groupby(groupby)
        .size()
        .unstack(fill_value=0)
    )


    

    