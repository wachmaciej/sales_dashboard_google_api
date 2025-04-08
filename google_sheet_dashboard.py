# Chunk 1 of 3

import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import datetime
import calendar
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import io
import gspread               
from google.oauth2.service_account import Credentials 
import os                    

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="YOY Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# --- Google Sheet Configuration ---
# <<< EDIT HERE >>>
# Replace with your actual Google Sheet file name
GOOGLE_SHEET_NAME = "RA_sales_dashboard_data"
# Replace with the exact name of the worksheet (tab) within your Google Sheet
WORKSHEET_NAME = "SAP_DATA"
# Path to your service account key file for local development.
# Change if the file is not in the same directory as the script.
LOCAL_KEY_PATH = 'service_account.json'
# <<< END EDIT HERE >>>

# --- Define Google API Scopes ---
# These define the permissions the script requests
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file' # Often needed for discovery/listing
]

# --- Title and Logo ---
# This remains as you provided
col1, col2 = st.columns([3, 1])
with col1:
    st.title("YOY Dashboard 📊")
with col2:
    # Check if logo file exists before trying to display it
    if os.path.exists("logo.png"):
        st.image("logo.png", width=300)
    else:
        st.write(" ") # Placeholder if logo not found

# --- Sidebar ---
# Session state initialization is removed as data is loaded directly now
st.sidebar.header("YOY Dashboard")
# We will add info about the connected sheet later in the main section

# =============================================================================
# NEW Function to Load Data from Google Sheets (Corrected Error Handling)
# =============================================================================
@st.cache_data(ttl=600, show_spinner="Fetching data from Google Sheet...") # Cache data for 10 minutes
def load_data_from_gsheet():
    """Loads data from the specified Google Sheet worksheet, handling missing secrets file."""
    creds = None
    secrets_used = False # Flag to check if secrets were successfully used

    # --- Authentication ---
    # Try using Streamlit secrets first
    try:
        # Check if the specific key exists in secrets IF the secrets file itself can be accessed
        if "gcp_service_account" in st.secrets:
            creds_json_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_json_dict, scopes=SCOPES)
            secrets_used = True
            # st.info("Using credentials from Streamlit secrets.") # Optional feedback
        # else: # This would run if secrets file exists but key is missing
             # st.info("Secrets file found, but 'gcp_service_account' key missing. Trying local file.")
            # pass # Proceed to local file check below

    except FileNotFoundError:
        # This specifically catches the error if secrets.toml doesn't exist
        #st.info("No Streamlit secrets file found. Trying local key file...")
        pass # Proceed to local file check below
    except Exception as e_secrets:
        # Catch other potential errors during secrets processing
        #st.warning(f"Error processing Streamlit secrets: {e_secrets}. Trying local key file.")
        pass # Proceed to local file check below


    # Fallback to local JSON file if secrets weren't successfully used
    if not secrets_used:
        if os.path.exists(LOCAL_KEY_PATH):
            try:
                 creds = Credentials.from_service_account_file(LOCAL_KEY_PATH, scopes=SCOPES)
                 #st.info(f"Using credentials from local file: '{LOCAL_KEY_PATH}'.") # Feedback
            except Exception as e_local:
                 # Handle errors loading from the local file specifically
                 st.error(f"Error loading credentials from local file '{LOCAL_KEY_PATH}': {e_local}")
                 st.stop() # Stop if local file exists but can't be read
        else:
            # This error occurs if secrets failed AND local file doesn't exist
            st.error(f"Authentication Error: Neither Streamlit Secrets nor local key file '{LOCAL_KEY_PATH}' were found or configured correctly.")
            st.stop() # Stop execution if no credentials found

    # Final check if credentials object was successfully created by either method
    if not creds:
         st.error("Authentication failed. Could not load credentials object.")
         st.stop()

    # --- Authorize and Open Sheet (Rest of the function remains the same) ---
    try:
        client = gspread.authorize(creds)
        try:
            spreadsheet = client.open_by_key('1p9MkE7pSF5WiZjPJCFrGwKAHQf3aZglzUKjMHMDprlo')
        except gspread.exceptions.SpreadsheetNotFound:
             st.error(f"Error: Google Sheet '{GOOGLE_SHEET_NAME}' not found. Please check the name and ensure it's shared with the service account email: {creds.service_account_email}")
             st.stop()

        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
             st.error(f"Error: Worksheet '{WORKSHEET_NAME}' not found in the spreadsheet '{GOOGLE_SHEET_NAME}'. Please check the worksheet name (tab name).")
             st.stop()

        #st.success(f"Connected to Google Sheet: '{GOOGLE_SHEET_NAME}', Worksheet: '{WORKSHEET_NAME}'")

        # Read Data into Pandas DataFrame
        data = worksheet.get_all_values()
        if not data or len(data) < 2:
            st.warning(f"No data found in worksheet '{WORKSHEET_NAME}' or only headers present.")
            return pd.DataFrame()

        headers = data.pop(0)
        df = pd.DataFrame(data, columns=headers)
        #st.info(f"Read {len(df)} rows from Google Sheet (before type conversion).")

        # --- Data Type Conversion section (as you edited it previously) ---
        # <<< Ensure your edited numeric_cols and date_cols lists are here >>>
        numeric_cols = [
            "Revenue", "Week", "Order Quantity", "Sales Value in Transaction Currency",
            "Sales Value (£)", "Year"
            # Add/Remove based on your actual columns
        ]
        date_cols = ["Date"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[£,]', '', regex=True).str.strip()
                df[col] = df[col].replace('', pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Optional warning moved inside the loop for clarity
                # if df[col].isnull().any() and not df[col].isna().all():
                #    st.warning(f"Column '{col}' contains non-numeric values that were ignored.")
            # else: st.warning(f"Numeric column '{col}' specified for conversion not found.")

        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].replace('', pd.NaT)
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                # Optional warning moved inside the loop
                # if df[col].isnull().any() and not df[col].isna().all():
                #    st.warning(f"Column '{col}' contains non-date values that were ignored.")
            # else: st.warning(f"Date column '{col}' specified for conversion not found.")

        df = df.replace('', None)
        #st.success("Data successfully loaded and initial types converted from Google Sheet.")
        return df

    # Catch specific gspread/auth errors or general exceptions during sheet access/reading
    except gspread.exceptions.APIError as e_api:
         st.error(f"Google Sheets API Error: {e_api}")
         #st.info("This might be due to insufficient permissions for the service account on the Sheet (needs 'Viewer') or API not enabled correctly.")
         st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while accessing Google Sheets after authentication: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

# =============================================================================
# Helper Functions for Sales Data (Your Original Functions)
# =============================================================================
# These functions remain exactly as you provided them.

def compute_custom_week(date):
    """Computes the custom week number, year, start, and end dates."""
    # Ensure input is a date object
    if not isinstance(date, datetime.date):
        # Handle cases where input might be NaT or other non-date types
        return None, None, None, None
    try:
        custom_dow = (date.weekday() + 2) % 7  # Saturday=0, Sunday=1, ..., Friday=6
        week_start = date - datetime.timedelta(days=custom_dow)
        week_end = week_start + datetime.timedelta(days=6)
        custom_year = week_end.year # Year is defined by the week's end date

        # Calculate the start of the first week of the custom_year
        first_day_of_year = datetime.date(custom_year, 1, 1)
        first_day_of_year_custom_dow = (first_day_of_year.weekday() + 2) % 7
        first_week_start_of_year = first_day_of_year - datetime.timedelta(days=first_day_of_year_custom_dow)

        # Handle edge case where the week_start might fall in the *previous* calendar year
        # but belongs to the first week of the custom_year if first_week_start_of_year is also in the previous year.
        if week_start < first_week_start_of_year:
             # This week likely belongs to the previous year's week numbering
             custom_year -= 1
             first_day_of_year = datetime.date(custom_year, 1, 1)
             first_day_of_year_custom_dow = (first_day_of_year.weekday() + 2) % 7
             first_week_start_of_year = first_day_of_year - datetime.timedelta(days=first_day_of_year_custom_dow)


        # Calculate the custom week number
        custom_week = ((week_start - first_week_start_of_year).days // 7) + 1

        # Sanity check for week 53 potentially belonging to the next year if Jan 1st starts mid-week
        # This logic might need refinement depending on specific ISO week definitions if required
        if custom_week == 53:
            last_day_of_year = datetime.date(custom_year, 12, 31)
            last_day_custom_dow = (last_day_of_year.weekday() + 2) % 7
            if last_day_custom_dow < 3: # If year ends on Sat, Sun, Mon (custom dow 0, 1, 2) - week 53 might belong to next year
                 # Check if Jan 1st of next year falls within this week
                 next_year_first_day = datetime.date(custom_year + 1, 1, 1)
                 if week_start <= next_year_first_day <= week_end:
                      # It's arguably week 1 of the next year
                      # This specific handling depends on exact business rules for week 53/1 transition
                      # For simplicity here, we keep it as week 53 of custom_year
                      pass # Or potentially adjust to year+1, week 1

        return custom_week, custom_year, week_start, week_end
    except Exception as e:
        # Log or handle error appropriately
        st.error(f"Error computing custom week for date {date}: {e}")
        return None, None, None, None


def get_custom_week_date_range(week_year, week_number):
    """Gets the start and end date for a given custom week year and number."""
    try:
        week_year = int(week_year)
        week_number = int(week_number)
        # Calculate the start of the first week of the week_year
        first_day = datetime.date(week_year, 1, 1)
        first_day_custom_dow = (first_day.weekday() + 2) % 7
        first_week_start = first_day - datetime.timedelta(days=first_day_custom_dow)
        # Calculate the start date of the requested week
        week_start = first_week_start + datetime.timedelta(weeks=week_number - 1)
        week_end = week_start + datetime.timedelta(days=6)
        return week_start, week_end
    except (ValueError, TypeError) as e:
         # Handle cases where year/week number are invalid
         # st.warning(f"Invalid input for get_custom_week_date_range: Year={week_year}, Week={week_number}. Error: {e}")
         return None, None


def get_quarter(week):
    """Determines the quarter based on the custom week number."""
    if pd.isna(week): return None
    try:
        week = int(week)
        if 1 <= week <= 13:
            return "Q1"
        elif 14 <= week <= 26:
            return "Q2"
        elif 27 <= week <= 39:
            return "Q3"
        elif 40 <= week <= 53: # Assuming up to 53 weeks possible
            return "Q4"
        else:
            return None
    except (ValueError, TypeError):
        return None # Handle non-integer weeks


def format_currency(value):
    """Formats a numeric value as currency."""
    if pd.isna(value): return "£N/A" # Handle NaN values
    try:
        return f"£{float(value):,.2f}"
    except (ValueError, TypeError):
        return "£Error" # Handle non-numeric input


def format_currency_int(value):
    """Formats a numeric value as integer currency."""
    if pd.isna(value): return "£N/A" # Handle NaN values
    try:
        return f"£{int(round(float(value))):,}"
    except (ValueError, TypeError):
        return "£Error" # Handle non-numeric input


# --- OLD load_data function is REMOVED ---
# @st.cache_data(show_spinner=False)
# def load_data(file): ...


@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(data):
    """Preprocesses the loaded data: converts types, calculates custom weeks/quarters."""
    df = data.copy() # Work on a copy to avoid modifying cached data
    #st.info("Starting data preprocessing...")

    # --- Initial Data Validation ---
    # Check for essential columns needed *before* processing
    required_input_cols = {"Date", "Sales Value (£)"} # Adjust if other raw cols are essential
    if not required_input_cols.issubset(set(df.columns)):
        missing_cols = required_input_cols.difference(set(df.columns))
        st.error(f"Input data is missing essential columns required for preprocessing: {missing_cols}")
        st.stop()

    # --- Type Conversion and Cleaning ---
    # Ensure 'Date' is datetime (should be handled by load_data_from_gsheet, but double-check)
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=["Date"], inplace=True)
    if len(df) < initial_rows:
        st.warning(f"Removed {initial_rows - len(df)} rows due to invalid 'Date' values.")

    # Ensure 'Sales Value (£)' is numeric (should be handled by load_data_from_gsheet)
    if "Sales Value (£)" in df.columns:
        df["Sales Value (£)"] = pd.to_numeric(df["Sales Value (£)"], errors='coerce')
        initial_rows_sales = len(df)
        df.dropna(subset=["Sales Value (£)"], inplace=True)
        if len(df) < initial_rows_sales:
            st.warning(f"Removed {initial_rows_sales - len(df)} rows due to invalid 'Sales Value (£)' values.")
    else:
        st.error("Essential column 'Sales Value (£)' not found.")
        st.stop()

    if df.empty:
        st.error("No valid data remaining after initial cleaning.")
        st.stop()

    # --- Feature Engineering ---
    # Calculate custom week details
    # The apply function calls compute_custom_week which expects a date object
    try:
        week_results = df["Date"].apply(lambda d: compute_custom_week(d.date()) if pd.notnull(d) else (None, None, None, None))
        df["Custom_Week"], df["Custom_Week_Year"], df["Custom_Week_Start"], df["Custom_Week_End"] = zip(*week_results)
    except Exception as e:
        st.error(f"Error calculating custom week details: {e}")
        st.stop()

    # Assign 'Week' and 'Quarter' based on calculated custom week
    df["Week"] = df["Custom_Week"] # Assign Custom_Week to Week
    df["Quarter"] = df["Week"].apply(get_quarter)

    # Convert calculated columns to appropriate integer types (allowing NAs)
    df["Week"] = pd.to_numeric(df["Week"], errors='coerce').astype('Int64')
    df["Custom_Week_Year"] = pd.to_numeric(df["Custom_Week_Year"], errors='coerce').astype('Int64')

    # --- Final Validation ---
    # Drop rows where crucial calculated fields are missing
    initial_rows_final = len(df)
    df.dropna(subset=["Week", "Custom_Week_Year", "Quarter"], inplace=True)
    if len(df) < initial_rows_final:
         st.warning(f"Removed {initial_rows_final - len(df)} rows due to missing calculated week/year/quarter.")

    # Check if essential columns for the dashboard exist AFTER processing
    required_output_cols = {"Week", "Custom_Week_Year", "Sales Value (£)", "Date", "Quarter"} # Add others like Listing, Product, etc.
    if not required_output_cols.issubset(set(df.columns)):
         missing_cols = required_output_cols.difference(set(df.columns))
         st.error(f"Preprocessing did not produce all required output columns: {missing_cols}")
         # Potentially stop execution or return None/empty df depending on desired behavior
         st.stop()

    if df.empty:
        st.error("No data remaining after full preprocessing.")
        st.stop()

    #st.success(f"Preprocessing complete. {len(df)} rows ready for analysis.")
    return df


# --- Charting and Table Functions (Your Original Code) ---
# These functions remain exactly as you provided them.

def create_yoy_trends_chart(data, selected_years, selected_quarters,
                            selected_channels=None, selected_listings=None,
                            selected_products=None, time_grouping="Week"):
    """Creates the YOY Trends line chart."""
    filtered = data.copy()
    # Apply filters
    if selected_years:
        # Ensure selected_years are compared as integers if Custom_Week_Year is Int64
        filtered = filtered[filtered["Custom_Week_Year"].isin([int(y) for y in selected_years])]
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
         if "Sales Channel" in filtered.columns:
            filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
         else: st.warning("Column 'Sales Channel' not found for filtering.")
    if selected_listings and len(selected_listings) > 0:
        if "Listing" in filtered.columns:
            filtered = filtered[filtered["Listing"].isin(selected_listings)]
        else: st.warning("Column 'Listing' not found for filtering.")
    if selected_products and len(selected_products) > 0:
        if "Product" in filtered.columns:
            filtered = filtered[filtered["Product"].isin(selected_products)]
        else: st.warning("Column 'Product' not found for filtering.")

    if filtered.empty:
        st.warning("No data available for YOY Trends chart with selected filters.")
        return go.Figure() # Return empty figure

    # Group data based on time_grouping
    if time_grouping == "Week":
        grouped = filtered.groupby(["Custom_Week_Year", "Week"])["Sales Value (£)"].sum().reset_index()
        x_col = "Week"
        x_axis_label = "Week"
        grouped = grouped.sort_values(by=["Custom_Week_Year", "Week"])
        title = "Weekly Revenue Trends by Custom Week Year"
    else: # Assume Quarter
        grouped = filtered.groupby(["Custom_Week_Year", "Quarter"])["Sales Value (£)"].sum().reset_index()
        x_col = "Quarter"
        x_axis_label = "Quarter"
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]
        grouped["Quarter"] = pd.Categorical(grouped["Quarter"], categories=quarter_order, ordered=True)
        grouped = grouped.sort_values(by=["Custom_Week_Year", "Quarter"])
        title = "Quarterly Revenue Trends by Custom Week Year"

    if grouped.empty:
        st.warning("No data available after grouping for YOY Trends chart.")
        return go.Figure()

    grouped["RevenueK"] = grouped["Sales Value (£)"] / 1000
    fig = px.line(grouped, x=x_col, y="Sales Value (£)", color="Custom_Week_Year", markers=True,
                  title=title,
                  labels={"Sales Value (£)": "Revenue (£)", x_col: x_axis_label, "Custom_Week_Year": "Year"},
                  custom_data=["RevenueK"])
    fig.update_traces(hovertemplate=f"{x_axis_label}: %{{x}}<br>Revenue: %{{customdata[0]:.1f}}K<extra></extra>") # Added <extra></extra> to remove trace info

    # Adjust x-axis range for weekly view
    if time_grouping == "Week":
        min_week = 0.8
        # Use pd.Series.max() which handles potential empty series after filtering/grouping
        max_week = grouped["Week"].max() if not grouped["Week"].empty else 52
        if pd.isna(max_week): max_week = 52 # Handle case where max week is NaN
        fig.update_xaxes(range=[min_week, max_week + 0.2], dtick=5) # Add padding and set ticks

    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50), legend_title_text='Year')
    return fig


def create_pivot_table(data, selected_years, selected_quarters, selected_channels,
                       selected_listings, selected_products, grouping_key="Listing"):
    """Creates the pivot table."""
    filtered = data.copy()
    # Apply filters (similar logic as in create_yoy_trends_chart)
    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin([int(y) for y in selected_years])]
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
         if "Sales Channel" in filtered.columns:
            filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
         else: st.warning("Column 'Sales Channel' not found for filtering pivot table.")
    if selected_listings and len(selected_listings) > 0:
        if "Listing" in filtered.columns:
            filtered = filtered[filtered["Listing"].isin(selected_listings)]
        else: st.warning("Column 'Listing' not found for filtering pivot table.")
    # Apply product filter only if grouping by Product
    if grouping_key == "Product" and selected_products and len(selected_products) > 0:
        if "Product" in filtered.columns:
            filtered = filtered[filtered["Product"].isin(selected_products)]
        else: st.warning("Column 'Product' not found for filtering pivot table.")

    if filtered.empty:
         st.warning("No data available for Pivot Table with selected filters.")
         return pd.DataFrame({grouping_key: ["No data"]}) # Return informative empty DataFrame

    # Check if grouping_key and Week columns exist
    if grouping_key not in filtered.columns or "Week" not in filtered.columns:
        st.error(f"Required columns ('{grouping_key}', 'Week') not found for creating pivot table.")
        return pd.DataFrame({grouping_key: ["Missing columns"]})


    pivot = pd.pivot_table(filtered, values="Sales Value (£)", index=grouping_key,
                           columns="Week", aggfunc="sum", fill_value=0)

    if pivot.empty:
        st.warning("Pivot table is empty after grouping.")
        return pd.DataFrame({grouping_key: ["No results"]})

    pivot["Total Revenue"] = pivot.sum(axis=1)
    pivot = pivot.round(0).astype(int) # Convert to integer after rounding

    # Rename columns to 'Week X'
    new_columns = {}
    for col in pivot.columns:
        if isinstance(col, (int, float)) and col != "Total Revenue":
             new_columns[col] = f"Week {int(col)}"
        elif col == "Total Revenue":
            new_columns[col] = "Total Revenue" # Keep Total Revenue column name
        # Handle potential non-numeric column names if any exist
        # else:
        #     new_columns[col] = str(col)
    pivot = pivot.rename(columns=new_columns)

    # Sort columns: Week 1, Week 2, ..., Total Revenue
    week_cols = sorted([col for col in pivot.columns if col.startswith("Week ")], key=lambda x: int(x.split()[1]))
    if "Total Revenue" in pivot.columns:
        pivot = pivot[week_cols + ["Total Revenue"]]
    else:
         pivot = pivot[week_cols]

    return pivot


def create_sku_line_chart(data, sku_text, selected_years, selected_channels=None, week_range=None):
    """Creates the SKU Trends line chart."""
    # Ensure required columns exist
    required_cols = {"Product SKU", "Custom_Week_Year", "Week", "Sales Value (£)", "Order Quantity"}
    if not required_cols.issubset(data.columns):
        missing = required_cols.difference(data.columns)
        st.error(f"Dataset is missing required columns for SKU chart: {missing}")
        st.stop() # Or return empty figure

    filtered = data.copy()
    # Filter by SKU text (case-insensitive partial match)
    filtered = filtered[filtered["Product SKU"].str.contains(sku_text, case=False, na=False)]

    # Apply other filters
    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin([int(y) for y in selected_years])]
    if selected_channels and len(selected_channels) > 0:
         if "Sales Channel" in filtered.columns:
            filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
         else: st.warning("Column 'Sales Channel' not found for filtering SKU chart.")

    if week_range:
        # Ensure week_range values are valid
        start_week, end_week = week_range
        filtered = filtered[(filtered["Week"] >= start_week) & (filtered["Week"] <= end_week)]

    if filtered.empty:
        st.warning("No data available for the entered SKU and filters.")
        # Returning None might cause issues later if not handled; return empty figure instead
        return go.Figure().update_layout(title_text=f"No data for SKU: '{sku_text}'")

    # Group data
    weekly_sku = filtered.groupby(["Custom_Week_Year", "Week"]).agg({
        "Sales Value (£)": "sum",
        "Order Quantity": "sum"
    }).reset_index().sort_values(by=["Custom_Week_Year", "Week"])

    if weekly_sku.empty:
         st.warning("No data after grouping for SKU chart.")
         return go.Figure().update_layout(title_text=f"No data for SKU: '{sku_text}' after grouping")

    # Prepare data for plotting
    weekly_sku["RevenueK"] = weekly_sku["Sales Value (£)"] / 1000

    # Determine plot range for x-axis
    if week_range:
        min_week_plot, max_week_plot = week_range
    else:
        # Use actual min/max weeks from the filtered data if no range selected
        min_week_plot = weekly_sku["Week"].min() if not weekly_sku["Week"].empty else 1
        max_week_plot = weekly_sku["Week"].max() if not weekly_sku["Week"].empty else 52
        if pd.isna(min_week_plot): min_week_plot = 1
        if pd.isna(max_week_plot): max_week_plot = 52


    fig = px.line(weekly_sku, x="Week", y="Sales Value (£)", color="Custom_Week_Year", markers=True,
                  title=f"Weekly Revenue Trends for SKU matching: '{sku_text}'",
                  labels={"Sales Value (£)": "Revenue (£)", "Custom_Week_Year": "Year"},
                  custom_data=["RevenueK", "Order Quantity"])

    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K<br>Units Sold: %{customdata[1]}<extra></extra>")
    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            range=[max(0.8, min_week_plot - 0.2), max_week_plot + 0.2], # Ensure range starts near 1
            dtick=5 if (max_week_plot - min_week_plot) > 10 else 1 # Adjust tick frequency
            ),
        yaxis=dict(rangemode="tozero"),
        margin=dict(t=50, b=50),
        legend_title_text='Year'
        )
    return fig


def create_daily_price_chart(data, listing, selected_years, selected_quarters, selected_channels, week_range=None):
    """Creates the Daily Average Price line chart."""
    # Ensure required columns exist
    # Note: Original code uses 'Year' column - check if this exists or use 'Custom_Week_Year'
    # Assuming 'Year' needs to be derived or exists. If using Custom_Week_Year, adjust filtering.
    # Let's add 'Custom_Week_Year' and derive 'Year' for consistency if needed.
    if 'Year' not in data.columns and 'Custom_Week_Year' in data.columns:
         data['Year'] = data['Custom_Week_Year'] # Or data['Date'].dt.year if more appropriate

    required_cols = {"Date", "Listing", "Year", "Sales Value in Transaction Currency", "Order Quantity", "Custom_Week"}
    if not required_cols.issubset(data.columns):
         missing = required_cols.difference(data.columns)
         st.error(f"Dataset is missing required columns for Daily Price chart: {missing}")
         # Consider stopping or returning an empty figure
         return go.Figure().update_layout(title_text=f"Missing data for Daily Prices: {missing}")


    # Filter data for the specific listing and selected years
    # Ensure comparison uses appropriate types (e.g., int for years)
    selected_years_int = [int(y) for y in selected_years]
    df_listing = data[(data["Listing"] == listing) & (data["Year"].isin(selected_years_int))].copy()

    # Apply additional filters
    if selected_quarters:
        if "Quarter" not in df_listing.columns:
            st.warning("Column 'Quarter' not found for Daily Price filtering.")
        else:
            df_listing = df_listing[df_listing["Quarter"].isin(selected_quarters)]

    if selected_channels and len(selected_channels) > 0:
        if "Sales Channel" not in df_listing.columns:
            st.warning("Column 'Sales Channel' not found for Daily Price filtering.")
        else:
            df_listing = df_listing[df_listing["Sales Channel"].isin(selected_channels)]

    # Filter by week range if provided
    if week_range:
        start_week, end_week = week_range
        if "Custom_Week" not in df_listing.columns:
             st.warning("Column 'Custom_Week' not found for Daily Price week range filtering.")
        else:
            df_listing = df_listing[(df_listing["Custom_Week"] >= start_week) & (df_listing["Custom_Week"] <= end_week)]

    if df_listing.empty:
        st.warning(f"No data available for '{listing}' with the selected filters.")
        return None # Or return empty figure: go.Figure().update_layout(title_text=f"No data for {listing}")

    # Determine display currency (handle potential missing column)
    display_currency = "GBP" # Default
    if "Original Currency" in df_listing.columns and selected_channels and len(selected_channels) > 0:
         unique_currencies = df_listing["Original Currency"].dropna().unique()
         if len(unique_currencies) > 0:
             display_currency = unique_currencies[0] # Use first found currency if filtered by channel
             if len(unique_currencies) > 1:
                  st.info(f"Note: Multiple transaction currencies found ({unique_currencies}). Displaying average price in {display_currency}.")


    # Convert Date column again just in case and group
    df_listing["Date"] = pd.to_datetime(df_listing["Date"])
    # Ensure numeric types for aggregation
    df_listing["Sales Value in Transaction Currency"] = pd.to_numeric(df_listing["Sales Value in Transaction Currency"], errors='coerce')
    df_listing["Order Quantity"] = pd.to_numeric(df_listing["Order Quantity"], errors='coerce')
    # Drop rows where conversion failed or quantity is zero/null before grouping
    df_listing.dropna(subset=["Sales Value in Transaction Currency", "Order Quantity"], inplace=True)
    df_listing = df_listing[df_listing["Order Quantity"] > 0]


    if df_listing.empty:
         st.warning(f"No valid sales/quantity data for '{listing}' to calculate daily price.")
         return None

    # Group by Date (as date object) and Year
    grouped = df_listing.groupby([df_listing["Date"].dt.date, "Year"]).agg(
        Total_Sales_Value=("Sales Value in Transaction Currency", "sum"),
        Total_Order_Quantity=("Order Quantity", "sum")
    ).reset_index()
    # grouped.rename(columns={"Date": "Date"}, inplace=True) # Already named Date

    # Calculate Average Price, handle division by zero
    grouped["Average Price"] = grouped["Total_Sales_Value"] / grouped["Total_Order_Quantity"]
    # Convert date back to datetime for Plotly compatibility if needed, or keep as date
    grouped["Date"] = pd.to_datetime(grouped["Date"])

    # Process data per year for plotting
    dfs_processed = []
    for yr in selected_years_int: # Iterate through selected years
        df_year = grouped[grouped["Year"] == yr].copy()
        if df_year.empty:
            continue

        # Use day of year for x-axis alignment
        df_year["Day"] = df_year["Date"].dt.dayofyear
        start_day = int(df_year["Day"].min())
        end_day = int(df_year["Day"].max())

        # Reindex to ensure continuous days for forward fill
        df_year = df_year.set_index("Day").reindex(range(start_day, end_day + 1))
        df_year.index.name = "Day"

        # Forward fill missing average prices
        df_year["Average Price"] = df_year["Average Price"].ffill()

        # Apply smoothing logic (optional, based on original code)
        # Ensure Average Price is float for comparison
        df_year["Average Price"] = df_year["Average Price"].astype(float)
        prices = df_year["Average Price"].values.copy()
        # Smoothing needs careful indexing and handling of NaNs after reindexing
        last_valid_price = None
        for i in range(len(prices)):
             current_price = prices[i]
             if pd.notna(current_price):
                 if last_valid_price is not None:
                     # Apply smoothing rules only if the previous price was valid
                     if current_price < 0.75 * last_valid_price:
                         prices[i] = last_valid_price # Keep previous price if drop is too large
                     elif current_price > 1.25 * last_valid_price:
                          prices[i] = last_valid_price # Keep previous price if jump is too large
                 # Update last valid price found
                 last_valid_price = prices[i]
             elif last_valid_price is not None:
                 # If current price is NaN but we had a previous valid one, fill with it
                 prices[i] = last_valid_price

        df_year["Smoothed Average Price"] = prices

        # Add Year column back and reset index
        df_year["Year"] = yr
        df_year = df_year.reset_index()
        # Drop rows where price is still NaN after ffill and smoothing
        df_year.dropna(subset=["Smoothed Average Price"], inplace=True)

        dfs_processed.append(df_year)

    if not dfs_processed:
        st.warning("No data available after processing for the Daily Price chart.")
        return None # Or empty figure

    combined = pd.concat(dfs_processed, ignore_index=True)

    if combined.empty:
         st.warning("Combined data is empty for the Daily Price chart.")
         return None

    # Create the line chart
    fig = px.line(
        combined,
        x="Day",
        y="Smoothed Average Price", # Plot the smoothed price
        color="Year",               # Color lines by year
        title=f"Daily Average Price for {listing}",
        labels={"Day": "Day of Year", "Smoothed Average Price": f"Avg Price ({display_currency})", "Year": "Year"},
        color_discrete_sequence=px.colors.qualitative.Set1 # Use a distinct color set
    )

    fig.update_yaxes(rangemode="tozero") # Ensure y-axis starts at zero
    fig.update_layout(margin=dict(t=50, b=50), legend_title_text='Year')
    return fig

# --- End of Chunk 1 ---

# Chunk 2 of 3

# =============================================================================
# Main Dashboard Code
# =============================================================================

# --- Load Data from Google Sheets ---
# This replaces the file uploader logic
df_raw = load_data_from_gsheet()

# Check if data loading was successful
if df_raw is None or df_raw.empty:
    st.warning("Failed to load data from Google Sheet or the sheet is empty. Dashboard cannot proceed.")
    # Optionally display the sheet/worksheet name being attempted
    st.info(f"Attempted to load from Sheet: '{GOOGLE_SHEET_NAME}', Worksheet: '{WORKSHEET_NAME}'")
    st.stop() # Stop execution if no raw data

# --- Preprocess Data ---
# Pass the raw DataFrame from Google Sheets to your existing preprocess function
try:
    # Pass a copy to prevent modifying cached raw data
    df = preprocess_data(df_raw.copy())
except Exception as e:
    st.error(f"An error occurred during data preprocessing: {e}")
    st.error("Please check the 'preprocess_data' function and the structure/content of your Google Sheet.")
    # Optionally log the full traceback for debugging
    import traceback
    st.error(traceback.format_exc())
    st.stop() # Stop execution if preprocessing fails

# Check if preprocessing returned valid data
if df is None or df.empty:
    st.error("Data is empty after preprocessing. Please check the 'preprocess_data' function logic and the source data.")
    st.stop() # Stop execution if no processed data

# --- Add Sheet Info to Sidebar ---
st.sidebar.info(f"Data loaded from:\nSheet: '{GOOGLE_SHEET_NAME}'\nWorksheet: '{WORKSHEET_NAME}'")
st.sidebar.markdown("---") # Separator


# --- Prepare Filters and Variables based on Processed Data 'df' ---
# Calculate available years from the processed data
if "Custom_Week_Year" not in df.columns:
    st.error("Critical Error: 'Custom_Week_Year' column not found after preprocessing.")
    st.stop()

# Ensure the column is numeric before finding unique years; handle potential NaNs
available_custom_years = sorted(pd.to_numeric(df["Custom_Week_Year"], errors='coerce').dropna().unique().astype(int))

if not available_custom_years:
    st.error("No valid 'Custom_Week_Year' data found after preprocessing. Check calculations and sheet content.")
    st.stop()

# Determine current and previous years for defaults
current_custom_year = available_custom_years[-1]
if len(available_custom_years) >= 2:
    prev_custom_year = available_custom_years[-2]
    yoy_default_years = [prev_custom_year, current_custom_year] # Default for YOY comparisons
else:
    # Only one year available
    prev_custom_year = None # No previous year
    yoy_default_years = [current_custom_year]

default_current_year = [current_custom_year] # Default for single-year views


# --- Define Dashboard Tabs ---
# Using the tab names from your original code
tabs = st.tabs([
    "KPIs",
    "YOY Trends",
    "Daily Prices",
    "SKU Trends",
    "Pivot Table",
    "Unrecognised Sales"
])

# =============================================================================
# Tab Implementations (Using your original code structure)
# =============================================================================

# -------------------------------
# Tab 1: KPIs (Your Original Code)
# -------------------------------
with tabs[0]:
    st.markdown("### Key Performance Indicators")
    with st.expander("KPI Filters", expanded=False):
        today = datetime.date.today()
        # Ensure week calculation columns exist
        if "Custom_Week_Year" not in df.columns or "Custom_Week" not in df.columns:
             st.error("Missing 'Custom_Week_Year' or 'Custom_Week' for KPI calculations.")
        else:
            # Get available weeks for the current year
            available_weeks = sorted(pd.to_numeric(df[df["Custom_Week_Year"] == current_custom_year]["Custom_Week"], errors='coerce').dropna().unique().astype(int))

            # Determine the last fully completed week
            full_weeks = []
            if available_weeks:
                 for wk in available_weeks:
                     week_start_dt, week_end_dt = get_custom_week_date_range(current_custom_year, wk)
                     if week_end_dt and week_end_dt < today: # Check if week end is before today
                         full_weeks.append(wk)

            # Set default week for the selectbox
            default_week = full_weeks[-1] if full_weeks else (available_weeks[-1] if available_weeks else 1)

            if available_weeks: # Only show selectbox if weeks are available
                selected_week = st.selectbox(
                    "Select Week for KPI Calculation",
                    options=available_weeks,
                    index=available_weeks.index(default_week) if default_week in available_weeks else 0,
                    key="kpi_week",
                    help="Select the week to calculate KPIs for. (Defaults to the last full week if available)"
                )

                # Display the date range for the selected week
                week_start_custom, week_end_custom = get_custom_week_date_range(current_custom_year, selected_week)
                if week_start_custom and week_end_custom:
                    st.info(f"Selected Week {selected_week}: {week_start_custom.strftime('%d %b')} - {week_end_custom.strftime('%d %b, %Y')}")
                else:
                    st.warning(f"Could not determine date range for Week {selected_week}, Year {current_custom_year}.")

            else:
                st.warning(f"No weeks found for the current year ({current_custom_year}) to calculate KPIs.")
                selected_week = None # No week selected if none available

    # Proceed with KPI calculation only if a week was selected
    if selected_week is not None:
        # Filter data for the selected week (use the variable from the selectbox)
        kpi_data = df[df["Custom_Week"] == selected_week].copy() # Use a copy for calculations

        if kpi_data.empty:
            st.info(f"No sales data found for Week {selected_week} to calculate KPIs.")
        else:
            # Calculate revenue summary
            revenue_summary = kpi_data.groupby("Custom_Week_Year")["Sales Value (£)"].sum()

            # Calculate units summary if column exists
            if "Order Quantity" in kpi_data.columns:
                # Ensure 'Order Quantity' is numeric before summing
                kpi_data["Order Quantity"] = pd.to_numeric(kpi_data["Order Quantity"], errors='coerce')
                units_summary = kpi_data.groupby("Custom_Week_Year")["Order Quantity"].sum()
            else:
                units_summary = None
                st.info("Column 'Order Quantity' not found, units and AOV KPIs will not be shown.")

            # Get all years present in the entire dataset for comparison columns
            all_custom_years_in_df = sorted(pd.to_numeric(df["Custom_Week_Year"], errors='coerce').dropna().unique().astype(int))

            # Display KPIs in columns
            kpi_cols = st.columns(len(all_custom_years_in_df))

            for idx, year in enumerate(all_custom_years_in_df):
                with kpi_cols[idx]:
                    # Get revenue for the current year and week
                    revenue = revenue_summary.get(year, 0) # Get revenue for this specific year from the filtered week's data

                    # Calculate delta vs previous year (if applicable)
                    delta_rev_str = "N/A" # Default delta
                    if idx > 0:
                        prev_year = all_custom_years_in_df[idx - 1]
                        # Need data for the *same week number* but *previous year*
                        prev_year_week_data = df[(df["Custom_Week_Year"] == prev_year) & (df["Custom_Week"] == selected_week)]
                        prev_rev = prev_year_week_data["Sales Value (£)"].sum() if not prev_year_week_data.empty else 0
                        delta_rev = revenue - prev_rev
                        # Format delta only if previous year had data for this week
                        delta_rev_str = f"£{int(round(delta_rev)):,}" if prev_rev != 0 or revenue != 0 else "N/A"
                    else:
                         delta_rev_str = "" # No delta for the first year

                    # Display Revenue Metric
                    st.metric(
                        label=f"Revenue {year} (Week {selected_week})",
                        value=format_currency_int(revenue) if revenue != 0 else "£0",
                        delta=delta_rev_str if delta_rev_str else None # Avoid showing delta=0 explicitly if no comparison
                    )

                    # Display Units and AOV if possible
                    if units_summary is not None:
                        total_units = units_summary.get(year, 0) # Units for this year in the selected week

                        # Calculate Units Delta
                        delta_units_str = "N/A"
                        if idx > 0:
                            prev_year = all_custom_years_in_df[idx - 1]
                            prev_year_week_data = df[(df["Custom_Week_Year"] == prev_year) & (df["Custom_Week"] == selected_week)]
                            # Ensure 'Order Quantity' is numeric
                            prev_year_week_data['Order Quantity'] = pd.to_numeric(prev_year_week_data['Order Quantity'], errors='coerce')
                            prev_units = prev_year_week_data["Order Quantity"].sum() if not prev_year_week_data.empty else 0

                            if prev_units != 0:
                                delta_units_percent = ((total_units - prev_units) / prev_units) * 100
                                delta_units_str = f"{delta_units_percent:.1f}%"
                            elif total_units != 0:
                                delta_units_str = "vs 0" # Indicate comparison against zero
                            # else: delta stays "N/A"
                        else:
                            delta_units_str = ""

                        st.metric(
                            label=f"Units Sold {year} (Week {selected_week})",
                            value=f"{total_units:,}" if total_units is not None else "N/A",
                            delta=delta_units_str if delta_units_str else None
                        )

                        # Calculate AOV and Delta
                        aov = revenue / total_units if total_units != 0 else 0
                        delta_aov_str = "N/A"
                        if idx > 0:
                            # Previous year AOV needs prev_rev and prev_units calculated above
                            prev_aov = prev_rev / prev_units if prev_units != 0 else 0
                            if prev_aov != 0:
                                delta_aov_percent = ((aov - prev_aov) / prev_aov) * 100
                                delta_aov_str = f"{delta_aov_percent:.1f}%"
                            elif aov != 0:
                                delta_aov_str = "vs £0"
                            # else: delta stays "N/A"
                        else:
                            delta_aov_str = ""

                        st.metric(
                            label=f"AOV {year} (Week {selected_week})",
                            value=format_currency(aov) if aov != 0 else "£0.00",
                            delta=delta_aov_str if delta_aov_str else None
                        )
    else:
        st.info("Select a week from the filters above to view KPIs.")


# -----------------------------------------
# Tab 2: YOY Trends (Your Original Code)
# -----------------------------------------
with tabs[1]:
    st.markdown("### YOY Weekly Revenue Trends")
    # --- Filters ---
    with st.expander("Chart Filters", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            yoy_years = st.multiselect("Year(s)", options=available_custom_years, default=yoy_default_years, key="yoy_years")
        with col2:
            quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            quarter_selection = st.selectbox("Quarter(s)", options=quarter_options, index=0, key="quarter_dropdown_yoy")
            if quarter_selection == "All Quarters":
                selected_quarters = ["Q1", "Q2", "Q3", "Q4"]
            elif quarter_selection == "Custom...":
                # Ensure Quarter column exists before accessing unique values
                if "Quarter" in df.columns:
                     quarter_opts = sorted(df["Quarter"].dropna().unique())
                else: quarter_opts = ["Q1", "Q2", "Q3", "Q4"] # Fallback
                selected_quarters = st.multiselect("Select quarters", options=quarter_opts, default=quarter_opts, key="custom_quarters_yoy")
            else:
                selected_quarters = [quarter_selection]
        with col3:
            # Check if 'Sales Channel' column exists before creating filter
            if "Sales Channel" in df.columns:
                channel_options = sorted(df["Sales Channel"].dropna().unique())
                selected_channels = st.multiselect("Channel(s)", options=channel_options, default=[], key="yoy_channels")
            else:
                selected_channels = []
                st.caption("Sales Channel filter unavailable (column missing)")
        with col4:
             # Check if 'Listing' column exists
             if "Listing" in df.columns:
                listing_options = sorted(df["Listing"].dropna().unique())
                selected_listings = st.multiselect("Listing(s)", options=listing_options, default=[], key="yoy_listings")
             else:
                selected_listings = []
                st.caption("Listing filter unavailable (column missing)")
        with col5:
             # Check if 'Product' column exists
             if "Product" in df.columns:
                 # Filter product options based on selected listing(s) if any
                if selected_listings:
                    product_options = sorted(df[df["Listing"].isin(selected_listings)]["Product"].dropna().unique())
                else:
                    product_options = sorted(df["Product"].dropna().unique())
                selected_products = st.multiselect("Product(s)", options=product_options, default=[], key="yoy_products")
             else:
                 selected_products = []
                 st.caption("Product filter unavailable (column missing)")

        # Time grouping option (kept as 'Week' from original code)
        time_grouping = "Week"

    # --- Create and Display Chart ---
    if not yoy_years:
        st.warning("Please select at least one year in the filters to display the YOY chart.")
    else:
        fig_yoy = create_yoy_trends_chart(df, yoy_years, selected_quarters, selected_channels, selected_listings, selected_products, time_grouping=time_grouping)
        st.plotly_chart(fig_yoy, use_container_width=True)

    # --- Revenue Summary Table (Your Original Logic) ---
    st.markdown("### Revenue Summary")
    st.markdown("") # Add space

    # Filter DataFrame based on selections for the summary table
    # Use the same filters as the chart
    filtered_df_summary = df.copy()
    if yoy_years:
        filtered_df_summary = filtered_df_summary[filtered_df_summary["Custom_Week_Year"].isin(yoy_years)]
    if selected_quarters:
        filtered_df_summary = filtered_df_summary[filtered_df_summary["Quarter"].isin(selected_quarters)]
    if selected_channels:
         if "Sales Channel" in filtered_df_summary.columns:
            filtered_df_summary = filtered_df_summary[filtered_df_summary["Sales Channel"].isin(selected_channels)]
    if selected_listings:
         if "Listing" in filtered_df_summary.columns:
            filtered_df_summary = filtered_df_summary[filtered_df_summary["Listing"].isin(selected_listings)]
    if selected_products:
         if "Product" in filtered_df_summary.columns:
            filtered_df_summary = filtered_df_summary[filtered_df_summary["Product"].isin(selected_products)]

    df_revenue = filtered_df_summary.copy()

    if df_revenue.empty:
        st.info("No data available for the selected filters to build the revenue summary table.")
    else:
        # Ensure required columns have correct types
        df_revenue["Custom_Week_Year"] = pd.to_numeric(df_revenue["Custom_Week_Year"], errors='coerce').astype('Int64')
        df_revenue["Week"] = pd.to_numeric(df_revenue["Week"], errors='coerce').astype('Int64')
        df_revenue.dropna(subset=["Custom_Week_Year", "Week"], inplace=True) # Drop rows where conversion failed

        if df_revenue.empty:
            st.info("No valid week/year data after type conversion for summary table.")
        else:
            filtered_current_year = df_revenue["Custom_Week_Year"].max()
            df_revenue_current = df_revenue[df_revenue["Custom_Week_Year"] == filtered_current_year].copy()

            # Ensure week start/end calculation columns exist or can be derived
            if "Custom_Week_Start" not in df_revenue_current.columns or "Custom_Week_End" not in df_revenue_current.columns:
                 # Attempt to derive if missing, otherwise skip this logic
                 try:
                     week_results_current = df_revenue_current.apply(lambda row: get_custom_week_date_range(row['Custom_Week_Year'], row['Week']), axis=1)
                     df_revenue_current[["Custom_Week_Start", "Custom_Week_End"]] = pd.DataFrame(week_results_current.tolist(), index=df_revenue_current.index)
                 except Exception:
                     st.warning("Cannot determine week start/end dates for summary table calculation.")
                     df_revenue_current = pd.DataFrame() # Prevent further processing

            if not df_revenue_current.empty:
                # Determine last complete week in the current year's data
                today = datetime.date.today()
                # Need to handle potential NaT in Custom_Week_End after derivation/conversion
                df_revenue_current["Custom_Week_End"] = pd.to_datetime(df_revenue_current["Custom_Week_End"], errors='coerce').dt.date
                df_full_weeks_current = df_revenue_current.dropna(subset=["Custom_Week_End"])
                df_full_weeks_current = df_full_weeks_current[df_full_weeks_current["Custom_Week_End"] < today].copy()

                # Get unique weeks and sort by end date to find the latest complete ones
                unique_weeks_current = (df_full_weeks_current.groupby(["Custom_Week_Year", "Week"])
                                        .agg(Week_End=("Custom_Week_End", "first")) # Get the end date for sorting
                                        .reset_index()
                                        .sort_values("Week_End"))

                if unique_weeks_current.empty:
                    st.info("Not enough complete week data in the filtered current year to build the revenue summary table.")
                else:
                    last_complete_week_row_current = unique_weeks_current.iloc[-1]
                    last_week_number = int(last_complete_week_row_current["Week"]) # Ensure integer
                    last_week_year = int(last_complete_week_row_current["Custom_Week_Year"])

                    # Get the last 4 complete weeks based on the sorted unique weeks
                    last_4_weeks_current = unique_weeks_current.tail(4)
                    last_4_week_numbers = last_4_weeks_current["Week"].astype(int).tolist() # Ensure integer list

                    # Determine grouping key (Listing or Product)
                    if "Product" in df_revenue.columns and selected_listings and len(selected_listings) == 1:
                         grouping_key = "Product"
                    elif "Listing" in df_revenue.columns:
                        grouping_key = "Listing"
                    else:
                        st.warning("Cannot determine grouping key (Listing/Product) for summary table.")
                        grouping_key = None # Set to None to skip grouping

                    if grouping_key:
                        # --- Calculations for Current Year ---
                        rev_last_4_current = (df_full_weeks_current[df_full_weeks_current["Week"].isin(last_4_week_numbers)]
                                            .groupby(grouping_key)["Sales Value (£)"].sum()
                                            .rename(f"Last 4 Weeks Revenue ({last_week_year})").round(0).astype(int))

                        rev_last_1_current = (df_full_weeks_current[df_full_weeks_current["Week"] == last_week_number]
                                            .groupby(grouping_key)["Sales Value (£)"].sum()
                                            .rename(f"Last Week Revenue ({last_week_year})").round(0).astype(int))

                        # --- Calculations for Previous Year (if available) ---
                        rev_last_4_last_year = pd.Series(dtype=int, name="Last 4 Weeks Revenue (Prev Year)")
                        rev_last_1_last_year = pd.Series(dtype=int, name="Last Week Revenue (Prev Year)")
                        prev_year_label = "Prev Year"

                        if len(yoy_years) >= 2:
                             # Assume the second to last selected year is the previous one
                             # Or find the year before last_week_year in the filtered data
                             available_years_in_filtered = sorted(pd.to_numeric(df_revenue["Custom_Week_Year"], errors='coerce').dropna().unique().astype(int))
                             year_index = available_years_in_filtered.index(last_week_year) if last_week_year in available_years_in_filtered else -1

                             if year_index > 0:
                                 last_year = available_years_in_filtered[year_index - 1]
                                 prev_year_label = str(last_year) # Use actual year in label
                                 df_revenue_last_year = df_revenue[df_revenue["Custom_Week_Year"] == last_year].copy()

                                 if not df_revenue_last_year.empty:
                                     rev_last_1_last_year = (df_revenue_last_year[df_revenue_last_year["Week"] == last_week_number]
                                                             .groupby(grouping_key)["Sales Value (£)"].sum()
                                                             .rename(f"Last Week Revenue ({last_year})").round(0).astype(int))

                                     rev_last_4_last_year = (df_revenue_last_year[df_revenue_last_year["Week"].isin(last_4_week_numbers)]
                                                             .groupby(grouping_key)["Sales Value (£)"].sum()
                                                             .rename(f"Last 4 Weeks Revenue ({last_year})").round(0).astype(int))

                        # --- Combine Results ---
                        # Get all unique keys (Listings/Products) from the current year filtered data
                        all_keys_current = pd.Series(sorted(df_revenue_current[grouping_key].dropna().unique()), name=grouping_key)
                        revenue_summary = pd.DataFrame(all_keys_current).set_index(grouping_key)

                        # Join the calculated series
                        revenue_summary = revenue_summary.join(rev_last_4_current, how="left")\
                                                         .join(rev_last_1_current, how="left")\
                                                         .join(rev_last_4_last_year.rename(f"Last 4 Weeks Revenue ({prev_year_label})"), how="left")\
                                                         .join(rev_last_1_last_year.rename(f"Last Week Revenue ({prev_year_label})"), how="left")

                        revenue_summary = revenue_summary.fillna(0) # Fill missing joins with 0

                        # Calculate Differences and % Changes
                        current_4wk_col = f"Last 4 Weeks Revenue ({last_week_year})"
                        prev_4wk_col = f"Last 4 Weeks Revenue ({prev_year_label})"
                        current_1wk_col = f"Last Week Revenue ({last_week_year})"
                        prev_1wk_col = f"Last Week Revenue ({prev_year_label})"

                        revenue_summary["Last 4 Weeks Diff"] = revenue_summary[current_4wk_col] - revenue_summary[prev_4wk_col]
                        revenue_summary["Last Week Diff"] = revenue_summary[current_1wk_col] - revenue_summary[prev_1wk_col]

                        # Calculate % change safely avoiding division by zero
                        revenue_summary["Last 4 Weeks % Change"] = revenue_summary.apply(
                            lambda row: (row["Last 4 Weeks Diff"] / row[prev_4wk_col] * 100) if row[prev_4wk_col] != 0 else (100.0 if row["Last 4 Weeks Diff"] > 0 else 0.0), axis=1)
                        revenue_summary["Last Week % Change"] = revenue_summary.apply(
                            lambda row: (row["Last Week Diff"] / row[prev_1wk_col] * 100) if row[prev_1wk_col] != 0 else (100.0 if row["Last Week Diff"] > 0 else 0.0), axis=1)


                        revenue_summary = revenue_summary.reset_index() # Make grouping key a column again

                        # Define desired column order using the dynamic year labels
                        desired_order = [grouping_key,
                                         current_4wk_col, prev_4wk_col, "Last 4 Weeks Diff", "Last 4 Weeks % Change",
                                         current_1wk_col, prev_1wk_col, "Last Week Diff", "Last Week % Change"]
                        # Ensure all desired columns exist before reordering
                        desired_order = [col for col in desired_order if col in revenue_summary.columns]
                        revenue_summary = revenue_summary[desired_order]

                        # ---- Total Summary Row Calculation ----
                        summary_row = {col: revenue_summary[col].sum() for col in desired_order if col != grouping_key}
                        summary_row[grouping_key] = "Total"

                        total_last4_last_year = summary_row[prev_4wk_col]
                        total_last_week_last_year = summary_row[prev_1wk_col]

                        summary_row["Last 4 Weeks % Change"] = (summary_row["Last 4 Weeks Diff"] / total_last4_last_year * 100) if total_last4_last_year != 0 else (100.0 if summary_row["Last 4 Weeks Diff"] > 0 else 0.0)
                        summary_row["Last Week % Change"] = (summary_row["Last Week Diff"] / total_last_week_last_year * 100) if total_last_week_last_year != 0 else (100.0 if summary_row["Last Week Diff"] > 0 else 0.0)

                        total_df = pd.DataFrame([summary_row])[desired_order] # Ensure same column order

                        # ---- Styling ----
                        def color_diff(val):
                            """Applies red/green color to negative/positive numbers."""
                            try:
                                val = float(val) # Ensure it's numeric for comparison
                                if val < 0: return 'color: red'
                                elif val > 0: return 'color: green'
                                else: return ''
                            except (ValueError, TypeError):
                                return '' # Return empty style for non-numeric

                        # Define formats using dynamic column names
                        formats = {
                            current_4wk_col: "{:,}", prev_4wk_col: "{:,}",
                            current_1wk_col: "{:,}", prev_1wk_col: "{:,}",
                            "Last 4 Weeks Diff": "{:,.0f}", "Last Week Diff": "{:,.0f}",
                            "Last 4 Weeks % Change": "{:.1f}%", "Last Week % Change": "{:.1f}%"
                        }
                        color_cols = ["Last 4 Weeks Diff", "Last Week Diff", "Last 4 Weeks % Change", "Last Week % Change"]

                        # Apply styling to Total row
                        styled_total = total_df.style.format(formats).applymap(color_diff, subset=color_cols)\
                                                 .set_properties(**{'font-weight': 'bold'})

                        st.markdown("##### Total Summary")
                        st.dataframe(styled_total, use_container_width=True, hide_index=True) # Hide index for total row

                        # Apply styling to Detailed rows
                        styled_main = revenue_summary.style.format(formats).applymap(color_diff, subset=color_cols)

                        st.markdown("##### Detailed Summary")
                        st.dataframe(styled_main, use_container_width=True, hide_index=True) # Hide index for detail rows
                    else:
                         st.info("Summary table requires 'Listing' or 'Product' column and valid selections.")

# --- End of Chunk 2 ---

# Chunk 3 of 3

# -------------------------------
# Tab 3: Daily Prices (Your Original Code)
# -------------------------------
with tabs[2]:
    st.markdown("### Daily Prices for Top Listings")
    # --- Filters ---
    with st.expander("Daily Price Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Default years for daily view (e.g., current and maybe next planned)
            # Adjust this logic if needed
            default_daily_years = [year for year in available_custom_years if year >= datetime.date.today().year -1] # Show current and previous year by default
            if not default_daily_years: default_daily_years = [current_custom_year] # Fallback to current if none found
            elif len(default_daily_years) > 2: default_daily_years = default_daily_years[-2:] # Limit to last 2 years if more match

            selected_daily_years = st.multiselect(
                "Select Year(s)",
                options=available_custom_years,
                default=default_daily_years,
                key="daily_years",
                help="Select year(s) to display daily prices."
                )
        with col2:
            quarter_options_daily = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            quarter_selection_daily = st.selectbox("Quarter(s)", options=quarter_options_daily, index=0, key="quarter_dropdown_daily")

            if quarter_selection_daily == "Custom...":
                 # Ensure Quarter column exists before accessing unique values
                if "Quarter" in df.columns:
                     quarter_opts_daily = sorted(df["Quarter"].dropna().unique())
                else: quarter_opts_daily = ["Q1", "Q2", "Q3", "Q4"] # Fallback
                selected_daily_quarters = st.multiselect("Select Quarter(s)", options=quarter_opts_daily, default=[], key="daily_quarters_custom", help="Select one or more quarters to filter.")
            elif quarter_selection_daily == "All Quarters":
                selected_daily_quarters = ["Q1", "Q2", "Q3", "Q4"]
            else:
                selected_daily_quarters = [quarter_selection_daily]
        with col3:
             # Check if 'Sales Channel' column exists
             if "Sales Channel" in df.columns:
                channel_options_daily = sorted(df["Sales Channel"].dropna().unique())
                selected_daily_channels = st.multiselect("Select Sales Channel(s)", options=channel_options_daily, default=[], key="daily_channels", help="Select one or more sales channels to filter the daily price data.")
             else:
                selected_daily_channels = []
                st.caption("Sales Channel filter unavailable")
        with col4:
             # Slider for week range
             daily_week_range = st.slider(
                 "Select Week Range",
                 min_value=1,
                 max_value=53, # Allow up to week 53
                 value=(1, 53), # Default to full year
                 step=1,
                 key="daily_week_range",
                 help="Select the range of weeks to display in the Daily Prices section."
                 )

    # --- Display Charts for Main Listings ---
    # Define your main listings here
    main_listings = ["Pattern Pants", "Pattern Shorts", "Solid Pants", "Solid Shorts", "Patterned Polos"] # As per your original code

    # Check if 'Listing' column exists
    if "Listing" not in df.columns:
        st.error("Column 'Listing' not found. Cannot display Daily Price charts.")
    else:
        # Filter the list of main_listings to only those present in the data
        available_main_listings = [l for l in main_listings if l in df["Listing"].unique()]
        if not available_main_listings:
             st.warning("None of the specified main listings found in the data.")
        else:
            for listing in available_main_listings:
                st.subheader(listing)
                fig_daily = create_daily_price_chart(df, listing, selected_daily_years, selected_daily_quarters, selected_daily_channels, week_range=daily_week_range)
                if fig_daily: # Check if the function returned a figure
                    st.plotly_chart(fig_daily, use_container_width=True)
                # else: # Optional: message if no chart generated for a specific listing due to filters
                #     st.info(f"No daily price chart generated for '{listing}' with current filters.")

    # --- Daily Prices Comparison Section ---
    st.markdown("### Daily Prices Comparison")
    with st.expander("Comparison Chart Filters", expanded=False):
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            comp_years = st.multiselect("Select Year(s)", options=available_custom_years, default=default_daily_years, key="comp_years", help="Select the year(s) for the comparison chart.")
        with comp_col2:
            comp_quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            comp_quarter_selection = st.selectbox("Quarter(s)", options=comp_quarter_options, index=0, key="quarter_dropdown_comp")
            if comp_quarter_selection == "Custom...":
                 if "Quarter" in df.columns:
                     comp_quarter_opts = sorted(df["Quarter"].dropna().unique())
                 else: comp_quarter_opts = ["Q1", "Q2", "Q3", "Q4"]
                 comp_quarters = st.multiselect("Select Quarter(s)", options=comp_quarter_opts, default=[], key="comp_quarters_custom", help="Select one or more quarters for comparison.")
            elif comp_quarter_selection == "All Quarters":
                comp_quarters = ["Q1", "Q2", "Q3", "Q4"]
            else:
                comp_quarters = [comp_quarter_selection]
        with comp_col3:
            if "Sales Channel" in df.columns:
                 comp_channel_opts = sorted(df["Sales Channel"].dropna().unique())
                 comp_channels = st.multiselect("Select Sales Channel(s)", options=comp_channel_opts, default=[], key="comp_channels", help="Select the sales channel(s) for the comparison chart.")
            else:
                 comp_channels = []
                 st.caption("Sales Channel filter unavailable")

        # Listing selection for comparison chart
        if "Listing" in df.columns:
             comp_listing_opts = sorted(df["Listing"].dropna().unique())
             # Select first listing as default if available
             comp_listing_default = comp_listing_opts[0] if comp_listing_opts else None
             comp_listing = st.selectbox("Select Listing", options=comp_listing_opts, index=0 if comp_listing_default else -1 , key="comp_listing", help="Select a listing for daily prices comparison.")
        else:
             comp_listing = None
             st.warning("Listing selection unavailable (column missing)")

    # Generate and display comparison chart
    if comp_listing and comp_years: # Ensure a listing and years are selected
        fig_comp = create_daily_price_chart(df, comp_listing, comp_years, comp_quarters, comp_channels) # week_range is not passed here per original code
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info(f"No comparison chart generated for '{comp_listing}' with current filters.")
    elif not comp_listing:
         st.info("Select a listing in the filters above to view the comparison chart.")
    elif not comp_years:
         st.info("Select at least one year in the filters above to view the comparison chart.")

# -------------------------------
# Tab 4: SKU Trends (Your Original Code)
# -------------------------------
with tabs[3]:
    st.markdown("### SKU Trends")
    if "Product SKU" not in df.columns:
        st.error("The dataset does not contain a 'Product SKU' column. SKU Trends cannot be displayed.")
    else:
        # --- Filters ---
        with st.expander("Chart Filters", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sku_text = st.text_input("Enter Product SKU", value="", key="sku_input", help="Enter a SKU (or part of it) to display its weekly revenue trends.")
            with col2:
                sku_years = st.multiselect("Select Year(s)", options=available_custom_years, default=default_current_year, key="sku_years", help="Default is the current custom week year.")
            with col3:
                if "Sales Channel" in df.columns:
                     sku_channel_opts = sorted(df["Sales Channel"].dropna().unique())
                     sku_channels = st.multiselect("Select Sales Channel(s)", options=sku_channel_opts, default=[], key="sku_channels", help="Select one or more sales channels to filter SKU trends. If empty, all channels are shown.")
                else:
                     sku_channels = []
                     st.caption("Sales Channel filter unavailable")
            with col4:
                 # Slider for week range
                 week_range_sku = st.slider("Select Week Range", min_value=1, max_value=53, value=(1, 53), step=1, key="sku_week_range", help="Select the range of weeks to display.")

        # --- Display Chart and Tables ---
        if sku_text.strip() == "":
            st.info("Please enter a Product SKU in the filters above to view its trends.")
        elif not sku_years:
             st.warning("Please select at least one year in the filters to view SKU trends.")
        else:
            # Create and display the SKU line chart
            fig_sku = create_sku_line_chart(df, sku_text, sku_years, selected_channels=sku_channels, week_range=week_range_sku)
            if fig_sku is not None: # Check if figure was created
                st.plotly_chart(fig_sku, use_container_width=True)
            # else: # Message handled within the function if no data

            # --- SKU Units Summary ---
            # Filter data again for the summary tables
            filtered_sku_data = df[df["Product SKU"].str.contains(sku_text, case=False, na=False)]
            if sku_years:
                filtered_sku_data = filtered_sku_data[filtered_sku_data["Custom_Week_Year"].isin(sku_years)]
            if sku_channels and len(sku_channels) > 0:
                 if "Sales Channel" in filtered_sku_data.columns:
                    filtered_sku_data = filtered_sku_data[filtered_sku_data["Sales Channel"].isin(sku_channels)]
            if week_range_sku:
                if "Week" in filtered_sku_data.columns:
                     filtered_sku_data = filtered_sku_data[(filtered_sku_data["Week"] >= week_range_sku[0]) & (filtered_sku_data["Week"] <= week_range_sku[1])]
                else: st.warning("Week column missing for week range filter in SKU summary.")


            if "Order Quantity" in filtered_sku_data.columns:
                # Ensure Order Quantity is numeric
                filtered_sku_data["Order Quantity"] = pd.to_numeric(filtered_sku_data["Order Quantity"], errors='coerce')
                filtered_sku_data.dropna(subset=["Order Quantity"], inplace=True) # Drop rows where conversion failed

                if not filtered_sku_data.empty:
                    # Total Units Sold Summary (Sum across all found SKUs matching the text)
                    total_units = filtered_sku_data.groupby("Custom_Week_Year")["Order Quantity"].sum().reset_index()
                    if not total_units.empty:
                        total_units_summary = total_units.set_index("Custom_Week_Year").T
                        total_units_summary.index = ["Total Units Sold (All Matching SKUs)"]
                        st.markdown("##### Total Units Sold Summary")
                        # Format as integer
                        st.dataframe(total_units_summary.astype(int), use_container_width=True)
                    else: st.info("No units sold data for total summary.")

                    # SKU Breakdown (Units per specific SKU matching the text)
                    sku_units = filtered_sku_data.groupby(["Product SKU", "Custom_Week_Year"])["Order Quantity"].sum().reset_index()
                    if not sku_units.empty:
                         # Pivot table for breakdown
                         sku_pivot = sku_units.pivot(index="Product SKU", columns="Custom_Week_Year", values="Order Quantity")
                         sku_pivot = sku_pivot.fillna(0).astype(int) # Fill NaNs and ensure integer
                         st.markdown("##### SKU Breakdown (Units Sold by Custom Week Year)")
                         st.dataframe(sku_pivot, use_container_width=True)
                    else: st.info("No data for SKU breakdown table.")
                else:
                    st.info("No valid 'Order Quantity' data found for the selected SKU filters.")
            else:
                st.info("Column 'Order Quantity' not found, cannot show units sold summary.")

# -------------------------------
# Tab 5: Pivot Table: Revenue by Week (Your Original Code)
# -------------------------------
with tabs[4]:
    st.markdown("### Pivot Table: Revenue by Week")
    # --- Filters ---
    with st.expander("Pivot Table Filters", expanded=False):
        pivot_years = st.multiselect("Select Year(s) for Pivot Table", options=available_custom_years, default=default_current_year, key="pivot_years", help="Default is the current custom week year.")

        if "Quarter" in df.columns:
             pivot_quarter_opts = sorted(df["Quarter"].dropna().unique())
             pivot_quarters = st.multiselect("Select Quarter(s)", options=pivot_quarter_opts, default=pivot_quarter_opts, key="pivot_quarters", help="Select one or more quarters to filter by.")
        else:
             pivot_quarters = []
             st.caption("Quarter filter unavailable")

        if "Sales Channel" in df.columns:
             pivot_channel_opts = sorted(df["Sales Channel"].dropna().unique())
             pivot_channels = st.multiselect("Select Sales Channel(s)", options=pivot_channel_opts, default=[], key="pivot_channels", help="Select one or more channels to filter. If empty, all channels are shown.")
        else:
             pivot_channels = []
             st.caption("Sales Channel filter unavailable")

        if "Listing" in df.columns:
             pivot_listing_opts = sorted(df["Listing"].dropna().unique())
             pivot_listings = st.multiselect("Select Listing(s)", options=pivot_listing_opts, default=[], key="pivot_listings", help="Select one or more listings to filter. If empty, all listings are shown.")
        else:
             pivot_listings = []
             st.caption("Listing filter unavailable")

        # Determine product options based on listing selection
        pivot_product_options = []
        if "Product" in df.columns:
             if pivot_listings and len(pivot_listings) > 0: # Filter products if listings are selected
                 pivot_product_options = sorted(df[df["Listing"].isin(pivot_listings)]["Product"].dropna().unique())
             else: # Otherwise show all products
                 pivot_product_options = sorted(df["Product"].dropna().unique())
             pivot_products = st.multiselect("Select Product(s)", options=pivot_product_options, default=[], key="pivot_products", help="Select one or more products to filter. Options depend on selected listings.")
        else:
             pivot_products = []
             st.caption("Product filter unavailable")


    # --- Create and Display Pivot Table ---
    if not pivot_years:
        st.warning("Please select at least one year for the Pivot Table.")
    else:
        # Determine grouping key (Listing or Product)
        # Group by Product only if exactly one Listing is selected, otherwise group by Listing
        grouping_key = "Product" if (pivot_listings and len(pivot_listings) == 1 and "Product" in df.columns) else ("Listing" if "Listing" in df.columns else None)

        if grouping_key:
            # Pass effective product filter only when grouping by Product
            effective_products = pivot_products if grouping_key == "Product" else []

            pivot = create_pivot_table(
                df,
                selected_years=pivot_years,
                selected_quarters=pivot_quarters,
                selected_channels=pivot_channels,
                selected_listings=pivot_listings,
                selected_products=effective_products, # Use filtered products only when grouping by Product
                grouping_key=grouping_key
                )

            # --- Add Date Range to MultiIndex Header if single year selected ---
            if len(pivot_years) == 1 and not pivot.empty and isinstance(pivot.columns, pd.Index): # Check it's not already MultiIndex
                 year_for_date = int(pivot_years[0])
                 new_columns_tuples = []
                 for col in pivot.columns:
                     if col == "Total Revenue":
                         new_columns_tuples.append(("Total Revenue", "")) # Add empty subheader for total
                     elif isinstance(col, str) and col.startswith("Week "):
                         try:
                             week_number = int(col.split()[1])
                             mon, fri = get_custom_week_date_range(year_for_date, week_number)
                             date_range = f"{mon.strftime('%d %b')} - {fri.strftime('%d %b')}" if mon and fri else ""
                             new_columns_tuples.append((col, date_range)) # Tuple: (Week X, Date Range)
                         except Exception:
                             new_columns_tuples.append((col, "")) # Fallback empty date range
                     else:
                         new_columns_tuples.append((str(col), "")) # Handle unexpected columns

                 if new_columns_tuples: # Only apply if tuples were created
                    pivot.columns = pd.MultiIndex.from_tuples(new_columns_tuples)

            # Display the pivot table DataFrame
            st.dataframe(pivot, use_container_width=True)
        else:
             st.error("Cannot create pivot table: Required grouping column ('Listing' or 'Product') not found in data.")


# -------------------------------
# Tab 6: Unrecognised Sales (Your Original Code)
# -------------------------------
with tabs[5]:
    st.markdown("### Unrecognised Sales")
    if "Listing" not in df.columns:
        st.error("Column 'Listing' not found. Cannot identify unrecognised sales.")
    else:
        # Filter rows where 'Listing' contains "unrecognised" (case-insensitive)
        unrecognised_sales = df[df["Listing"].str.contains("unrecognised", case=False, na=False)].copy() # Use copy

        # Define columns to potentially drop (as per your original code)
        # Check if these columns exist before dropping to avoid errors
        columns_to_drop_orig = ["Year", "Weekly Sales Value (£)", "YOY Growth (%)"]
        columns_to_drop_existing = [col for col in columns_to_drop_orig if col in unrecognised_sales.columns]

        if columns_to_drop_existing:
             unrecognised_sales = unrecognised_sales.drop(columns=columns_to_drop_existing, errors='ignore')

        if unrecognised_sales.empty:
            st.info("No unrecognised sales found based on 'Listing' column containing 'unrecognised'.")
        else:
            st.info(f"Found {len(unrecognised_sales)} rows potentially related to unrecognised sales.")
            # Display the filtered DataFrame
            # Select or reorder columns for better display if needed
            display_cols = ["Date", "Sales Channel", "Listing", "Product SKU", "Product", "Sales Value (£)", "Order Quantity"] # Example order
            display_cols_existing = [col for col in display_cols if col in unrecognised_sales.columns]
            st.dataframe(unrecognised_sales[display_cols_existing], use_container_width=True, hide_index=True)


# --- Footer ---
st.markdown("---")
st.write("YOY Dashboard v2.0 (Google Sheets Integration)")

# --- End of Chunk 3 / End of Script ---
