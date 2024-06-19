import streamlit as st
import pandas as pd
import Sigma_preprocess
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import func
import streamlit as st
import pandas as pd
import numpy as np
import model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, \
    VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


file_path = r"Alpha_raw.csv"
df = Sigma_preprocess.read_large_csv(file_path, chunk_size=10000)
df = Sigma_preprocess.preprocessing(df)
df=df.drop(['product_description','product_keywords','product_gender_target','brand_url','product_material', 'seller_badge'],axis=1)


df = df.drop_duplicates()


st.sidebar.title('Business Analysis Dashboard')
user_menu = st.sidebar.radio('Select an Option', ('Key Stats', 'Top 10s', 'Product wise Analysis', 'Seller wise Analysis','Geographical Analysis','Item Price Prediction'))

if user_menu == 'Key Stats':
    st.sidebar.header("Key Statistics")


    top_selling_product_type = df.groupby('product_type')['sold'].sum().idxmax()


    most_successful_seller = df.groupby('seller_id')['sold'].sum().idxmax()


    highest_pass_rate_seller = df.loc[df['seller_pass_rate'].idxmax()]['seller_id']


    most_liked_product = df.loc[df['product_like_count'].idxmax()]['product_name']


    highest_priced_product = df.loc[df['price_usd'].idxmax()]['product_name']


    most_sellers_country = df['seller_country'].value_counts().idxmax()


    top_brand_name = df.groupby('brand_name')['sold'].sum().idxmax()


    top_selling_product_per_season = df.groupby('product_season')['sold'].sum().idxmax()


    average_price_per_product_type = df.groupby('product_type')['price_usd'].mean().sort_values(ascending=False)


    most_reserved_product_name = df.nlargest(1, 'reserved')['product_name'].iloc[0]


    top_country_based_on_sales = df.groupby('seller_country')['sold'].sum().nlargest(1)


    brand_with_highest_average_price = df.groupby('brand_name')['price_usd'].mean().idxmax()


    product_type_with_highest_avg_like_count = df.groupby('product_type')['product_like_count'].mean().idxmax()


    seller_with_highest_followers = df.loc[df['seller_num_followers'].idxmax()]['seller_id']


    warehouse_with_most_products_name = df.groupby('warehouse_name')['available'].sum().idxmax()



    st.markdown("## Key Statistics")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### :package: Top Product Type")
        st.markdown(f"<h4 style='color: blue;'>{top_selling_product_type}</h4>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"### :star: Top Successful Seller")
        st.markdown(f"<h4 style='color: green;'>{most_successful_seller}</h4>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"### :trophy: Most Reliable Seller")
        st.markdown(f"<h4 style='color: red;'>{highest_pass_rate_seller}</h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### :heart: Most Liked Product")
        st.markdown(f"<h4 style='color: purple;'>{most_liked_product}</h4>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"### :moneybag: Most Costly Product")
        st.markdown(f"<h4 style='color: orange;'>{highest_priced_product}</h4>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"### :earth_africa: Top Seller Country")
        st.markdown(f"<h4 style='color: teal;'>{most_sellers_country}</h4>", unsafe_allow_html=True)

    st.markdown("---")

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown(f"### :bookmark_tabs: Most Reserved Product")
        st.markdown(f"<h4 style='color: purple;'>{most_reserved_product_name}</h4>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"### :chart_with_upwards_trend: Top Brand (By Sales)")
        st.markdown(f"<h4 style='color: blue;'>{top_brand_name}</h4>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"### :crown: Costliest Brand")
        st.markdown(f"<h4 style='color: teal;'>{brand_with_highest_average_price}</h4>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### :thumbsup: Favourite Product Type")
        st.markdown(f"<h4 style='color: blue;'>{product_type_with_highest_avg_like_count}</h4>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"### :busts_in_silhouette: Most Followed Seller")
        st.markdown(f"<h4 style='color: green;'>{seller_with_highest_followers}</h4>", unsafe_allow_html=True)

    with col3:
        st.markdown("### :package: Most Stocked Warehouse")
        st.markdown(f"<h4 style='color: red;'>{warehouse_with_most_products_name}</h4>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    body {
        font-family: 'Helvetica Neue', sans-serif;
        background: url('https://www.bing.com/images/search?view=detailV2&ccid=gfwRnYyc&id=B4F533034E31F962244F4171905608E43F016707&thid=OIP.gfwRnYycPq_c0FRH0-10XAHaEK&mediaurl=https%3a%2f%2fimg.freepik.com%2fpremium-vector%2fonline-shopping-digital-technology-with-icon-blue-background-ecommerce-online-store-marketing_252172-219.jpg&exph=352&expw=626&q=alpha+e+commerce+background+image+for+webpages&simid=608046801572035341&FORM=IRPRST&ck=8075F72F261F10296714AC0ED694F9E1&selectedIndex=24&itb=0&ajaxhist=0&ajaxserp=0') no-repeat center center fixed;
        background-size: cover;
    }
    .main .block-container {
        padding-top: 50px;
        padding-bottom: 50px;
        padding-left: 50px;
        padding-right: 50px;
        backdrop-filter: blur(10px);
    }
    .css-1d391kg {
        backdrop-filter: blur(10px);
    }
    .block-container h4 {
        color: #333;
        font-size: 1.2em;
    }
    .block-container .stButton button {
        background-color: #1c1c1c;
        color: white;
        border-radius: 4px;
        padding: 0.6em 1.2em;
        margin: 0.5em;
    }
    .block-container .stButton button:hover {
        background-color: #333;
    }
    .block-container .stMarkdown {
        margin-bottom: 20px;
    }
    .block-container h3 {
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if user_menu == 'Top 10s':
    st.sidebar.header("Top 10s Analysis")

    columns = ['product_type', 'product_name', 'product_category', 'product_season', 'brand_name', 'seller_country']
    metrics = ['sold', 'product_like_count', 'price_usd', 'available', 'reserved']

    st.markdown("<h1 style='text-align: center;'>The Top 10s</h1>", unsafe_allow_html=True)

    selected_column = st.selectbox("Select Column", columns)
    selected_metric = st.selectbox("Select Metric", metrics)

    if selected_column and selected_metric:
        if selected_metric in ['price_usd', 'product_like_count']:
            top_10 = df.groupby(selected_column)[selected_metric].mean().nlargest(10).reset_index()
            metric_type = "(Average)"
        else:
            top_10 = df.groupby(selected_column)[selected_metric].sum().nlargest(10).reset_index()
            metric_type = "(Total)"

        top_10['Rank'] = top_10[selected_metric].rank(method='max', ascending=False).astype(int)


        # Highlight of the top 3
        def highlight_top(row):
            if row['Rank'] == 1:
                return ['font-weight: bold'] * len(row)
            elif row['Rank'] == 2:
                return ['font-weight: bold'] * len(row)
            elif row['Rank'] == 3:
                return ['font-weight: bold'] * len(row)
            return [''] * len(row)


        st.markdown(f"### Top 10 {selected_column} by {selected_metric} {metric_type}")
        st.table(top_10.style.apply(highlight_top, axis=1))

    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
        padding: 20px;
    }
    .stSidebar {
        background-color: #ffffff;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.write("Please select an option from the sidebar")

if user_menu == 'Product wise Analysis':
    st.sidebar.header("Product wise Analysis")

    columns = ['product_type', 'product_category']
    selected_column = st.selectbox("Select Column", columns)

    if selected_column == 'product_type':
        func.generate_graphs(df, selected_column)

    elif selected_column == 'product_category':
        func.generate_graphs(df, selected_column)






if user_menu == 'Seller wise Analysis':
    st.sidebar.header("Seller wise Analysis")
    func.seller_analysis(df)

if user_menu == 'Geographical Analysis':
    st.sidebar.header("Geographical Analysis")
    func.geographic_analysis(df)

if user_menu == 'Item Price Prediction':
    st.sidebar.header("Item Price Prediction")
    model.ml()
