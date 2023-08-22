import streamlit as st 

st.set_page_config(
    page_title='ML IE',
    page_icon='📈'
)


st.title("INNOVATIVE EXAM - Machine Learning")
st.subheader("Group no 4 - 36 to 47")

if st.sidebar.button("Best Fit Line and Residuals"):
    
    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    import plotly.express as px




    st.title("Best Fit line")
    code1 = """
    # Importing all libraries

    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    import plotly.express as px

    """
    st.markdown("```python\n" + code1 + "\n```")

    code2 = '''
    # Load the advertising dataset

    advertising = pd.read_csv("advertising.csv")
    advertising
    '''

    st.markdown("```python\n" + code2 + "\n```")


    # Load the advertising dataset
    advertising = pd.read_csv("advertising.csv")
    st.subheader("Dataset")
    st.dataframe(advertising)


    code3 = '''
    # Data overview 
    advertising.describe()
    '''

    st.markdown("```python\n" + code3 + "\n```")


    object_columns = advertising.select_dtypes(include=['object','bool']).columns.tolist()
    num_columns = advertising.select_dtypes(include=['int', 'float']).columns.tolist()

    col1, col2 = st.columns(2)

    if len(num_columns) > 0:
        with col1:
            selected_num_column = st.selectbox('Select a Numerical column', num_columns)
            if selected_num_column:
                neg = []
                zeros = []

                statistics = advertising[selected_num_column].describe()

                for val in advertising[selected_num_column]:
                    if val < 0:
                        neg.append(1)
                    if val == 0:
                        zeros.append(1)

                statistics["Value count"] = statistics.pop("count")
                statistics["Mean"] = statistics.pop("mean")
                statistics.pop("std")
                statistics["Minimum"] = statistics.pop("min")
                statistics["Quartile 1 (25%)"] = statistics.pop("25%")
                statistics["Quartile 2 (50%)"] = statistics.pop("50%")
                statistics["Quartile 3 (75%)"] = statistics.pop("75%")
                statistics["Maximum"] = statistics.pop("max")
                statistics["Negative Values"] = sum(neg)
                statistics["Zero Values"] = sum(zeros)
                statistics["Null Values"] = advertising[selected_num_column].isna().sum()

                statistics_df = pd.DataFrame(statistics)

                st.write("Overview for", selected_num_column)
                st.write(statistics_df)

    if len(object_columns) > 0:
        with col2:
            selected_cat_column = st.selectbox('Select a Categorical column', object_columns)
            if selected_cat_column:
                values_to_convert = ['na', 'not applicable']
                new_df = advertising.copy()
                new_df[selected_cat_column] = new_df[selected_cat_column].str.lower().replace(values_to_convert, 'NA')

                cat_missing = new_df[selected_cat_column].isnull().sum()

                statistics = {
                    'Distinct Values': new_df[selected_cat_column].nunique(),
                    'Null Values': cat_missing
                }
                statistics_df = pd.DataFrame(statistics, index=[selected_cat_column]).T

                st.write("Overview for", selected_cat_column)
                st.write(statistics_df)

                value_counts_df = pd.DataFrame(new_df[selected_cat_column].value_counts()).rename(columns={selected_cat_column: 'Value Count'})
                st.write("Value Counts for", selected_cat_column)
                st.write(value_counts_df)


    code4 = '''
    # Scatter Plots of Variables vs. Sales

    for column in advertising.columns:
        if column != 'Sales':
            fig = px.scatter(advertising, x=column, y='Sales', 
            title=f'{column} vs. Sales')
            st.plotly_chart(fig)

    '''

    st.markdown("```python\n" + code4 + "\n```")



    st.subheader("Scatter Plots of Variables vs. Sales")
    for column in advertising.columns:
        if column != 'Sales':
            fig = px.scatter(advertising, x=column, y='Sales', title=f'{column} vs. Sales')
            st.plotly_chart(fig)



    code5 = '''
    # Split data into training and test sets

    X = advertising['TV']
    y = advertising['Sales']
    X_train, X_test, y_train, y_test = train_test_split
                            (X,y, train_size=0.7, test_size=0.3, random_state=100)
    '''

    st.markdown("```python\n" + code5 + "\n```")


    # Split data into training and test sets
    X = advertising['TV']
    y = advertising['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


    code6 = '''
    # Add a constant to X_train and fit the regression model

    X_train_sm = sm.add_constant(X_train)
    lr = sm.OLS(y_train, X_train_sm).fit()
    '''

    st.markdown("```python\n" + code6 + "\n```")

    X_train_sm = sm.add_constant(X_train)
    lr = sm.OLS(y_train, X_train_sm).fit()

    code7 = '''
    # Scatter Plot with Fitted Line

    fig = px.scatter(x=X_train, y=y_train, title='Scatter Plot with Fitted Line')
    fig.add_trace(px.line(x=X_train, y=lr.predict(X_train_sm), 
                                line_shape='linear').data[0])
    st.plotly_chart(fig)
    '''

    st.markdown("```python\n" + code7 + "\n```")


    fig = px.scatter(x=X_train, y=y_train, title='Scatter Plot with Fitted Line')
    fig.add_trace(px.line(x=X_train, y=lr.predict(X_train_sm), line_shape='linear').data[0])
    st.plotly_chart(fig)

    code8 = '''
    # Add a constant to X_test and predict y values

    X_test_sm = sm.add_constant(X_test)
    y_pred = lr.predict(X_test_sm)

    '''

    st.markdown("```python\n" + code8 + "\n```")

    # Add a constant to X_test and predict y values
    X_test_sm = sm.add_constant(X_test)
    y_pred = lr.predict(X_test_sm)

    code9 = '''
    # Calculate RMSE and R-squared values

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    '''

    st.markdown("```python\n" + code9 + "\n```")

    # Calculate RMSE and R-squared values
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.success(f"RMSE: {rmse:.2f}")
    st.success(f"R-squared: {r2:.2f}")

    # Reshape data for sklearn LinearRegression
    X_train_lm = X_train.values.reshape(-1, 1)
    X_test_lm = X_test.values.reshape(-1, 1)


    code10 = '''
    # Fit the Linear Regression model

    lr = LinearRegression()
    lr.fit(X_train_lm, y_train)

    # Reshape data for sklearn LinearRegression

    X_train_lm = X_train.values.reshape(-1, 1)
    X_test_lm = X_test.values.reshape(-1, 1)

    '''

    st.markdown("```python\n" + code10 + "\n```")

    # Reshape data for sklearn LinearRegression
    X_train_lm = X_train.values.reshape(-1, 1)
    X_test_lm = X_test.values.reshape(-1, 1)

    # Fit the Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train_lm, y_train)

    # Display intercept and slope
    st.success(f"Intercept: {lr.intercept_:.2f}")
    st.success(f"Slope: {lr.coef_[0]:.2f}")


    st.header("Residuals")

    code11 = '''

    # Calculate residuals
    residuals = y_train - lr.predict(X_train_sm)

    # Plot residuals vs. fitted values

    fig_residuals_fitted = px.scatter(x=lr.predict(X_train_sm), y=residuals, title='Residuals vs. Fitted Values')
    fig_residuals_fitted.add_shape(type='line', yref='y', y0=0, y1=0, x0=0, x1=max(lr.predict(X_train_sm)), line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_residuals_fitted)

    '''
    st.markdown("```python\n" + code11 + "\n```")



    # Split data into training and test sets
    X = advertising['TV']
    y = advertising['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

    # Add a constant to X_train and fit the regression model
    X_train_sm = sm.add_constant(X_train)
    lr = sm.OLS(y_train, X_train_sm).fit()

    # Calculate residuals
    residuals = y_train - lr.predict(X_train_sm)


    fig_residuals_fitted = px.scatter(x=lr.predict(X_train_sm), y=residuals, title='Residuals vs. Fitted Values')
    fig_residuals_fitted.add_shape(type='line', yref='y', y0=0, y1=0, x0=0, x1=max(lr.predict(X_train_sm)), line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_residuals_fitted)

    code12 = '''
    # Histogram for Residuals

    fig_histogram = px.histogram(residuals, nbins=15, title='Histogram of Residuals')
    st.plotly_chart(fig_histogram)
    '''
    st.markdown("```python\n" + code12 + "\n```")

    fig_histogram = px.histogram(residuals, nbins=15, title='Histogram of Residuals')
    st.plotly_chart(fig_histogram)

if st.sidebar.button(" Multiple Linear Regression"):

    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import plotly.express as px
    import plotly.subplots as sp
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots




    st.title("Multiple Linear Regression")

    code1 = """
    # Importing all libraries

    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import plotly.express as px
    import plotly.subplots as sp
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    # Loading the dataset

    data = pd.read_csv(r'Housing.csv')
    """
    st.markdown("```python\n" + code1 + "\n```")


    data = pd.read_csv(r'Housing.csv')
    st.subheader("Dataset")
    st.write(data)

    code2 = """
    # Data Overview 

    data.describe()
    """
    st.markdown("```python\n" + code2 + "\n```")

    object_columns = data.select_dtypes(include=['object','bool']).columns.tolist()
    num_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()

    col1, col2 = st.columns(2)

    if len(num_columns) > 0:
        with col1:
            selected_num_column = st.selectbox('Select a Numerical column', num_columns)
            if selected_num_column:
                neg = []
                zeros = []

                statistics = data[selected_num_column].describe()

                for val in data[selected_num_column]:
                    if val < 0:
                        neg.append(1)
                    if val == 0:
                        zeros.append(1)

                statistics["Value count"] = statistics.pop("count")
                statistics["Mean"] = statistics.pop("mean")
                statistics.pop("std")
                statistics["Minimum"] = statistics.pop("min")
                statistics["Quartile 1 (25%)"] = statistics.pop("25%")
                statistics["Quartile 2 (50%)"] = statistics.pop("50%")
                statistics["Quartile 3 (75%)"] = statistics.pop("75%")
                statistics["Maximum"] = statistics.pop("max")
                statistics["Negative Values"] = sum(neg)
                statistics["Zero Values"] = sum(zeros)
                statistics["Null Values"] = data[selected_num_column].isna().sum()

                statistics_df = pd.DataFrame(statistics)

                st.write("Overview for", selected_num_column)
                st.write(statistics_df)

    if len(object_columns) > 0:
        with col2:
            selected_cat_column = st.selectbox('Select a Categorical column', object_columns)
            if selected_cat_column:
                values_to_convert = ['na', 'not applicable']
                new_df = data.copy()
                new_df[selected_cat_column] = new_df[selected_cat_column].str.lower().replace(values_to_convert, 'NA')

                cat_missing = new_df[selected_cat_column].isnull().sum()

                statistics = {
                    'Distinct Values': new_df[selected_cat_column].nunique(),
                    'Null Values': cat_missing
                }
                statistics_df = pd.DataFrame(statistics, index=[selected_cat_column]).T

                st.write("Overview for", selected_cat_column)
                st.write(statistics_df)

                value_counts_df = pd.DataFrame(new_df[selected_cat_column].value_counts()).rename(columns={selected_cat_column: 'Value Count'})
                st.write("Value Counts for", selected_cat_column)
                st.write(value_counts_df)

    st.write("")
    st.subheader("Outliear detection")

    code3 = """
    # Create subplots using Plotly's make_subplots
    fig = sp.make_subplots(rows=2, cols=3, subplot_titles=("Price", "Area", "Bedrooms", "Bathrooms", "Stories", "Parking"))

    # Add box plots to each subplot
    fig.add_trace(go.Box(x=data['price'], name='Price'), row=1, col=1)
    fig.add_trace(go.Box(x=data['area'], name='Area'), row=1, col=2)
    fig.add_trace(go.Box(x=data['bedrooms'], name='Bedrooms'), row=1, col=3)
    fig.add_trace(go.Box(x=data['bathrooms'], name='Bathrooms'), row=2, col=1)
    fig.add_trace(go.Box(x=data['stories'], name='Stories'), row=2, col=2)
    fig.add_trace(go.Box(x=data['parking'], name='Parking'), row=2, col=3)

    # Update layout
    fig.update_layout(height=600, width=800, showlegend=False)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    """
    st.markdown("```python\n" + code3 + "\n```")



    def detectOutliears():
        # Create subplots using Plotly's make_subplots
        fig = sp.make_subplots(rows=2, cols=3, subplot_titles=("Price", "Area", "Bedrooms", "Bathrooms", "Stories", "Parking"))

        # Add box plots to each subplot
        fig.add_trace(go.Box(x=data['price'], name='Price'), row=1, col=1)
        fig.add_trace(go.Box(x=data['area'], name='Area'), row=1, col=2)
        fig.add_trace(go.Box(x=data['bedrooms'], name='Bedrooms'), row=1, col=3)
        fig.add_trace(go.Box(x=data['bathrooms'], name='Bathrooms'), row=2, col=1)
        fig.add_trace(go.Box(x=data['stories'], name='Stories'), row=2, col=2)
        fig.add_trace(go.Box(x=data['parking'], name='Parking'), row=2, col=3)

        # Update layout
        fig.update_layout(height=600, width=800, showlegend=False)

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)

    detectOutliears()

    st.write("")
    st.subheader("Outlier Reduction")
    code4 = """
    # Create a subplot for outlier reduction of 'price'
    fig_price = px.box(data, x='price', title='Outlier Reduction for Price')
    fig_price.update_traces(marker=dict(size=2), selector=dict(type='box'))

    st.plotly_chart(fig_price)

    # Calculate quartiles and IQR for 'price'
    Q1_price = data['price'].quantile(0.25)
    Q3_price = data['price'].quantile(0.75)
    IQR_price = Q3_price - Q1_price

    # Filter data based on outlier boundaries for 'price'
    data = data[(data['price'] >= Q1_price - 1.5 * IQR_price) & (data['price'] <= Q3_price + 1.5 * IQR_price)]

    # Create a subplot for outlier reduction of 'area'
    fig_area = px.box(data, x='area', title='Outlier Reduction for Area')
    fig_area.update_traces(marker=dict(size=2), selector=dict(type='box'))

    st.plotly_chart(fig_area)

    # Calculate quartiles and IQR for 'area'
    Q1_area = data['area'].quantile(0.25)
    Q3_area = data['area'].quantile(0.75)
    IQR_area = Q3_area - Q1_area

    # Filter data based on outlier boundaries for 'area'
    data = data[(data['area'] >= Q1_area - 1.5 * IQR_area) & (data['area'] <= Q3_area + 1.5 * IQR_area)]

    """
    st.markdown("```python\n" + code4 + "\n```")

    # Create a subplot for outlier reduction of 'price'
    fig_price = px.box(data, x='price', title='Outlier Reduction for Price')
    fig_price.update_traces(marker=dict(size=2), selector=dict(type='box'))

    st.plotly_chart(fig_price)

    # Calculate quartiles and IQR for 'price'
    Q1_price = data['price'].quantile(0.25)
    Q3_price = data['price'].quantile(0.75)
    IQR_price = Q3_price - Q1_price

    # Filter data based on outlier boundaries for 'price'
    data = data[(data['price'] >= Q1_price - 1.5 * IQR_price) & (data['price'] <= Q3_price + 1.5 * IQR_price)]

    # Create a subplot for outlier reduction of 'area'
    fig_area = px.box(data, x='area', title='Outlier Reduction for Area')
    fig_area.update_traces(marker=dict(size=2), selector=dict(type='box'))

    st.plotly_chart(fig_area)

    # Calculate quartiles and IQR for 'area'
    Q1_area = data['area'].quantile(0.25)
    Q3_area = data['area'].quantile(0.75)
    IQR_area = Q3_area - Q1_area

    # Filter data based on outlier boundaries for 'area'
    data = data[(data['area'] >= Q1_area - 1.5 * IQR_area) & (data['area'] <= Q3_area + 1.5 * IQR_area)]

    detectOutliears()


    code5 = """
    # Pair Plot using Plotly Express
    fig_pair = px.scatter_matrix(data, title="Pair Plot")
    fig_pair.update_layout(height=800, width=800)  # Set larger size
    st.plotly_chart(fig_pair)

    """
    st.markdown("```python\n" + code5 + "\n```")

    # Pair Plot using Plotly Express
    fig_pair = px.scatter_matrix(data, title="Pair Plot")
    fig_pair.update_layout(height=800, width=800)  # Set larger size
    st.plotly_chart(fig_pair)

    code6 = """

    # Create subplots for box plots
    fig_box = make_subplots(rows=2, cols=3, subplot_titles=("Mainroad", "Guestroom", "Basement", "Hotwaterheating", "Airconditioning", "Furnishingstatus"))
    row, col = 1, 1

    # Loop through categorical columns for box plots
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus']
    for column in categorical_columns:
        box_fig = px.box(data, x=column, y='price', title=f"Box Plot of {column.capitalize()} vs. Price")
        box_fig.update_layout(height=500, width=500)  # Set larger size
        fig_box.add_trace(box_fig.data[0], row=row, col=col)
        fig_box.update_xaxes(title_text=column.capitalize(), row=row, col=col)
        fig_box.update_yaxes(title_text="Price", row=row, col=col)
        col += 1
        if col > 3:
            col = 1
            row += 1

    # Update subplot layout
    fig_box.update_layout(title="Box Plots", showlegend=False)

    st.plotly_chart(fig_box)

    """
    st.markdown("```python\n" + code6 + "\n```")

    # Create subplots for box plots
    fig_box = make_subplots(rows=2, cols=3, subplot_titles=("Mainroad", "Guestroom", "Basement", "Hotwaterheating", "Airconditioning", "Furnishingstatus"))
    row, col = 1, 1

    # Loop through categorical columns for box plots
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus']
    for column in categorical_columns:
        box_fig = px.box(data, x=column, y='price', title=f"Box Plot of {column.capitalize()} vs. Price")
        box_fig.update_layout(height=700, width=400)  # Set larger size
        fig_box.add_trace(box_fig.data[0], row=row, col=col)
        fig_box.update_xaxes(title_text=column.capitalize(), row=row, col=col)
        fig_box.update_yaxes(title_text="Price", row=row, col=col)
        col += 1
        if col > 3:
            col = 1
            row += 1

    # Update subplot layout
    fig_box.update_layout(title="Box Plots", showlegend=False)

    st.plotly_chart(fig_box)

    st.write("")
    st.subheader("Dummy variable")

    code7 = """
    def toNumeric(x):
        return x.map({"no":0,"yes":1})
    def convert_binary():     
        for column in list(data.select_dtypes(['object']).columns):
            if(column != 'furnishingstatus'):
                data[[column]] = data[[column]].apply(toNumeric)
    convert_binary()


    status = pd.get_dummies(data['furnishingstatus'])
    st.write(status)
    status = pd.get_dummies(data['furnishingstatus'], drop_first=True)
    data = pd.concat([data, status], axis=1)
    data.drop(columns='furnishingstatus',inplace=True)
    st.write(data)

    """
    st.markdown("```python\n" + code7 + "\n```")

    def toNumeric(x):
        return x.map({"no":0,"yes":1})
    def convert_binary():     
        for column in list(data.select_dtypes(['object']).columns):
            if(column != 'furnishingstatus'):
                data[[column]] = data[[column]].apply(toNumeric)
    convert_binary()



    status = pd.get_dummies(data['furnishingstatus'])
    st.write("")
    st.write(status)
    status = pd.get_dummies(data['furnishingstatus'], drop_first=True)
    data = pd.concat([data, status], axis=1)
    data.drop(columns='furnishingstatus',inplace=True)
    st.write(data)


    st.write("")
    st.subheader("Pre Processing")

    code8 = """

    Y = data.price
    # includes the fields other than prices
    X = data.iloc[:,1:]
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    def preprocessing(X):    
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        variables = X_scaled
        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
        vif["Features"] = X.columns
        st.write(vif)

    preprocessing(X)

    X.drop(['area','bedrooms'], axis=1, inplace=True)
    preprocessing(X)

    """
    st.markdown("```python\n" + code8 + "\n```")


    Y = data.price
    # includes the fields other than prices
    X = data.iloc[:,1:]
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    def preprocessing(X):    
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        variables = X_scaled
        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
        vif["Features"] = X.columns
        st.write(vif)

    st.write("")
    st.write("Before")
    preprocessing(X)

    X.drop(['area','bedrooms'], axis=1, inplace=True)
    st.write("")
    st.write("After")
    preprocessing(X)


    st.write("")
    st.subheader("Model fitting")

    code9 = """

    # Model fitting and output

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state=355)

    from sklearn.linear_model import LinearRegression
    regression = LinearRegression()
    regression.fit(x_train,y_train)

    y_predict = regression.predict(x_test)

    # Scatter plot
    fig = px.scatter(x=y_test, y=y_predict, labels={'x': 'y_test', 'y': 'y_pred'})
    fig.update_layout(title='y_test vs y_pred', xaxis_title='y_test', yaxis_title='y_pred')
    st.plotly_chart(fig)



    """
    st.markdown("```python\n" + code9 + "\n```")

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state=355)

    from sklearn.linear_model import LinearRegression
    regression = LinearRegression()
    regression.fit(x_train,y_train)

    y_predict = regression.predict(x_test)

    # Scatter plot
    fig = px.scatter(x=y_test, y=y_predict, labels={'x': 'y_test', 'y': 'y_pred'})
    fig.update_layout(title='y_test vs y_pred', xaxis_title='y_test', yaxis_title='y_pred')
    st.plotly_chart(fig)



if st.sidebar.button("Logistic Regression"):

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import plotly.express as px
    from sklearn.metrics import confusion_matrix
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve



    st.title("Logistic Regression")


    code1 = """
    # Import necessary libraries

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import plotly.express as px
    from sklearn.metrics import confusion_matrix
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve



    # data = pd.read_csv(r'/Applications/ML/marketing.csv')
    st.subheader("Dataset")
    st.write(data)
    """
    st.markdown("```python\n" + code1 + "\n```")

    data = pd.read_csv(r'marketing.csv')
    st.subheader("Dataset")
    st.write(data)

    code2 = """
    # Data Overview
    data.describe()
    """

    st.markdown("```python\n" + code2 + "\n```")

    st.write("")
    st.subheader("Data Overview")
    object_columns = data.select_dtypes(include=['object','bool']).columns.tolist()
    num_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()

    col1, col2 = st.columns(2)

    if len(num_columns) > 0:
        with col1:
            selected_num_column = st.selectbox('Select a Numerical column', num_columns)
            if selected_num_column:
                neg = []
                zeros = []

                statistics = data[selected_num_column].describe()

                for val in data[selected_num_column]:
                    if val < 0:
                        neg.append(1)
                    if val == 0:
                        zeros.append(1)

                statistics["Value count"] = statistics.pop("count")
                statistics["Mean"] = statistics.pop("mean")
                statistics.pop("std")
                statistics["Minimum"] = statistics.pop("min")
                statistics["Quartile 1 (25%)"] = statistics.pop("25%")
                statistics["Quartile 2 (50%)"] = statistics.pop("50%")
                statistics["Quartile 3 (75%)"] = statistics.pop("75%")
                statistics["Maximum"] = statistics.pop("max")
                statistics["Negative Values"] = sum(neg)
                statistics["Zero Values"] = sum(zeros)
                statistics["Null Values"] = data[selected_num_column].isna().sum()

                statistics_df = pd.DataFrame(statistics)

                st.write("Overview for", selected_num_column)
                st.write(statistics_df)

    if len(object_columns) > 0:
        with col2:
            selected_cat_column = st.selectbox('Select a Categorical column', object_columns)
            if selected_cat_column:
                values_to_convert = ['na', 'not applicable']
                new_df = data.copy()
                new_df[selected_cat_column] = new_df[selected_cat_column].str.lower().replace(values_to_convert, 'NA')

                cat_missing = new_df[selected_cat_column].isnull().sum()

                statistics = {
                    'Distinct Values': new_df[selected_cat_column].nunique(),
                    'Null Values': cat_missing
                }
                statistics_df = pd.DataFrame(statistics, index=[selected_cat_column]).T

                st.write("Overview for", selected_cat_column)
                st.write(statistics_df)

                value_counts_df = pd.DataFrame(new_df[selected_cat_column].value_counts()).rename(columns={selected_cat_column: 'Value Count'})
                st.write("Value Counts for", selected_cat_column)
                st.write(value_counts_df)


    st.write("")
    st.subheader("EDA")

    code3 = """
    # Count the occurrences of each category in the 'job' column
    job_counts = data['job'].value_counts()

    # Create a bar plot 
    fig = px.bar(job_counts, x=job_counts.index, y=job_counts.values, title='Purchase Frequency for Job Title')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Job',
        yaxis_title='Frequency of Purchase',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)
    """

    st.markdown("```python\n" + code3 + "\n```")

    # Count the occurrences of each category in the 'job' column
    job_counts = data['job'].value_counts()

    # Create a bar plot using Plotly Express
    fig = px.bar(job_counts, x=job_counts.index, y=job_counts.values, title='Purchase Frequency for Job Title')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Job',
        yaxis_title='Frequency of Purchase',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    code4 = """
    # Create a cross-tabulation table
    table = pd.crosstab(data['marital'], data['y'])

    # Calculate proportions
    table_proportions = table.div(table.sum(1), axis=0)

    # Create a stacked bar plot 
    fig = px.bar(table_proportions, x=table_proportions.index, y=table_proportions.columns,
                title='Stacked Bar Chart of Marital Status vs Purchase', barmode='stack')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Marital Status',
        yaxis_title='Proportion of Customers',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    """

    st.markdown("```python\n" + code4 + "\n```")

    # Create a cross-tabulation table
    table = pd.crosstab(data['marital'], data['y'])

    # Calculate proportions
    table_proportions = table.div(table.sum(1), axis=0)

    # Create a stacked bar plot using Plotly Express
    fig = px.bar(table_proportions, x=table_proportions.index, y=table_proportions.columns,
                title='Stacked Bar Chart of Marital Status vs Purchase', barmode='stack')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Marital Status',
        yaxis_title='Proportion of Customers',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)



    code5 = """

    # Create a cross-tabulation table
    table = pd.crosstab(data['education'], data['y'])

    # Calculate proportions
    table_proportions = table.div(table.sum(1), axis=0)

    # Create a stacked bar plot 
    fig = px.bar(table_proportions, x=table_proportions.index, y=table_proportions.columns,
                title='Stacked Bar Chart of Education vs Purchase', barmode='stack')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Education',
        yaxis_title='Proportion of Customers',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    """

    st.markdown("```python\n" + code5 + "\n```")

    # Create a cross-tabulation table
    table = pd.crosstab(data['education'], data['y'])

    # Calculate proportions
    table_proportions = table.div(table.sum(1), axis=0)

    # Create a stacked bar plot using Plotly Express
    fig = px.bar(table_proportions, x=table_proportions.index, y=table_proportions.columns,
                title='Stacked Bar Chart of Education vs Purchase', barmode='stack')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Education',
        yaxis_title='Proportion of Customers',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    code6 = """
    # Create a cross-tabulation table
    table = pd.crosstab(data['day_of_week'], data['y'])

    # Create a bar plot
    fig = px.bar(table, x=table.index, y=table.columns,
                title='Purchase Frequency for Day of Week')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Frequency of Purchase',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    """

    st.markdown("```python\n" + code6 + "\n```")


    # Create a cross-tabulation table
    table = pd.crosstab(data['day_of_week'], data['y'])

    # Create a bar plot using Plotly Express
    fig = px.bar(table, x=table.index, y=table.columns,
                title='Purchase Frequency for Day of Week')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Frequency of Purchase',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    code7 = """

    # Create a cross-tabulation table
    table = pd.crosstab(data['month'], data['y'])

    # Create a bar plot 
    fig = px.bar(table, x=table.index, y=table.columns,
                title='Purchase Frequency for Month')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Frequency of Purchase',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    """

    st.markdown("```python\n" + code7 + "\n```")

    # Create a cross-tabulation table
    table = pd.crosstab(data['month'], data['y'])

    # Create a bar plot using Plotly Express
    fig = px.bar(table, x=table.index, y=table.columns,
                title='Purchase Frequency for Month')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Frequency of Purchase',
        xaxis=dict(type='category'),  # Ensure categorical data is recognized
    )

    # Display the plot 
    st.plotly_chart(fig)

    code8 = """

    # Create a histogram 
    fig = px.histogram(data, x='age', title='Histogram of Age',
                    labels={'age': 'Age', 'count': 'Frequency'})

    # Display the plot 
    st.plotly_chart(fig)

    """

    st.markdown("```python\n" + code8 + "\n```")

    # Create a histogram using Plotly Express
    fig = px.histogram(data, x='age', title='Histogram of Age',
                    labels={'age': 'Age', 'count': 'Frequency'})

    # Display the plot using Streamlit
    st.plotly_chart(fig)

    code9 = """

    # Create a bar plot 
    fig = px.bar(data, x='poutcome', title='Purchase Frequency for Poutcome',
                labels={'poutcome': 'Poutcome', 'count': 'Frequency'})

    # Display the plot 
    st.plotly_chart(fig)

    """
    st.markdown("```python\n" + code9 + "\n```")

    # Create a bar plot using Plotly Express
    fig = px.bar(data, x='poutcome', title='Purchase Frequency for Poutcome',
                labels={'poutcome': 'Poutcome', 'count': 'Frequency'})

    # Display the plot using Streamlit
    st.plotly_chart(fig)

    st.write("")
    st.subheader("Preprocessing")

    code10 = """
    # Dummpy variables

    cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(data[var], prefix=var)
        data1=data.join(cat_list)
        data=data1
    cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    data_vars=data.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    data_final=data[to_keep]
    st.write(data_final.head(20))

    """
    st.markdown("```python\n" + code10 + "\n```")

    cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(data[var], prefix=var)
        data1=data.join(cat_list)
        data=data1
    cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    data_vars=data.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    data_final=data[to_keep]
    st.write("")
    st.write(data_final.head(20))


    st.write("")
    st.subheader("Oversampled Data Information")

    code11 = """
    # Separate features and target variable
    X = data_final.loc[:, data_final.columns != 'y']
    y = data_final.loc[:, data_final.columns == 'y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Apply SMOTE to the training data
    os = SMOTE(random_state=0)
    os_data_X, os_data_y = os.fit_resample(X_train, y_train)

    # Convert oversampled data to DataFrames
    columns = X_train.columns
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

    st.subheader(" Oversampled Data Information")
    st.success(f"Length of oversampled data: {len(os_data_X)}")
    st.success(f"Number of no subscription in oversampled data: {len(os_data_y[os_data_y['y'] == 0])}")
    st.success(f"Number of subscription in oversampled data: {len(os_data_y[os_data_y['y'] == 1])}")
    st.success(f"Proportion of no subscription data in oversampled data: {len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X):.2f}")
    st.success(f"Proportion of subscription data in oversampled data: {len(os_data_y[os_data_y['y'] == 1]) / len(os_data_X):.2f}")

    """
    st.markdown("```python\n" + code11 + "\n```")



    # Separate features and target variable
    X = data_final.loc[:, data_final.columns != 'y']
    y = data_final.loc[:, data_final.columns == 'y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Apply SMOTE to the training data
    os = SMOTE(random_state=0)
    os_data_X, os_data_y = os.fit_resample(X_train, y_train)

    # Convert oversampled data to DataFrames
    columns = X_train.columns
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])


    st.success(f"Length of oversampled data: {len(os_data_X)}")
    st.success(f"Number of no subscription in oversampled data: {len(os_data_y[os_data_y['y'] == 0])}")
    st.success(f"Number of subscription in oversampled data: {len(os_data_y[os_data_y['y'] == 1])}")
    st.success(f"Proportion of no subscription data in oversampled data: {len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X):.2f}")
    st.success(f"Proportion of subscription data in oversampled data: {len(os_data_y[os_data_y['y'] == 1]) / len(os_data_X):.2f}")

    st.subheader("Model fitting and accuracy")
    code12 = """

    # Model fitting and accuracy

    cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
        'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
        'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
    X=os_data_X[cols]
    y=os_data_y['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    st.success('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    """
    st.markdown("```python\n" + code12 + "\n```")

    cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
        'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
        'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
    X=os_data_X[cols]
    y=os_data_y['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    st.success('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


    st.subheader(" Confusion Matrix ")
    code12 = """

    # Confusion Matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    st.success(confusion_matrix)

    # Create a DataFrame for the confusion matrix
    cm_matrix = pd.DataFrame(data=confusion_matrix, columns=['Actual Positive:1', 'Actual Negative:0'], 
                            index=['Predict Positive:1', 'Predict Negative:0'])

    # Create a black background figure
    plt.figure(figsize=(8, 6), facecolor='black')

    # Create a heatmap using Seaborn with custom color map
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

    # Set the title, xlabel, and ylabel properties
    plt.title('Confusion Matrix Heatmap', color='white')
    plt.xlabel('Actual', color='white')
    plt.ylabel('Predicted', color='white')

    # Set the color of annotations to white
    plt.rcParams['text.color'] = 'white'

    # Display the heatmap in Streamlit
    st.pyplot(plt)

    """
    st.markdown("```python\n" + code12 + "\n```")
    confusion_matrix = confusion_matrix(y_test, y_pred)
    st.success(confusion_matrix)

    # Create a DataFrame for the confusion matrix
    cm_matrix = pd.DataFrame(data=confusion_matrix, columns=['Actual Positive:1', 'Actual Negative:0'], 
                            index=['Predict Positive:1', 'Predict Negative:0'])

    # Create a black background figure
    plt.figure(figsize=(8, 6), facecolor='black')

    # Create a heatmap using Seaborn with custom color map
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

    # Set the title, xlabel, and ylabel properties
    plt.title('Confusion Matrix Heatmap', color='white')
    plt.xlabel('Actual', color='white')
    plt.ylabel('Predicted', color='white')

    # Set the color of annotations to white
    plt.rcParams['text.color'] = 'white'

    # Display the heatmap in Streamlit
    st.pyplot(plt)

    st.subheader("Classification Report")

    code13 = """

    # Calculate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert the report dictionary to a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Display the classification report as a table
    st.table(report_df)

    """
    st.markdown("```python\n" + code13 + "\n```")

    # Calculate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert the report dictionary to a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Display the classification report as a table
    st.table(report_df)


    st.subheader("ROC Curve")

    code14 = """

    # ROC curve: 

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # Display the matplotlib figure using st.pyplot()
    st.pyplot(plt)


    """
    st.markdown("```python\n" + code14 + "\n```")

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # Display the matplotlib figure using st.pyplot()
    st.pyplot(plt)

if st.sidebar.button("Naive Bayes Classifier"):

    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.model_selection import train_test_split
    import warnings
    from sklearn.metrics import accuracy_score
    import category_encoders as ce
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report



    warnings.filterwarnings('ignore')

    st.title("Naive Bayes Classifier")

    code1 = """

    # Import necessary libraries

    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.model_selection import train_test_split
    import warnings
    from sklearn.metrics import accuracy_score
    import category_encoders as ce
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report


    # Load the data
    data = pd.read_csv(r'/Applications/ML/adult.csv')
    st.subheader("Dataset")
    st.write(data)

    """
    st.markdown("```python\n" + code1 + "\n```")

    data = pd.read_csv(r'adult.csv')
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

    data.columns = col_names
    st.subheader("Dataset")
    st.write(data)

    code2 = """
    # Data preprocessing

    data['workclass'] = data['workclass'].str.strip().replace('?', np.nan)
    data['occupation'] = data['occupation'].str.strip().replace('?', np.nan)
    data['native_country'] = data['native_country'].str.strip().replace('?', np.nan)

    data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)
    data['occupation'].fillna(data['occupation'].mode()[0], inplace=True)
    data['native_country'].fillna(data['native_country'].mode()[0], inplace=True) 
    """

    st.markdown("```python\n" + code2 + "\n```")

    data['workclass'] = data['workclass'].str.strip().replace('?', np.nan)
    data['occupation'] = data['occupation'].str.strip().replace('?', np.nan)
    data['native_country'] = data['native_country'].str.strip().replace('?', np.nan)

    data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)
    data['occupation'].fillna(data['occupation'].mode()[0], inplace=True)
    data['native_country'].fillna(data['native_country'].mode()[0], inplace=True) 



    code3 = """
    # Data Overview
    data.describe()
    """

    st.markdown("```python\n" + code3 + "\n```")

    st.write("")
    st.subheader("Data Overview")

    def overview():
        object_columns = data.select_dtypes(include=['object','bool']).columns.tolist()
        num_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()

        col1, col2 = st.columns(2)

        if len(num_columns) > 0:
            with col1:
                selected_num_column = st.selectbox('Select a Numerical column', num_columns)
                if selected_num_column:
                    neg = []
                    zeros = []

                    statistics = data[selected_num_column].describe()

                    for val in data[selected_num_column]:
                        if val < 0:
                            neg.append(1)
                        if val == 0:
                            zeros.append(1)

                    statistics["Value count"] = statistics.pop("count")
                    statistics["Mean"] = statistics.pop("mean")
                    statistics.pop("std")
                    statistics["Minimum"] = statistics.pop("min")
                    statistics["Quartile 1 (25%)"] = statistics.pop("25%")
                    statistics["Quartile 2 (50%)"] = statistics.pop("50%")
                    statistics["Quartile 3 (75%)"] = statistics.pop("75%")
                    statistics["Maximum"] = statistics.pop("max")
                    statistics["Negative Values"] = sum(neg)
                    statistics["Zero Values"] = sum(zeros)
                    statistics["Null Values"] = data[selected_num_column].isna().sum()

                    statistics_df = pd.DataFrame(statistics)

                    st.write("Overview for", selected_num_column)
                    st.write(statistics_df)

        if len(object_columns) > 0:
            with col2:
                selected_cat_column = st.selectbox('Select a Categorical column', object_columns)
                if selected_cat_column:
                    values_to_convert = ['na', 'not applicable']
                    new_df = data.copy()
                    new_df[selected_cat_column] = new_df[selected_cat_column].str.lower().replace(values_to_convert, 'NA')

                    cat_missing = new_df[selected_cat_column].isnull().sum()

                    statistics = {
                        'Distinct Values': new_df[selected_cat_column].nunique(),
                        'Null Values': cat_missing
                    }
                    statistics_df = pd.DataFrame(statistics, index=[selected_cat_column]).T

                    st.write("Overview for", selected_cat_column)
                    st.write(statistics_df)

                    value_counts_df = pd.DataFrame(new_df[selected_cat_column].value_counts()).rename(columns={selected_cat_column: 'Value Count'})
                    st.write("Value Counts for", selected_cat_column)
                    st.write(value_counts_df)
    overview()


    st.subheader("Split the data")

    code4 = """

    # Split data into separate training and test set 

    X = data.drop(['income'], axis=1)

    y = data['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    X_train.shape, X_test.shape

    """
    st.markdown("```python\n" + code4 + "\n```")


    X = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    suc = X_train.shape, X_test.shape
    st.success(suc)

    st.subheader("Encoding")

    code5 = """

    # Encoding

    encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                    'race', 'sex', 'native_country'])

    """
    st.markdown("```python\n" + code5 + "\n```")

    encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                    'race', 'sex', 'native_country'])

    X_train = encoder.fit_transform(X_train)

    code6= """

    # Encoding

    X_test = encoder.transform(X_test)

    st.write(X_train)
    st.success(X_train.shape)

    """
    st.markdown("```python\n" + code6 + "\n```")
    st.write("")
    st.write(X_train)
    st.success(X_train.shape)

    X_test = encoder.transform(X_test)

    code7 = """

    # Encoding

    X_test = encoder.transform(X_test)

    st.write(X_test)
    st.success(X_test.shape)

    """
    st.markdown("```python\n" + code7 + "\n```")
    st.write("")
    st.write(X_test)
    st.success(X_test.shape)

    st.subheader('Feature Scaling')

    code8 = """

    # Feature Scaling

    cols = X_train.columns
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    X_train.head()

    """
    st.markdown("```python\n" + code8 + "\n```")

    cols = X_train.columns
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    X_train.head()


    st.subheader('Model training')

    code9 = """

    # Model training

    # instantiate the model
    gnb = GaussianNB()

    # fit the model
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    y_pred
    """
    st.markdown("```python\n" + code9 + "\n```")


    # instantiate the model
    gnb = GaussianNB()

    # fit the model
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    st.success(y_pred)

    st.subheader("Model Accuracy")

    code10 = """

    # Model Accuracy

    st.success('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    y_pred_train = gnb.predict(X_train)

    st.success(y_pred_train)

    st.success('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

    st.success('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

    st.success('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

    """
    st.markdown("```python\n" + code10 + "\n```")


    st.success('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    y_pred_train = gnb.predict(X_train)

    st.success(y_pred_train)

    st.success('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

    st.success('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

    st.success('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

    st.subheader("Confusion Matrix")

    code11 = """
    # Confusion Matrix
    st.success('Confusion matrix' + str(cm))
    st.success('True Positives(TP) = ' + str(cm[0,0]))
    st.success('True Negatives(TN) = ' + str(cm[1,1]))
    st.success('False Positives(FP) = ' + str(cm[0,1]))
    st.success('False Negatives(FN) = ' + str(cm[1,0]))

    """
    st.markdown("```python\n" + code11 + "\n```")

    cm = confusion_matrix(y_test, y_pred)

    st.success('Confusion matrix\n' + str(cm))
    st.success('\nTrue Positives(TP) = ' + str(cm[0,0]))
    st.success('\nTrue Negatives(TN) = ' + str(cm[1,1]))
    st.success('\nFalse Positives(FP) = ' + str(cm[0,1]))
    st.success('\nFalse Negatives(FN) = ' + str(cm[1,0]))

    code11 = """
    # Confusion Matrix

    # Create a DataFrame for the confusion matrix
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                            index=['Predict Positive:1', 'Predict Negative:0'])

    # Create a black background figure
    plt.figure(figsize=(8, 6), facecolor='black')

    # Create a heatmap using Seaborn with custom color map
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

    # Set the title, xlabel, and ylabel properties
    plt.title('Confusion Matrix Heatmap', color='white')
    plt.xlabel('Actual', color='white')
    plt.ylabel('Predicted', color='white')

    # Set the color of annotations to white
    plt.rcParams['text.color'] = 'white'

    # Display the heatmap in Streamlit
    st.pyplot(plt)



    """
    st.markdown("```python\n" + code11 + "\n```")
    st.write("")

    # Create a DataFrame for the confusion matrix
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                            index=['Predict Positive:1', 'Predict Negative:0'])

    # Create a black background figure
    plt.figure(figsize=(8, 6), facecolor='black')

    # Create a heatmap using Seaborn with custom color map
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

    # Set the title, xlabel, and ylabel properties
    plt.title('Confusion Matrix Heatmap', color='white')
    plt.xlabel('Actual', color='white')
    plt.ylabel('Predicted', color='white')

    # Set the color of annotations to white
    plt.rcParams['text.color'] = 'white'

    # Display the heatmap in Streamlit
    st.pyplot(plt)

    code12 = """
    # Classification report
    print(classification_report(y_test, y_pred))
    """
    st.markdown("```python\n" + code12 + "\n```")
    st.write("")

    # Calculate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert the report dictionary to a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Display the classification report as a table
    st.table(report_df)

    st.subheader("Model Evaluation")
    code12 = """
    # Model Evaluation

    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    # print classification accuracy
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    st.success('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

    """
    st.markdown("```python\n" + code12 + "\n```")
    st.write("")

    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    # print classification accuracy
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    st.success('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

    code12 = """
    # Model Evaluation

    # print classification error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    st.success('Classification error : {0:0.4f}'.format(classification_error))
    """
    st.markdown("```python\n" + code12 + "\n```")

    # print classification error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    st.success('Classification error : {0:0.4f}'.format(classification_error))

    code12 = """
    # Model Evaluation

    # print precision score

    precision = TP / float(TP + FP)
    st.success('Precision : {0:0.4f}'.format(precision))
    """
    st.markdown("```python\n" + code12 + "\n```")

    # print precision score

    precision = TP / float(TP + FP)
    st.success('Precision : {0:0.4f}'.format(precision))

    code12 = """
    # Model Evaluation

    # print recall score

    recall = TP / float(TP + FN)
    st.success('Recall or Sensitivity : {0:0.4f}'.format(recall))
    """
    st.markdown("```python\n" + code12 + "\n```")

    # print recall score

    recall = TP / float(TP + FN)
    st.success('Recall or Sensitivity : {0:0.4f}'.format(recall))

    code12 = """
    # Model Evaluation

    # print true_positive_rate 

    true_positive_rate = TP / float(TP + FN)
    st.success('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

    """
    st.markdown("```python\n" + code12 + "\n```")


    # print true_positive_rate 
    true_positive_rate = TP / float(TP + FN)
    st.success('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

    code12 = """
    # Model Evaluation

    # print false_positive_rate 

    false_positive_rate = FP / float(FP + TN)
    st.success('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

    """
    st.markdown("```python\n" + code12 + "\n```")

    # print false_positive_rate 
    false_positive_rate = FP / float(FP + TN)
    st.success('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

    code12 = """
    # Model Evaluation

    # print specificity

    specificity = TN / (TN + FP)
    st.success('Specificity : {0:0.4f}'.format(specificity))

    """
    st.markdown("```python\n" + code12 + "\n```")

    # print specificity
    specificity = TN / (TN + FP)
    st.success('Specificity : {0:0.4f}'.format(specificity))




        
