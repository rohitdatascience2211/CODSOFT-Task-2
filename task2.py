import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# Load the dataset with the specified encoding
file_path = r"C:\Users\prade\Downloads\IMDb Movies India.csv"
data = pd.read_csv(file_path, encoding='latin1')

# Assuming 'X_train' and 'y_train' are your feature and target variables
X_train = data.drop(columns=['Year'])  
y_train = data['Rating']               

# Drop rows with null values
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]  

# Identify categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Define transformers for numerical and categorical features
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ~X_train.columns.isin(categorical_features)),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Print success message
print("Pipeline fitted successfully.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

