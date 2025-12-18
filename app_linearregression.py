import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
#Page configuration
st.set_page_config("Simple Linear Regression",layout="centered")
# install requiremnts : pip install -r requirements.txt #

# load css #

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")
# Title #
st.markdown("""
            <div class="card">
                <h1>Simple Linear Regression</h1>
            <p>Predict <b> Tip Amount </b> from <b> Total Bill</b> using Linear Regression</p>
            </div>
            """,unsafe_allow_html=True)

# Load dataset #
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

#Dataset Preview #
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

#Prepare Data#
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Train Model#
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Metrics #
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

#visualization #
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip amount")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.5)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip Amount ($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#Performance Metrics #
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R-squared (R2)",f"{r2:.2f}")
c4.metric("Adjusted R-squared",f"{adj_r2:.2f}")
st.markdown('</div>',unsafe_allow_html=True)

#m and c #
st.markdown(f"""
            <div class="card">
                <h3>Model Interception</h3>
                <p><b>co-effecient : </b> {model.coef_[0]:.3f}<br>
                <b>intercept : </b> {model.intercept_:.3f}</p>
            </div>
            """,unsafe_allow_html=True)

#Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Tip Prediction")

total_bill = st.slider(
    "Select Total Bill Amount ($)",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),30.0
)

tip = model.predict(scaler.transform([[total_bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

