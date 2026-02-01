import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ML Model Evaluator",
    page_icon="📊",
    layout="wide"
)

# Title and Header
st.title("🤖 Machine Learning Model Evaluator")
st.markdown("### Upload your dataset and evaluate different classification models")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("⚙️ Configuration")

# a. Dataset upload option
st.sidebar.subheader("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    print(df.head())

    st.subheader("📋 Dataset Preview")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10))
    
    # Select target column
    # st.sidebar.subheader("🎯 Select Target Column")
    # target_column = st.sidebar.selectbox("Target Variable", df.columns)
   
    target_column = 'y'
    if target_column:
        # Prepare features and target
        X = df.drop(columns=['y'],axis=1)
        y = df['y']
        
        # Handle categorical variables in features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # b. Model selection dropdown
        st.sidebar.subheader("🔬 Select Model")
        model_option = st.sidebar.selectbox(
            "Choose a classification model",
            ["Logistic Regression", "DecisionTreeClassifier", "K-Nearest Neighbor", "Naive Bayes Classifier","Random Forest","XGBoost"]
        )

        # Train button
        if st.sidebar.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                # Select and train model
                if model_option == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                elif model_option == "DecisionTreeClassifier":
                    model = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
                elif model_option == "K-Nearest Neighbor":
                    model = KNeighborsClassifier(n_neighbors=5)
                elif model_option == "Naive Bayes Classifier":
                    model = GaussianNB()
                elif model_option == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = XGBClassifier(random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['model_name'] = model_option
                
                st.success(f"✅ {model_option} trained successfully!")
        
        # Display results if model is trained
        if 'model' in st.session_state:
            st.markdown("---")
            st.subheader(f"📊 Results for {st.session_state['model_name']}")
            
            # c. Display of evaluation metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            
            accuracy = accuracy_score(y_test, y_pred)
            rocaucscore = roc_auc_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("AUC Score", f"{rocaucscore:.3f}")
            with col3:
                st.metric("Precision", f"{precision:.3f}")
            with col4:
                st.metric("Recall", f"{recall:.3f}")
            with col5:
                st.metric("F1-Score", f"{f1:.3f}")
            with col6:
                mcc = (precision * recall) / (precision + recall + 1e-10)  # Simplified MCC calculation
                st.metric("MCC Score", f"{mcc:.3f}")
            
            st.markdown("---")
            
            # d. Confusion matrix and classification report
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("🔢 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                # Create heatmap using plotly
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred {i}" for i in range(len(cm))],
                    y=[f"True {i}" for i in range(len(cm))],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                ))
                
                fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                    width=400,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                st.subheader("📄 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"), height=400)

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file from the sidebar to get started!")
    
    st.markdown("""
    ### Instructions:
    1. **Upload your CSV dataset** using the sidebar
    2. **Select the target column** (the variable you want to predict)
    3. **Choose a classification model** from the dropdown
    4. **Click 'Predict and generate Metrics'** 
    
    ### Supported Models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Naive Bayes Classifier - Gaussian or Multinomial
    5. Ensemble Model - Random Forest
    6. Ensemble Model - XGBoost
    
    ### Evaluation Metrics:
    1. Accuracy
    2. AUC Score
    3. Precision
    4. Recall
    5. F1 Score
    6. Matthews Correlation Coeﬃcient (MCC Score)
    """)

# Footer
st.markdown("---")
st.markdown("**BITS Pilani** | Amitesh Choudhary Project Submission")