from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'Telco-Customer-Churn.csv'
DATA_URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

CATEGORICAL = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
NUMERIC = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
FEATURES = CATEGORICAL + NUMERIC


def make_demo_data(n=900, seed=42):
    rng = np.random.default_rng(seed)
    contracts = rng.choice(['Month-to-month','One year','Two year'], n, p=[.55,.24,.21])
    tenure = rng.integers(0,73,n)
    monthly = np.round(rng.uniform(20,120,n),2)
    internet = rng.choice(['DSL','Fiber optic','No'], n, p=[.34,.44,.22])
    payment = rng.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'], n, p=[.34,.22,.22,.22])
    senior = rng.binomial(1,.16,n)
    partner = rng.choice(['Yes','No'],n,p=[.48,.52])
    depend = rng.choice(['Yes','No'],n,p=[.30,.70])
    phone = rng.choice(['Yes','No'],n,p=[.90,.10])
    yesno = lambda p=.5: rng.choice(['Yes','No'],n,p=[p,1-p])
    no_phone = np.where(phone=='No','No phone service',yesno(.55))
    service = lambda p=.45: np.where(internet=='No','No internet service',yesno(p))
    total = np.round(np.maximum(0, monthly * tenure + rng.normal(0,120,n)),2)
    risk = (-1.0 + .9*(contracts=='Month-to-month') + .55*(internet=='Fiber optic') + .5*(payment=='Electronic check') + .65*(tenure<12) + .45*(monthly>80) + .35*senior + .25*(service(.25)=='No') + rng.normal(0,.35,n))
    prob = 1/(1+np.exp(-risk))
    churn = np.where(rng.random(n)<prob,'Yes','No')
    return pd.DataFrame({
        'customerID':[f'DEMO-{i:05d}' for i in range(1,n+1)], 'gender':rng.choice(['Male','Female'],n), 'SeniorCitizen':senior,
        'Partner':partner,'Dependents':depend,'tenure':tenure,'PhoneService':phone,'MultipleLines':no_phone,
        'InternetService':internet,'OnlineSecurity':service(.38),'OnlineBackup':service(.48),'DeviceProtection':service(.46),
        'TechSupport':service(.36),'StreamingTV':service(.48),'StreamingMovies':service(.48),'Contract':contracts,
        'PaperlessBilling':yesno(.59),'PaymentMethod':payment,'MonthlyCharges':monthly,'TotalCharges':total,'Churn':churn
    })


def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return clean(df), 'IBM Telco Customer Churn (local)'
    try:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        df = pd.read_csv(DATA_PATH)
        return clean(df), 'IBM Telco Customer Churn (downloaded from IBM GitHub)'
    except Exception:
        return clean(make_demo_data()), 'Demo fallback dataset (download unavailable)'


def clean(df):
    df = df.copy()
    if 'TotalCharges' in df:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop_duplicates(subset=['customerID'])
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    return df


def add_features(df):
    out = df.copy()
    service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    out['ServiceCount'] = sum((out[c].eq('Yes')).astype(int) for c in service_cols if c in out)
    out['TenureBand'] = pd.cut(out['tenure'], [-1,6,12,24,48,72], labels=['0-6m','7-12m','13-24m','25-48m','49-72m'])
    out['AvgMonthlyValue'] = np.where(out['tenure']>0, out['TotalCharges']/out['tenure'], out['MonthlyCharges'])
    return out


def build_pipeline():
    pre = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL)
    ])
    return Pipeline([('preprocessor',pre),('model',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42))])


def train(df):
    X = df[FEATURES]
    y = (df['Churn']=='Yes').astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    model = build_pipeline(); model.fit(Xtr,ytr)
    p = model.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
    metrics = {'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred),'recall':recall_score(yte,pred),'f1':f1_score(yte,pred),'roc_auc':roc_auc_score(yte,p)}
    return model, metrics


def score(model, df):
    out = df.copy()
    out['ChurnProbability'] = model.predict_proba(out[FEATURES])[:,1]
    out['RiskScore'] = (out['ChurnProbability']*100).round(1)
    out['RiskLevel'] = pd.cut(out['ChurnProbability'],[-.01,.35,.60,1.01],labels=['Low','Medium','High'])
    out['RevenueAtRisk'] = np.where(out['RiskLevel'].astype(str).eq('High'), out['MonthlyCharges']*6, np.where(out['RiskLevel'].astype(str).eq('Medium'),out['MonthlyCharges']*3,0))
    out['RiskDriver'] = out.apply(primary_driver, axis=1)
    out['RetentionAction'] = out.apply(retention_action, axis=1)
    return out


def primary_driver(r):
    if r['Contract']=='Month-to-month': return 'Month-to-month contract'
    if r['tenure']<=12: return 'Early-tenure customer'
    if r['PaymentMethod']=='Electronic check': return 'Electronic-check payment'
    if r['MonthlyCharges']>=80: return 'High monthly charges'
    if r['TechSupport']=='No': return 'No tech support'
    if r['OnlineSecurity']=='No': return 'No online security'
    return 'General churn risk'


def retention_action(r):
    if r['Contract']=='Month-to-month' and r['tenure']<=12: return 'Offer annual plan + onboarding support'
    if r['MonthlyCharges']>=80: return 'Review plan value + targeted loyalty offer'
    if r['TechSupport']=='No': return 'Offer technical-support trial'
    if r['PaymentMethod']=='Electronic check': return 'Promote automatic payment option'
    if r['OnlineSecurity']=='No': return 'Bundle security add-on with retention offer'
    return 'Send personalized retention outreach'
