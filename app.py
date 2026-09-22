import io
import streamlit as st
import pandas as pd
import plotly.express as px
from src.churn_engine import load_data, train, score, add_features

st.set_page_config(page_title='Customer Churn Risk & Retention Intelligence', page_icon='📊', layout='wide')

st.markdown('''<style>
.block-container{padding-top:1.2rem;max-width:1450px}.metric-card{padding:16px;border-radius:14px;background:#f7f8fa;border:1px solid #e5e7eb}.small{color:#64748b;font-size:13px}.hero{padding:24px;border-radius:18px;background:linear-gradient(135deg,#111827,#334155);color:white;margin-bottom:18px}.hero h1{margin:0 0 6px}.hero p{margin:0;color:#dbeafe}
</style>''', unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>Customer Churn Risk & Retention Intelligence System</h1><p>Predict churn risk, understand risk drivers, estimate revenue at risk, and prioritize retention actions.</p></div>', unsafe_allow_html=True)

df, source = load_data()
model, metrics = train(df)
scored = score(model, df)
view = add_features(scored)

with st.sidebar:
    st.header('Control Center')
    st.caption(source)
    risk_filter = st.multiselect('Risk level', ['High','Medium','Low'], default=['High','Medium','Low'])
    contract_filter = st.multiselect('Contract', sorted(df['Contract'].dropna().unique()), default=list(sorted(df['Contract'].dropna().unique())))
    search = st.text_input('Customer ID search')
    st.divider()
    st.subheader('Model')
    st.write(f"ROC-AUC: **{metrics['roc_auc']:.3f}**")
    st.write(f"Recall: **{metrics['recall']:.3f}**")
    st.caption('Threshold = 0.50 for classification; risk tiers use calibrated operational bands.')

filtered = scored[scored['RiskLevel'].astype(str).isin(risk_filter) & scored['Contract'].isin(contract_filter)].copy()
if search:
    filtered = filtered[filtered['customerID'].str.contains(search, case=False, na=False)]

# KPI row
c1,c2,c3,c4,c5 = st.columns(5)
churn_rate = (df['Churn'].eq('Yes').mean()*100)
high = int((scored['RiskLevel'].astype(str)=='High').sum())
rev_risk = float(scored['RevenueAtRisk'].sum())
for col,title,value,sub in [
    (c1,'Customers',f'{len(df):,}','Current portfolio'),
    (c2,'Observed churn',f'{churn_rate:.1f}%','Historical label in dataset'),
    (c3,'High-risk customers',f'{high:,}','Risk score ≥ 60'),
    (c4,'Revenue at risk',f'${rev_risk:,.0f}','Illustrative 6/3-month exposure'),
    (c5,'Model ROC-AUC',f'{metrics["roc_auc"]:.3f}','Holdout evaluation')]:
    col.markdown(f'<div class="metric-card"><div class="small">{title}</div><h2>{value}</h2><div class="small">{sub}</div></div>',unsafe_allow_html=True)

st.write('')
tab1,tab2,tab3,tab4,tab5 = st.tabs(['Executive Overview','Risk Analytics','Customer 360','Retention Actions','Model Performance'])

with tab1:
    a,b = st.columns(2)
    with a:
        x=view.groupby('RiskLevel',observed=False).size().reset_index(name='Customers')
        st.plotly_chart(px.bar(x,x='RiskLevel',y='Customers',title='Risk distribution'),use_container_width=True)
    with b:
        x=df.groupby('Contract').agg(Customers=('customerID','count'),ChurnRate=('Churn',lambda s:(s=='Yes').mean()*100)).reset_index()
        st.plotly_chart(px.bar(x,x='Contract',y='ChurnRate',title='Observed churn rate by contract',text_auto='.1f'),use_container_width=True)
    a,b = st.columns(2)
    with a:
        x=view.groupby('TenureBand',observed=False).agg(Customers=('customerID','count'),Risk=('ChurnProbability','mean')).reset_index(); x['Risk']=x['Risk']*100
        st.plotly_chart(px.line(x,x='TenureBand',y='Risk',markers=True,title='Average predicted churn risk by tenure band'),use_container_width=True)
    with b:
        x=view.groupby('InternetService').agg(Customers=('customerID','count'),Charges=('MonthlyCharges','mean')).reset_index()
        st.plotly_chart(px.bar(x,x='InternetService',y='Charges',title='Average monthly charges by internet service',text_auto='.2f'),use_container_width=True)

with tab2:
    a,b = st.columns(2)
    with a:
        st.plotly_chart(px.scatter(view.sample(min(1200,len(view)),random_state=42),x='tenure',y='MonthlyCharges',size='RiskScore',color='RiskLevel',hover_name='customerID',title='Customer risk map'),use_container_width=True)
    with b:
        x=view.groupby('RiskDriver').agg(Customers=('customerID','count'),RevenueAtRisk=('RevenueAtRisk','sum')).reset_index().sort_values('Customers',ascending=False)
        st.plotly_chart(px.bar(x.head(8),x='Customers',y='RiskDriver',orientation='h',title='Top operational risk drivers'),use_container_width=True)
    st.dataframe(filtered[['customerID','Contract','tenure','MonthlyCharges','ChurnProbability','RiskLevel','RiskDriver','RevenueAtRisk']].sort_values('ChurnProbability',ascending=False),use_container_width=True,hide_index=True)

with tab3:
    st.subheader('Customer-level risk profile')
    options = filtered['customerID'].tolist() if not filtered.empty else scored['customerID'].tolist()
    if options:
        cid=st.selectbox('Select customer',options)
        r=scored[scored['customerID']==cid].iloc[0]
        a,b,c,d=st.columns(4)
        a.metric('Risk score',f"{r.RiskScore:.1f}/100")
        b.metric('Risk level',str(r.RiskLevel))
        c.metric('Monthly charges',f"${r.MonthlyCharges:,.2f}")
        d.metric('Tenure',f"{int(r.tenure)} months")
        profile=pd.DataFrame({'Attribute':['Contract','Internet','Payment','Tech Support','Security','Partner','Dependents','Primary risk driver','Recommended action'], 'Value':[r.Contract,r.InternetService,r.PaymentMethod,r.TechSupport,r.OnlineSecurity,r.Partner,r.Dependents,r.RiskDriver,r.RetentionAction]})
        st.table(profile)

with tab4:
    st.subheader('Retention action queue')
    queue=view[view['RiskLevel'].astype(str).isin(['High','Medium'])].copy()
    queue['Priority'] = queue['RiskScore'] + queue['MonthlyCharges'].rank(pct=True)*10
    queue=queue.sort_values(['RiskLevel','Priority'],ascending=[True,False])
    st.dataframe(queue[['customerID','RiskLevel','RiskScore','MonthlyCharges','RevenueAtRisk','RiskDriver','RetentionAction']].head(100),use_container_width=True,hide_index=True)
    csv=queue.to_csv(index=False).encode()
    st.download_button('Download retention action queue',csv,'retention_action_queue.csv','text/csv')
    st.info('Retention recommendations are decision-support rules based on observable customer attributes; they do not guarantee that an action will prevent churn.')

with tab5:
    m=pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1','ROC-AUC'],'Score':[metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1'],metrics['roc_auc']]})
    st.dataframe(m,use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(m,x='Metric',y='Score',range_y=[0,1],title='Holdout model metrics',text_auto='.3f'),use_container_width=True)
    st.markdown('**Model:** Logistic Regression with class weighting, one-hot encoding for categorical variables, median imputation and standardization for numeric features.')

st.divider()
st.caption('Portfolio/educational implementation. Dataset: IBM Telco Customer Churn sample. Revenue-at-risk is an illustrative prioritization estimate, not an accounting forecast.')
