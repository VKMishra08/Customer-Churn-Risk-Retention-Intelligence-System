from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
url='https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
out=ROOT/'data'/'Telco-Customer-Churn.csv'
out.parent.mkdir(exist_ok=True)
urllib.request.urlretrieve(url,out)
print(f'Downloaded: {out}')
