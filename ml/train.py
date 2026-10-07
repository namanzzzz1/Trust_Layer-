"""Reproducible PaySim benchmark. Usage: python ml/train.py --data path/to/paysim.zip"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, hashlib, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,confusion_matrix,roc_curve
ROOT=Path(__file__).resolve().parents[1]
TYPES=['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']
KEYS=['log_amount','hour_sin','hour_cos']+['type_'+x for x in TYPES]
def features(d):
 h=d.step.to_numpy()%24
 return np.column_stack([np.log1p(d.amount),np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)]+[(d.type==t).astype(float) for t in TYPES])
def metrics(y,p,threshold=None):
 fpr,tpr,t=roc_curve(y,p); out={'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'recall_at_1pct_fpr':float(max(tpr[fpr<=.01])),'rows':len(y),'fraud_count':int(sum(y)),'fraud_rate':float(np.mean(y))}
 if threshold is not None:
  q=p>=threshold;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
  out.update(precision=float(precision_score(y,q,zero_division=0)),recall=float(recall_score(y,q,zero_division=0)),false_positive_rate=float(fp/(fp+tn)),confusion_matrix={'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
 return out
def export(m):
 if hasattr(m,'_predictors'):
  trees=[]
  for stage in m._predictors:
   n=stage[0].nodes
   trees.append([[int(x['feature_idx']),float(x['num_threshold']),int(x['left']),int(x['right']),float(x['value']),int(x['is_leaf'])] for x in n])
  return {'kind':'hist_gradient_boosting','base':float(m._baseline_prediction[0,0]),'trees':trees}
 s=m[0];l=m[1]
 return {'kind':'logistic','mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'weights':l.coef_[0].tolist(),'intercept':float(l.intercept_[0])}
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--sample-rate',type=float,default=.15);args=a.parse_args()
 if not 0<args.sample_rate<=1:raise ValueError('sample-rate must be in (0,1]')
 digest=hashlib.sha256(Path(args.data).read_bytes()).hexdigest();rng=np.random.default_rng(42);chunks=[];total=0;frauds=0
 with zipfile.ZipFile(args.data) as z:
  member=next(n for n in z.namelist() if n.endswith('.csv'))
  with z.open(member) as f:
   for d in pd.read_csv(f,usecols=['step','type','amount','isFraud'],chunksize=250000):
    total+=len(d);frauds+=int(d.isFraud.sum());chunks.append(d[rng.random(len(d))<args.sample_rate])
 d=pd.concat(chunks,ignore_index=True).sort_values('step',kind='stable');del chunks
 bounds=[int(d.step.quantile(q)) for q in [.60,.75,.85]]
 masks=[d.step<=bounds[0],(d.step>bounds[0])&(d.step<=bounds[1]),(d.step>bounds[1])&(d.step<=bounds[2]),d.step>bounds[2]]
 split=[d[m] for m in masks];arr=[(features(s),s.isFraud.to_numpy()) for s in split]
 print(json.dumps({'source_rows':total,'sample_rows':len(d),'bounds':bounds,'splits':[{'rows':len(s),'fraud':int(s.isFraud.sum())} for s in split]}),flush=True)
 for _,y in arr:
  if min(np.bincount(y,minlength=2))<20:raise ValueError('Too few examples of either class in a split')
 specs=[('Logistic regression',lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=250,C=1))),('Gradient boosting · 15 leaves',lambda:HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.07,l2_regularization=5,min_samples_leaf=60,early_stopping=False,random_state=42)),('Gradient boosting · 31 leaves',lambda:HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,learning_rate=.06,l2_regularization=10,min_samples_leaf=80,early_stopping=False,random_state=42)),('Weighted gradient boosting',lambda:HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.06,l2_regularization=10,min_samples_leaf=80,class_weight={0:1,1:10},early_stopping=False,random_state=42))]
 results=[];best=None;bestap=-1
 for name,factory in specs:
  start=time.time();m=factory().fit(*arr[0]);p=m.predict_proba(arr[1][0])[:,1];met=metrics(arr[1][1],p);results.append({'name':name,'validation':met,'training_seconds':round(time.time()-start,2)})
  print(json.dumps(results[-1]),flush=True)
  if met['average_precision']>bestap:bestap=met['average_precision'];best=(name,factory)
 name,factory=best;training=pd.concat(split[:2]);m=factory().fit(features(training),training.isFraud)
 calp=m.predict_proba(arr[2][0])[:,1];neg=calp[arr[2][1]==0]
 def threshold(fpr):return float(np.nextafter(np.quantile(neg,1-fpr,method='higher'),np.inf))
 review=threshold(.01);block=threshold(.001)
 testp=m.predict_proba(arr[3][0])[:,1]
 meta={'model_id':'paysim-v1-'+digest[:10],'selected_model':name,'trained_at':pd.Timestamp.now(tz='UTC').isoformat(),'dataset':{'title':'PaySim','url':'https://www.kaggle.com/datasets/ealaxi/paysim1','license':'CC BY-SA 4.0','synthetic':True,'upi_data':False,'sha256':digest,'source_rows':total,'source_fraud':frauds,'sample_rate':args.sample_rate,'sample_rows':len(d),'seed':42},'features':KEYS,'excluded':['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud','nameOrig','nameDest','absolute step'],'protocol':'Chronological step-separated 60/15/10/15 split; candidate selection on validation AP; selected candidate refit on train+validation; thresholds on calibration negatives; final test untouched. No SMOTE. Random row sampling independent of labels.','boundaries':bounds,'splits':[{'name':n,'rows':len(s),'fraud':int(s.isFraud.sum()),'step_min':int(s.step.min()),'step_max':int(s.step.max())} for n,s in zip(['train','validation','calibration','test'],split)],'candidates':results,'thresholds':{'review':review,'block':block,'calibration_review_fpr':float(np.mean(neg>=review)),'calibration_block_fpr':float(np.mean(neg>=block))},'test':metrics(arr[3][1],testp,review),'test_block':metrics(arr[3][1],testp,block),'sklearn_version':sklearn.__version__,'limitations':['Synthetic African mobile-money benchmark, not UPI or real bank data.','Fraud labels represent simulated account takeover, not coerced payments.','Only amount, transaction type and cyclic hour are used; weak signal and simulator artifacts limit generalization.','Threshold false-positive targets apply to calibration data; later data may shift.','No raw balances, identities, post-transaction fields or fraud flags are prediction inputs.']}
 artifact={'metadata':meta,'types':TYPES,'model':export(m)}
 (ROOT/'models').mkdir(exist_ok=True)
 (ROOT/'models/paysim.json').write_text(json.dumps(artifact,separators=(',',':')))
 (ROOT/'models/report.json').write_text(json.dumps(meta,indent=2))
 preview=split[3].sample(n=min(300,len(split[3])),random_state=10).sort_values('step'); mixed=pd.concat([split[3][split[3].isFraud==1].sample(n=min(20,int(split[3].isFraud.sum())),random_state=7),split[3][split[3].isFraud==0].sample(n=40,random_state=7)]).sample(frac=1,random_state=7)
 def records(frame):return [{'transaction_id':'PS-'+str(i),'amount':float(r.amount),'type':r.type,'hour':int(r.step%24),'label':int(r.isFraud)} for i,r in frame.iterrows()]
 (ROOT/'models/examples.json').write_text(json.dumps({'sample':records(preview),'replay':records(mixed),'replay_note':'Curated held-out replay: 20 fraud + 40 legitimate records, not live bank traffic or representative prevalence.'}))
 parity=preview.head(30)
 (ROOT/'models/parity.json').write_text(json.dumps({'records':records(parity),'probabilities':m.predict_proba(features(parity))[:,1].tolist()}))
 print(json.dumps({'selected':name,'test':meta['test'],'thresholds':meta['thresholds']}),flush=True)
if __name__=='__main__':main()
