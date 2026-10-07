export function validateTransaction(row){
 if(!row||typeof row!=='object'||Array.isArray(row))throw Error('A transaction object is required.');
 const allowed=['transaction_id','amount','type','hour'];for(const k of Object.keys(row))if(!allowed.includes(k))throw Error('Unsupported field: '+k+'. Only transaction_id, amount, type and hour are accepted.');
 if(typeof row.amount!=='number'||!Number.isFinite(row.amount)||row.amount<=0||row.amount>1e9)throw Error('amount must be a number greater than 0 and at most 1,000,000,000.');
 if(!['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER'].includes(row.type))throw Error('type must be CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER.');
 if(!Number.isInteger(row.hour)||row.hour<0||row.hour>23)throw Error('hour must be an integer from 0 to 23.');
 if(row.transaction_id!==undefined&&(typeof row.transaction_id!=='string'||!/^[A-Za-z0-9_-]{1,80}$/.test(row.transaction_id)))throw Error('transaction_id must contain 1–80 letters, numbers, underscores or hyphens.');
 return row;
}
export function predict(artifact,row){
 validateTransaction(row);const x=[Math.log1p(row.amount),Math.sin(2*Math.PI*row.hour/24),Math.cos(2*Math.PI*row.hour/24),...artifact.types.map(t=>+(t===row.type))],m=artifact.model;let raw;
 if(m.kind==='logistic')raw=m.intercept+x.reduce((s,v,i)=>s+(v-m.mean[i])/m.scale[i]*m.weights[i],0);
 else {raw=m.base;for(const tree of m.trees){let n=tree[0];while(!n[5])n=tree[x[n[0]]<=n[1]?n[2]:n[3]];raw+=n[4]}}
 const score=1/(1+Math.exp(-raw)),{review,block}=artifact.metadata.thresholds;
 return {transaction_id:row.transaction_id||crypto.randomUUID(),fraud_score:score,action:score>=block?'Review urgently':score>=review?'Review':'Low risk',model_id:artifact.metadata.model_id,model_name:artifact.metadata.selected_model,score_scope:'PaySim transaction fraud; not a coercion probability',thresholds:{review,block},payment_action_executed:false,signals:[{name:'Transaction type',value:row.type},{name:'Amount (dataset currency units)',value:row.amount},{name:'Hour',value:row.hour}],warnings:['Research decision support only. No payment has been executed or blocked.','Model trained on synthetic mobile-money data, not validated for real UPI.']};
}
