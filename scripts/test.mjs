import assert from 'node:assert/strict';import fs from 'node:fs';import {predict} from '../worker/inference.mjs';import worker from '../dist/server/index.js';
const a=JSON.parse(fs.readFileSync('models/paysim.json')),parity=JSON.parse(fs.readFileSync('models/parity.json'));let maxError=0;
parity.records.forEach(({label,...r},i)=>{const p=predict(a,r).fraud_score;maxError=Math.max(maxError,Math.abs(p-parity.probabilities[i]));assert(Math.abs(p-parity.probabilities[i])<1e-10)});
async function req(path,body,method=body?'POST':'GET',headers={}){return worker.fetch(new Request('https://example.test'+path,{method,headers:{'Content-Type':'application/json',...headers},body:body?JSON.stringify(body):undefined}))}
const sample={transaction_id:'TEST-001',amount:2500,type:'TRANSFER',hour:14};let r=await req('/api/predict',sample);assert.equal(r.status,200);let result=await r.json();assert.equal(result.payment_action_executed,false);assert(result.fraud_score>=0&&result.fraud_score<=1);
for(const data of [{...sample,amount:-1},{...sample,hour:24},{...sample,type:'UPI'},{...sample,amount:'100'},{...sample,isFraud:1},{...sample,transaction_id:'<script>'}])assert.equal((await req('/api/predict',data)).status,400);
assert.equal((await req('/api/predict',sample,'POST',{Origin:'https://evil.example'})).status,403);
assert.equal((await req('/api/predict/batch',{transactions:[sample,{...sample,hour:99}]})).status,400);
r=await req('/api/predict/batch',{transactions:[sample,sample]});assert.equal(r.status,200);assert.equal((await r.json()).count,2);
assert.equal((await req('/api/predict/batch',{transactions:Array(201).fill(sample)})).status,400);
assert.equal((await req('/api/predict',null)).status,405);
assert.equal((await req('/api/missing')).status,404);
for(const url of ['/','/style.css','/app.js','/lab.html','/lab.js','/engine.js','/worker.js','/api/health','/api/model','/api/demo','/api/openapi.json'])assert.equal((await req(url)).status,200,url);
const schema=await(await req('/api/openapi.json')).json();assert.equal(schema.openapi,'3.1.0');
const huge=new Request('https://example.test/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...sample,pad:'x'.repeat(150001)})});assert.equal((await worker.fetch(huge)).status,400);
console.log(JSON.stringify({checks:'passed',parity_records:parity.records.length,max_python_javascript_error:maxError,api_contracts:'valid and invalid single/batch requests, origin, payload limit, static routes'}));
