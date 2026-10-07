const secureHeaders={'X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin','Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self' https://chatgpt.com"};
function json(data,status=200){return new Response(JSON.stringify(data),{status,headers:{...secureHeaders,'Content-Type':'application/json'}})}
async function body(request){if(!request.headers.get('content-type')?.includes('application/json'))throw Error('Content-Type must be application/json.');const reader=request.body?.getReader();if(!reader)throw Error('JSON body required.');let length=0,parts=[];while(true){const{value,done}=await reader.read();if(done)break;length+=value.byteLength;if(length>150000){await reader.cancel();throw Error('Request exceeds 150 KB.')}parts.push(value)}let all=new Uint8Array(length),o=0;for(const p of parts){all.set(p,o);o+=p.length}try{return JSON.parse(new TextDecoder().decode(all))}catch{throw Error('Invalid JSON body.')}}
export default {async fetch(request){const url=new URL(request.url),p=url.pathname;try{
 if(p.startsWith('/api/')){
  const origin=request.headers.get('origin');if(origin&&origin!==url.origin)return json({error:'Cross-origin requests are not allowed.'},403);
  if(request.method==='GET'&&p==='/api/health')return json({status:'ok',model_loaded:true,model_id:artifact.metadata.model_id,live_bank_connected:false});
  if(request.method==='GET'&&p==='/api/model')return json(artifact.metadata);
  if(request.method==='GET'&&p==='/api/demo')return json(examples);
  if(request.method==='GET'&&p==='/api/openapi.json')return new Response(assets['/openapi.json'].body,{headers:{...secureHeaders,'Content-Type':'application/json'}});
  if(request.method==='POST'&&(p==='/api/predict'||p==='/api/predict/batch')){const start=performance.now(),data=await body(request);if(p==='/api/predict'){const result=predict(artifact,data);return json({...result,inference_ms:Math.round((performance.now()-start)*100)/100,scored_at:new Date().toISOString()})}
   if(!data||Object.keys(data).length!==1||!Array.isArray(data.transactions)||data.transactions.length<1||data.transactions.length>200)throw Error('Provide {transactions: [...]} with 1–200 transactions.');data.transactions.forEach(validateTransaction);return json({predictions:data.transactions.map(r=>predict(artifact,r)),count:data.transactions.length,inference_ms:Math.round((performance.now()-start)*100)/100,model_id:artifact.metadata.model_id});}
  const known=['/api/health','/api/model','/api/demo','/api/openapi.json','/api/predict','/api/predict/batch'];return json({error:known.includes(p)?'Method not allowed.':'Endpoint not found.'},known.includes(p)?405:404);
 }
 if(request.method!=='GET'&&request.method!=='HEAD')return json({error:'Method not allowed.'},405);
 const asset=assets[p==='/'?'/index.html':p];if(!asset)return json({error:'Not found'},404);return new Response(request.method==='HEAD'?null:asset.binary?Uint8Array.from(atob(asset.body),c=>c.charCodeAt(0)):asset.body,{headers:{...secureHeaders,'Content-Type':asset.type}});
 }catch(e){return json({error:e.message||'Unable to score request.'},400)}}};
