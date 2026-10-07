importScripts('engine.js');onmessage=e=>{try{postMessage({result:TrustEngine.train(e.data)})}catch(error){postMessage({error:error.message})}};
