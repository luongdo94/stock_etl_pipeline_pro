// Audit-only numerical checks. This does NOT execute the repository's Python implementation.
// Inputs are fictional EUR units; no company financials or market data are asserted.
import assert from 'node:assert/strict';
export function transcription(f,g,r,tg=.02,n=10){
 if(r<=tg) throw Error('r must exceed terminal growth');
 let cf=f,pv=0;for(let t=1;t<=n;t++){cf*=1+g+(tg-g)*(t-1)/Math.max(n-1,1);pv+=cf/(1+r)**t;}
 return pv+cf*(1+tg)/(r-tg)/(1+r)**n;
}
export function reference(f,g,r,tg=.02,n=10){
 const growths=Array.from({length:n},(_,i)=>g+(tg-g)*i/Math.max(n-1,1));
 const cashflows=growths.map((_,i)=>f*growths.slice(0,i+1).reduce((product,x)=>product*(1+x),1));
 const explicit=cashflows.reduce((s,x,i)=>s+x/(1+r)**(i+1),0);
 const terminal=cashflows[n-1]*(1+tg)/(r-tg)/(1+r)**n;
 return {cashflows,explicit,terminal,equity:explicit+terminal,terminalShare:terminal/(explicit+terminal)};
}
export const cases=[
 {id:'mature',archetype:'Mature profitable compounder',cfo:180,capex:60,sbc:10,interest:5,lease:5,shares:10,g:.05,r:.09},
 {id:'growth',archetype:'High growth with reinvestment and SBC',cfo:180,capex:100,sbc:30,interest:0,lease:0,shares:10,g:.20,r:.11},
 {id:'cyclical',archetype:'Cyclical industrial; median of adjusted FCF [80,110,200]',fcfe:110,shares:10,g:.03,r:.10},
 {id:'negative',archetype:'Negative near-term FCFE',cfo:50,capex:80,sbc:10,interest:0,lease:0,shares:10,g:.15,r:.12},
 {id:'sbc',archetype:'Substantial SBC',cfo:180,capex:60,sbc:60,interest:0,lease:0,shares:10,g:.10,r:.11},
 {id:'excess_cash',archetype:'Same operating cash flow; excess cash 500',fcfe:100,shares:10,g:.05,r:.09,excessCash:500},
 {id:'net_debt',archetype:'Same operating cash flow; net debt 1000, assumed rolled over',fcfe:100,shares:10,g:.05,r:.09,netDebt:1000}
];
export const results=cases.map(c=>{
 const f=c.fcfe??(c.cfo-c.capex-c.sbc-c.interest-c.lease);
 if(f<=0)return {...c,fcfe:f,expectedHelperValue:null,sourceEquivalentValue:null,note:'dcf_per_share rejects nonpositive initial FCFE; no turnaround forecast implemented'};
 const ref=reference(f,c.g,c.r),actual=transcription(f,c.g,c.r)/c.shares;
 assert.ok(Math.abs(actual-ref.equity/c.shares)<1e-10);
 return {...c,fcfe:f,expectedHelperValue:ref.equity/c.shares,sourceEquivalentValue:actual,terminalShare:ref.terminalShare,explicitPV:ref.explicit,terminalPV:ref.terminal};
});
assert.ok(Math.abs(transcription(100,.02,.09)-100*1.02/(.09-.02))<1e-9);
export const sensitivity=[.08,.09,.10].flatMap(r=>[.01,.02,.03].map(tg=>({r,tg,perShare:transcription(100,.05,r,tg)/10})));
export function residualReference(b,roe,r,payout=.5,n=10){
 const rates=Array.from({length:n},(_,i)=>roe+(r-roe)*(i+1)/n);
 return b+rates.reduce((s,x,i)=>s+(x-r)*b*rates.slice(0,i).reduce((v,y)=>v*(1+Math.max(y,0)*(1-payout)),1)/(1+r)**(i+1),0);
}
export const controls={
 constantGrowthValue:transcription(100,.02,.09),constantGrowthIndependent:100*1.02/(.09-.02),
 riAtCostOfEquity:residualReference(100,.10,.10),riAboveCost:residualReference(100,.15,.10),riBelowCost:residualReference(100,.05,.10),
 payoutNativeEpsBug:.05*90/10,payoutSameCurrencyExpected:.05*90/(10*.9),
 roicGapFallback:(20*.8+40*.8)/2/140,
 missingScoreAt60pctCoverage:50+(100-50)*(.6/.8),
 portfolioIdenticalReturns:(1.1*1.05-1)*100,benchmarkDroppedFirst:(1.05-1)*100,
 rotationExpectedCostBp:40+10,rotationSourceEquivalentCostBp:2*10,
 retainedNamesWeightTurnover:Math.abs(.8-.5)+Math.abs(.2-.5)
};
assert.equal(controls.riAtCostOfEquity,100);assert.ok(controls.riAboveCost>100&&controls.riBelowCost<100);
export const payload={method:'Independently expanded cash-flow reference compared with a manual JavaScript transcription of core/valuation.py:365-384. No repository Python execution. All inputs fictional.',results,sensitivity,controls};
console.log(JSON.stringify(payload,null,2));
