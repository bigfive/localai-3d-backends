"""Exact query-chunking for noncausal inference attention on MPS."""
import torch
import torch.nn.functional as F
ORIGINAL=F.scaled_dot_product_attention
SEEN=set()
def bounded_attention(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
 if query.device.type!='mps' or query.shape[-2]<=256:
  return ORIGINAL(query,key,value,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
 if attn_mask is not None or dropout_p!=0 or is_causal:
  raise ValueError('Bounded SF3D attention requires mask-free noncausal inference with dropout=0')
 shape=(tuple(query.shape),tuple(key.shape))
 if shape not in SEEN:
  print('Query-chunked MPS attention:',shape,flush=True);SEEN.add(shape)
 outputs=[]
 for start in range(0,query.shape[-2],256):
  outputs.append(ORIGINAL(query[...,start:start+256,:],key,value,dropout_p=0,is_causal=False,scale=scale,enable_gqa=enable_gqa))
  torch.mps.synchronize()
 return torch.cat(outputs,dim=-2)
def install():F.scaled_dot_product_attention=bounded_attention
if __name__=='__main__':
 torch.manual_seed(42)
 q=torch.randn(1,4,768,64,device='mps');k=torch.randn(1,4,512,64,device='mps');v=torch.randn_like(k)
 with torch.no_grad():
  expected=ORIGINAL(q,k,v);actual=bounded_attention(q,k,v)
 error=(expected-actual).abs().max().item();print('Maximum attention difference:',error,flush=True)
 torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
 print('PASS: query chunking preserves attention output',flush=True)
