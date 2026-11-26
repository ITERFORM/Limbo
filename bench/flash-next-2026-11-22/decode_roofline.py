# Copyright (c) 2026 ITERFORM. All rights reserved.
GiB=1024**3; MiB=1024**2
bw_gpu=288e9*0.75; bw_pcie=6.6e9*0.9; ebytes=1.952*MiB
def win(W,U,hr,p,dense,cpu_bw,state=0.22,kv=0.10):
    look=48*U
    gpu=dense*GiB+state*GiB+kv*GiB+look*(hr+(1-hr)*p)*ebytes
    t_gpu=gpu/bw_gpu; t_pcie=look*(1-hr)*p*ebytes/bw_pcie; t_cpu=look*(1-hr)*(1-p)*ebytes/cpu_bw
    return (max(t_gpu,t_pcie,t_cpu)*1.10+1.5e-3)*1e3, t_gpu*1e3, t_pcie*1e3, t_cpu*1e3
print('hit = resident share of lookups; p = share of the misses prefetched over PCIe in time (0 = none); CPU lane takes the rest')
for dense,lab in ((3.83,'HC bf16 (dense 3.83 GiB)'),(3.23,'HC int8 (dense 3.23 GiB)')):
  print(lab)
  for name,hr,p,cpu in (('Strata-like  hit .77 p 0   cpu 28 GB/s (7 workers)',.77,0,28e9),
                        ('Limbo       hit .91 p .68 cpu  8 GB/s (2 workers)',.91,.68,8e9),
                        ('Limbo       hit .91 p 0   cpu  8 GB/s (2 workers)',.91,0,8e9),
                        ('Limbo       hit .95 p .68 cpu  8 GB/s (2 workers)',.95,.68,8e9),
                        ('Limbo       hit .99 p 0   cpu  8 GB/s',.99,0,8e9)):
    row=[]
    for W,U,A in ((1,10,1.0),(3,21.2,2.2),(4,25.6,3.0),(4,25.6,4.0)):
        t,tg,tp,tc=win(W,U,hr,p,dense,cpu)
        row.append('W=%d A=%.1f: %5.1f ms [g%.0f p%.0f c%.0f] -> %3.0f tok/s'%(W,A,t,tg,tp,tc,A/t*1e3))
    print('  %-52s'%name,' | '.join(row))
