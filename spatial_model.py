"""Static 4m² feature partitioning with sparse binary mixed-integer optimization.

All demo observations are synthetic. Continuous internal geometry is raster sampled.
No temperature/airflow physics is solved here. See MODEL.md for definitions.
"""
from dataclasses import dataclass, asdict
import math
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import milp, Bounds, LinearConstraint
import scenario_generator as demo

PRESETS=demo.PRESETS
DENSITY_MODES=demo.DENSITY_MODES
WEIGHTS=np.array([.6,.1,.1,.2])
FEATURES=('밀도','평활 유입','평활 유출','환경 지표')

@dataclass
class Settings:
    preset:str='random'
    shape_seed:str='20261107'
    people_seed:str='13579'
    flow_seed:str='27182'
    environment_seed:str='31415'
    distribution:str='smooth'
    density_level:float=.6
    width_m:int=48
    height_m:int=32
    minutes:float=10.
    smoothing:float=1.
    density_weight:float=.6
    tolerance:float=1.
    rectangle_penalty:float=.08
    outside_penalty:float=.5
    time_limit:float=20.

def validate(s):
    if s.preset not in PRESETS or s.distribution not in DENSITY_MODES: raise ValueError('알 수 없는 시나리오 유형입니다.')
    if (s.width_m,s.height_m) not in [(32,20),(48,32),(64,40)]: raise ValueError('지원하지 않는 공간 크기입니다.')
    vals=[s.density_level,s.minutes,s.smoothing,s.density_weight,s.tolerance,s.rectangle_penalty,s.outside_penalty,s.time_limit]
    if not all(math.isfinite(v) for v in vals): raise ValueError('설정에 유한한 숫자를 입력하세요.')
    if not 0<=s.density_level<=5 or not s.minutes>0 or not 0<=s.smoothing<=20 or not 0<=s.density_weight<=1 or not 0<s.tolerance<=5 or not 0<s.rectangle_penalty<=1 or not 0<=s.outside_penalty<=30 or not 1<=s.time_limit<=300: raise ValueError('설정값이 허용 범위를 벗어났습니다.')

def rng(seed):
    import hashlib
    return np.random.default_rng(int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8],'little'))

def edges_for(mask,w,h):
    """4-neighbor connections across actual touching raster pixels, no corner links."""
    edges=[]
    for y in range(h):
        for x in range(w):
            i=y*w+x
            if x+1<w and np.any(mask[y*6:(y+1)*6,(x+1)*6-1]&mask[y*6:(y+1)*6,(x+1)*6]): edges.append((i,i+1))
            if y+1<h and np.any(mask[(y+1)*6-1,x*6:(x+1)*6]&mask[(y+1)*6,x*6:(x+1)*6]): edges.append((i,i+w))
    return edges

def smooth(values,edges,strength):
    n=len(values); row=[];col=[];data=[];degree=np.zeros(n)
    for a,b in edges:
        degree[a]+=1;degree[b]+=1;row.extend([a,b]);col.extend([b,a]);data.extend([-strength,-strength])
    matrix=coo_matrix((data,(row,col)),shape=(n,n)).tocsr()+diags(1+strength*degree)
    result=spsolve(matrix,values)
    return np.maximum(result,0)

def standardize(values,area):
    total=area.sum();mean=(values*area[:,None]).sum(axis=0)/total
    sd=np.sqrt(((values-mean)**2*area[:,None]).sum(axis=0)/total)
    scale=np.where(sd>1e-10,sd,1.)
    out=(values-mean)/scale;out[:,sd<=1e-10]=0
    return out,mean,sd

def generate(s):
    validate(s);w=s.width_m//2;h=s.height_m//2
    original=np.array(demo.generate_shape(s.preset,s.shape_seed)).reshape(demo.HEIGHT,demo.WIDTH)
    yy=np.minimum((np.arange(h*6)+.5)*demo.HEIGHT/(h*6),demo.HEIGHT-1).astype(int)
    xx=np.minimum((np.arange(w*6)+.5)*demo.WIDTH/(w*6),demo.WIDTH-1).astype(int)
    mask=original[np.ix_(yy,xx)]
    area=mask.reshape(h,6,w,6).sum(axis=(1,3)).ravel()/9.
    active=area>0;total=area.sum()
    if not total: raise ValueError('실제 공간이 비어 있습니다.')
    expected=np.array(demo.generate_density(original.ravel().tolist(),s.people_seed,s.distribution,s.density_level)).reshape(demo.HEIGHT,demo.WIDTH)
    fine_density=expected[np.ix_(yy,xx)]*mask
    expected_count=fine_density.reshape(h,6,w,6).sum(axis=(1,3)).ravel()/9.
    people_rng=rng(s.people_seed);counts=people_rng.poisson(expected_count);counts[~active]=0
    density=np.divide(counts,area,out=np.zeros_like(area),where=active)
    edges=edges_for(mask,w,h);wflow=rng(s.flow_seed)
    # Synthetic crossing observations share the same edge count on both cells.
    attraction=smooth(wflow.normal(size=w*h),edges,2.)
    attraction=np.exp(.6*attraction)
    cin=np.zeros(w*h,dtype=int);cout=cin.copy()
    for a,b in edges:
        occupancy=(counts[a]+counts[b])/2
        rate=.22*occupancy
        ab=wflow.poisson(s.minutes*rate*attraction[b]);ba=wflow.poisson(s.minutes*rate*attraction[a])
        cout[a]+=ab;cin[b]+=ab;cout[b]+=ba;cin[a]+=ba
    fin0=np.divide(cin,area*s.minutes,out=np.zeros_like(area),where=active)
    fout0=np.divide(cout,area*s.minutes,out=np.zeros_like(area),where=active)
    fin=smooth(fin0,edges,s.smoothing);fout=smooth(fout0,edges,s.smoothing)
    envrng=rng(s.environment_seed);X,Y=np.meshgrid((np.arange(w)+.5)/w,(np.arange(h)+.5)/h)
    phase=envrng.uniform(0,2*np.pi,4)
    shade=(np.sin(X*5+phase[0])+np.cos(Y*4+phase[1])>0)
    sun=np.where(shade,70.,750.)
    temperature=31+2*X+1.5*Y+.8*np.sin(X*4+phase[2])
    ux=1.1*np.cos(phase[3])*(.4+X)+.35*np.sin(Y*5)
    uy=1.1*np.sin(phase[3])*(.4+Y)+.35*np.cos(X*4)
    humidity=58+9*Y+4*np.cos(X*3+phase[0])
    env=np.column_stack([v.ravel() for v in (temperature,ux,uy,sun,humidity)])
    envz,envmean,envsd=standardize(env,area)
    # One common speed scale for x and y preserves rotation of the distance metric.
    windscale=np.sqrt((envsd[1]**2+envsd[2]**2)/2)
    if windscale>1e-10:envz[:,1:3]=(env[:,1:3]-envmean[1:3])/windscale
    ed=np.sqrt(np.sum(envz**2,axis=1)/4.)
    z=np.column_stack((density,fin,fout,ed));y,means,scales=standardize(z,area);y[~active]=0
    person_positions=[]
    for i,n in enumerate(counts):
        if not n: continue
        cy,cx=divmod(i,w);valid=np.argwhere(mask[cy*6:(cy+1)*6,cx*6:(cx+1)*6])
        for _ in range(n):
            py,px=valid[people_rng.integers(len(valid))]
            person_positions.append(((cx*6+px+.5)/(w*6),(cy*6+py+.5)/(h*6)))
    return dict(w=w,h=h,mask=mask,area=area,active=active,counts=counts,density=density,cin=cin,cout=cout,fin0=fin0,fout0=fout0,fin=fin,fout=fout,env=env,env_reference=envmean,env_sd=envsd,environment=ed,z=z,y=y,means=means,scales=scales,edges=edges,people=person_positions)

def candidate_rectangles(scene,s,progress=lambda _:None):
    w,h=scene['w'],scene['h'];area=scene['area'];active=scene['active'];y=scene['y'];z=scene['z']
    wd=s.density_weight;weights=np.array([wd,(1-wd)/4,(1-wd)/4,(1-wd)/2])
    # wd=.6 gives .6/.1/.1/.2. Feature ranges are diagnostic, never hard gates.
    baseline=float(np.sum(area[:,None]*weights*y*y));total=area.sum()
    limits=np.array([1.,2.,2.,2.])*s.tolerance
    grid_y=y.reshape(h,w,4);grid_a=area.reshape(h,w);grid_z=z.reshape(h,w,4)
    result=[];attempted=0;rejected=0
    for top in range(h):
        ca=np.zeros(w);su=np.zeros((w,4));sq=su.copy();original=su.copy();lo=np.full((w,4),np.inf);hi=np.full((w,4),-np.inf)
        for bottom in range(top,h):
            a=grid_a[bottom];v=grid_y[bottom];present=a>0
            ca+=a;su+=a[:,None]*v;sq+=a[:,None]*v*v;original+=a[:,None]*grid_z[bottom]
            lo=np.minimum(lo,np.where(present[:,None],v,np.inf));hi=np.maximum(hi,np.where(present[:,None],v,-np.inf))
            for left in range(w):
                ar=0.;ss=np.zeros(4);qq=ss.copy();oz=ss.copy();mn=np.full(4,np.inf);mx=np.full(4,-np.inf)
                for right in range(left,w):
                    ar+=ca[right];ss+=su[right];qq+=sq[right];oz+=original[right];mn=np.minimum(mn,lo[right]);mx=np.maximum(mx,hi[right])
                    if ar<=0:continue
                    attempted+=1
                    rw=right-left+1;rh=bottom-top+1
                    err=float(np.dot(weights,np.maximum(0,qq-ss*ss/ar)))
                    outside=max(0.,rw*rh*4-ar)
                    cost=(err/baseline if baseline>1e-12 else 0)+s.rectangle_penalty+s.outside_penalty*outside/total
                    # Spatially tighten: never keep an entirely empty outer row/column.
                    part=grid_a[top:bottom+1,left:right+1]
                    if not (part[0].any() and part[-1].any() and part[:,0].any() and part[:,-1].any()):continue
                    result.append(dict(x=left,y=top,w=rw,h=rh,inside=ar,outside=outside,error=err,cost=cost,mean=(oz/ar).tolist(),ranges=(mx-mn).tolist()))
        progress(f'직사각형 후보 생성 {top+1}/{h}행 · 유효 후보 {len(result):,}개')
    return result,baseline,weights,dict(attempted=attempted,rejected=rejected)

def optimize(scene,s,progress=lambda _:None):
    validate(s);start=time.perf_counter();candidates,baseline,weights,stats=candidate_rectangles(scene,s,progress)
    w,h=scene['w'],scene['h'];n=w*h;rows=[];cols=[]
    for j,r in enumerate(candidates):
        for yy in range(r['y'],r['y']+r['h']):
            rows.extend(range(yy*w+r['x'],yy*w+r['x']+r['w']));cols.extend([j]*r['w'])
    A=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,len(candidates))).tocsc()
    cost=np.array([r['cost'] for r in candidates]);lower=scene['active'].astype(float)
    progress(f'0–1 정수최적화 · 후보 {len(candidates):,}개 / 최대 {s.time_limit:g}초')
    # Valid single-cell cover supplies an incumbent/cutoff even if the solver times out.
    fallback=[r for r in candidates if r['w']==1 and r['h']==1]
    full=[r for r in candidates if np.isclose(r['inside'],scene['area'].sum())]
    if full and min(full,key=lambda r:r['cost'])['cost']<sum(r['cost'] for r in fallback):fallback=[min(full,key=lambda r:r['cost'])]
    fallbackcost=sum(r['cost'] for r in fallback)
    result=milp(cost,integrality=np.ones(len(cost)),bounds=Bounds(0,1),constraints=LinearConstraint(A,lower,np.ones(n)),options={'time_limit':s.time_limit,'mip_rel_gap':.001,'presolve':False})
    if result.x is None:
        chosen=fallback;status='정수해 미확보: 유효한 대체 분할 사용';gap=None;optimal=False
    else:
        chosen=[r for r,v in zip(candidates,result.x) if v>.5]
        verify(scene,chosen)
        chosen_cost=sum(r['cost'] for r in chosen)
        if chosen_cost>fallbackcost+1e-8:chosen=fallback;status='유효 대체 분할이 더 저렴하여 사용';gap=None;optimal=False
        else:status='정수최적화 완료' if result.status==0 else '시간 제한: 현재까지 찾은 정수해';gap=getattr(result,'mip_gap',None);optimal=result.status==0
    verify(scene,chosen)
    return dict(rectangles=chosen,candidate_count=len(candidates),baseline=baseline,weights=weights.tolist(),status=status,gap=gap,optimal=optimal,seconds=time.perf_counter()-start,objective=sum(r['cost'] for r in chosen),error_ratio=sum(r['error'] for r in chosen)/baseline if baseline>1e-12 else 0,outside=sum(r['outside'] for r in chosen),settings=asdict(s),unrestricted=True,heterogeneity_warnings=int(sum(np.any(np.array(r['ranges'])>np.array([1,2,2,2])*s.tolerance+1e-9) for r in chosen)),**stats)

def verify(scene,rects,limits=None):
    cover=np.zeros((scene['h'],scene['w']),dtype=int)
    for r in rects:
        cover[r['y']:r['y']+r['h'],r['x']:r['x']+r['w']]+=1
        if limits is not None and np.any(np.array(r['ranges'])>limits+1e-7):raise AssertionError('특성 허용 범위 위반')
    cover=cover.ravel()
    if not np.all(cover[scene['active']]==1) or np.any(cover>1):raise AssertionError('완전 커버/겹침 제약 위반')
    return True
