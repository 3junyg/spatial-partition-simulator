"""Static 1m² feature partitioning with sparse binary mixed-integer optimization.

All demo observations are synthetic. Continuous internal geometry is raster sampled.
No temperature/airflow physics is solved here. See MODEL.md for definitions.
"""
from dataclasses import dataclass, asdict
import math
import time
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import milp, Bounds, LinearConstraint
import scenario_generator as demo

PRESETS=demo.PRESETS
DENSITY_MODES=demo.DENSITY_MODES
WEIGHTS=np.array([.6,.1,.1,.2])
FEATURES=('밀도','평활 유입','평활 유출','환경 지표')
CELL_M = 1

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
    if not 0<=s.density_level<=5 or not s.minutes>0 or not 0<=s.smoothing<=20 or not 0<=s.density_weight<=1 or not 0<s.tolerance<=5 or not 0<s.rectangle_penalty<=1 or not 0<=s.outside_penalty<=10 or not 1<=s.time_limit<=300: raise ValueError('설정값이 허용 범위를 벗어났습니다.')

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

def generate(s, cell_m=CELL_M):
    validate(s)
    if cell_m not in (1, 2):
        raise ValueError('격자 한 변은 1m 또는 2m여야 합니다.')
    w, h = s.width_m // cell_m, s.height_m // cell_m
    pixels_per_m2 = 36 / cell_m ** 2
    original=np.array(demo.generate_shape(s.preset,s.shape_seed)).reshape(demo.HEIGHT,demo.WIDTH)
    yy=np.minimum((np.arange(h*6)+.5)*demo.HEIGHT/(h*6),demo.HEIGHT-1).astype(int)
    xx=np.minimum((np.arange(w*6)+.5)*demo.WIDTH/(w*6),demo.WIDTH-1).astype(int)
    mask=original[np.ix_(yy,xx)]
    area=mask.reshape(h,6,w,6).sum(axis=(1,3)).ravel()/pixels_per_m2
    active=area>0;total=area.sum()
    if not total: raise ValueError('실제 공간이 비어 있습니다.')
    expected=np.array(demo.generate_density(original.ravel().tolist(),s.people_seed,s.distribution,s.density_level)).reshape(demo.HEIGHT,demo.WIDTH)
    fine_density=expected[np.ix_(yy,xx)]*mask
    expected_count=fine_density.reshape(h,6,w,6).sum(axis=(1,3)).ravel()/pixels_per_m2
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
    return dict(w=w,h=h,cell_m=cell_m,mask=mask,area=area,active=active,counts=counts,density=density,cin=cin,cout=cout,fin0=fin0,fout0=fout0,fin=fin,fout=fout,env=env,env_reference=envmean,env_sd=envsd,environment=ed,z=z,y=y,means=means,scales=scales,edges=edges,people=person_positions)

def candidate_rectangles(scene,s,progress=lambda _:None):
    w,h=scene['w'],scene['h'];area=scene['area'];active=scene['active'];y=scene['y'];z=scene['z']
    cell_m=scene.get('cell_m', CELL_M)
    wd=s.density_weight;weights=np.array([wd,(1-wd)/4,(1-wd)/4,(1-wd)/2])
    # wd=.6 gives .6/.1/.1/.2. Feature ranges are diagnostic, never hard gates.
    baseline=float(np.sum(area[:,None]*weights*y*y));total=area.sum()
    def integral(values):
        grid = values.reshape(h,w,-1)
        return np.pad(grid.cumsum(axis=0).cumsum(axis=1), ((1,0),(1,0),(0,0)))

    occupancy = integral(active.astype(int))[:,:,0]
    grid_area = integral(area)[:,:,0]
    first = integral(area[:,None]*y)
    second = integral(area[:,None]*y*y)
    original = integral(area[:,None]*z)
    active_y,active_x=np.nonzero(active.reshape(h,w))
    ymin,ymax=int(active_y.min()),int(active_y.max())+1
    xmin,xmax=int(active_x.min()),int(active_x.max())+1
    boxes=[]
    attempted=0
    for top in range(ymin,ymax):
        for bottom in range(top+1,ymax+1):
            height=bottom-top
            band=occupancy[bottom]-occupancy[top]
            above=occupancy[top]-occupancy[max(0,top-1)]
            below=occupancy[min(h,bottom+1)]-occupancy[bottom]
            for left in range(xmin,xmax):
                if band[left+1]==band[left]:
                    continue
                right=np.arange(left+1,xmax+1)
                width=right-left
                count=band[right]-band[left]
                tight=(band[right]>band[right-1]) & (occupancy[top+1,right]-occupancy[top,right]>occupancy[top+1,left]-occupancy[top,left]) & (occupancy[bottom,right]-occupancy[bottom-1,right]>occupancy[bottom,left]-occupancy[bottom-1,left])
                attempted+=int(tight.sum())
                if cell_m==1:
                    small=(height<=4)&(width<=4)
                    aligned=(top%4==0)&(bottom%4==0)&(left%4==0)&(right%4==0)
                    medium=(height<=12)&(width<=12)&(top%2==0)&(bottom%2==0)&(left%2==0)&(right%2==0)
                    maximal_x=((left==0) or band[left]-band[left-1]<height)&((right==w)|(band[np.minimum(w,right+1)]-band[right]<height))
                    maximal_y=((top==0)|(above[right]-above[left]<width))&((bottom==h)|(below[right]-below[left]<width))
                    bounds=(top==ymin)&(bottom==ymax)&(left==xmin)&(right==xmax)
                    tight &= small|aligned|medium|((count==width*height)&(maximal_x|maximal_y))|bounds
                boxes.extend((top,bottom,left,int(r)) for r in right[tight])
        progress(f'직사각형 후보 생성 {top-ymin+1}/{ymax-ymin}행 · 유효 후보 {len(boxes):,}개')

    top,bottom,left,right=np.asarray(boxes,dtype=int).T
    def rectangle_sum(prefix):
        return prefix[bottom,right]-prefix[top,right]-prefix[bottom,left]+prefix[top,left]
    inside=rectangle_sum(grid_area)
    mean=rectangle_sum(original)/inside[:,None]
    errors=np.maximum(0,rectangle_sum(second)-rectangle_sum(first)**2/inside[:,None])@weights
    widths,heights=right-left,bottom-top
    outside=np.maximum(0,widths*heights*cell_m**2-inside)
    costs=(errors/baseline if baseline>1e-12 else np.zeros_like(errors))+s.rectangle_penalty+s.outside_penalty*outside/total
    result=[dict(x=int(x),y=int(y0),w=int(dx),h=int(dy),inside=float(a),outside=float(o),error=float(e),cost=float(c),mean=m.tolist())
            for x,y0,dx,dy,a,o,e,c,m in zip(left,top,widths,heights,inside,outside,errors,costs,mean)]
    return result,baseline,weights,dict(attempted=attempted,rejected=attempted-len(result))


def run_cover(active, transpose=False):
    """Tile active cells with maximal runs, merging identical runs across rows."""
    if transpose:
        active=active.T
    pending={}
    result=[]
    for y,row in enumerate(active):
        cuts=np.flatnonzero(np.diff(np.r_[False,row,False]))
        current={}
        for x,end in zip(cuts[::2],cuts[1::2]):
            key=(int(x),int(end-x))
            current[key]=pending.pop(key,y)
        result.extend((x,start,width,y-start) for (x,width),start in pending.items())
        pending=current
    result.extend((x,start,width,len(active)-start) for (x,width),start in pending.items())
    return [(y,x,h,w) for x,y,w,h in result] if transpose else result

def merge_cover(rectangles, lookup):
    """Improve a feasible fallback by merging compatible rectangles."""
    while 1<len(rectangles)<=80:
        best=None
        saving=1e-9
        for i,a in enumerate(rectangles):
            for j in range(i+1,len(rectangles)):
                b=rectangles[j]
                x,y=min(a['x'],b['x']),min(a['y'],b['y'])
                right=max(a['x']+a['w'],b['x']+b['w'])
                bottom=max(a['y']+a['h'],b['y']+b['h'])
                merged=lookup.get((x,y,right-x,bottom-y))
                if merged is None or abs(merged['inside']-a['inside']-b['inside'])>1e-7:
                    continue
                gain=a['cost']+b['cost']-merged['cost']
                if gain<=saving:
                    continue
                if any(x<c['x']+c['w'] and c['x']<right and y<c['y']+c['h'] and c['y']<bottom
                       for k,c in enumerate(rectangles) if k!=i and k!=j):
                    continue
                best=(i,j,merged)
                saving=gain
        if best is None:
            break
        i,j,merged=best
        rectangles[i]=merged
        rectangles.pop(j)
    return rectangles


def optimize(scene,s,progress=lambda _:None):
    validate(s);start=time.perf_counter();candidates,baseline,weights,stats=candidate_rectangles(scene,s,progress)
    w,h=scene['w'],scene['h'];n=w*h
    footprints=np.fromiter((r['w']*r['h'] for r in candidates),dtype=np.int32,count=len(candidates))
    starts=np.r_[np.int32(0),np.cumsum(footprints,dtype=np.int32)]
    rows=np.empty(starts[-1],dtype=np.int32)
    x_offsets=np.arange(w,dtype=np.int32)
    y_offsets=np.arange(h,dtype=np.int32)*w
    for j,r in enumerate(candidates):
        origin=r['y']*w+r['x']
        offsets=y_offsets[:r['h'],None]+x_offsets[None,:r['w']]
        rows[starts[j]:starts[j+1]]=origin+offsets.ravel()
    A=csc_matrix((np.ones(len(rows)),rows,starts),shape=(n,len(candidates)))
    cost=np.array([r['cost'] for r in candidates]);lower=scene['active'].astype(float)
    progress(f'0–1 정수최적화 · 후보 {len(candidates):,}개 / 최대 {s.time_limit:g}초')
    lookup={(r['x'],r['y'],r['w'],r['h']):r for r in candidates}
    active_grid=scene['active'].reshape(h,w)
    covers=[[r for r in candidates if r['w']==1 and r['h']==1]]
    covers.extend(merge_cover([lookup[box] for box in run_cover(active_grid,transpose)],lookup)
                  for transpose in (False,True))
    covers.extend([r] for r in candidates if abs(r['inside']-scene['area'].sum())<1e-7)
    fallback=min(covers,key=lambda rs:sum(r['cost'] for r in rs))
    fallbackcost=sum(r['cost'] for r in fallback)
    result=milp(cost,integrality=np.ones(len(cost)),bounds=Bounds(0,1),constraints=LinearConstraint(A,lower,np.ones(n)),options={'time_limit':s.time_limit,'mip_rel_gap':.001,'presolve':False})
    if result.x is None:
        chosen=fallback;status='정수해 미확보: 유효한 대체 분할 사용';gap=None;optimal=False
    else:
        chosen=[r for r,v in zip(candidates,result.x) if v>.5]
        verify(scene,chosen)
        chosen_cost=sum(r['cost'] for r in chosen)
        if chosen_cost>fallbackcost+1e-8:chosen=fallback;status='유효 대체 분할이 더 저렴하여 사용';gap=None;optimal=False
        else:status='후보 범위 내 정수최적화 완료' if result.status==0 else '시간 제한: 현재까지 찾은 정수해';gap=getattr(result,'mip_gap',None);optimal=result.status==0
    verify(scene,chosen)
    grid_y=scene['y'].reshape(h,w,4)
    for r in chosen:
        xs=slice(r['x'],r['x']+r['w'])
        ys=slice(r['y'],r['y']+r['h'])
        r['ranges']=np.ptp(grid_y[ys,xs][active_grid[ys,xs]],axis=0).tolist()
    return dict(rectangles=chosen,candidate_count=len(candidates),baseline=baseline,weights=weights.tolist(),status=status,gap=gap,optimal=optimal,seconds=time.perf_counter()-start,objective=sum(r['cost'] for r in chosen),error_ratio=sum(r['error'] for r in chosen)/baseline if baseline>1e-12 else 0,outside=sum(r['outside'] for r in chosen),settings=asdict(s),unrestricted=True,heterogeneity_warnings=int(sum(np.any(np.array(r['ranges'])>np.array([1,2,2,2])*s.tolerance+1e-9) for r in chosen)),**stats)

def verify(scene,rects,limits=None):
    cover=np.zeros((scene['h'],scene['w']),dtype=int)
    for r in rects:
        cover[r['y']:r['y']+r['h'],r['x']:r['x']+r['w']]+=1
        if limits is not None and np.any(np.array(r['ranges'])>limits+1e-7):raise AssertionError('특성 허용 범위 위반')
    cover=cover.ravel()
    if not np.all(cover[scene['active']]==1) or np.any(cover>1):raise AssertionError('완전 커버/겹침 제약 위반')
    return True
