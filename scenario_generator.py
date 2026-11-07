"""Seeded spatial partition model. Standard library only; no global optimality claim."""
from dataclasses import dataclass, asdict
import random
import math

WIDTH, HEIGHT = 120, 72
WORLD_WIDTH, WORLD_HEIGHT = 100.0, 60.0
PRESETS = {'random':'랜덤 연결 공간','stadium':'경기장 관람객 통로','concert':'콘서트 야외 공연장','patio':'야외 식당 테라스','terminal':'개방형 공항 터미널','mall':'야외 쇼핑몰 보행 통로','transit':'반개방형 교통 대기 공간'}

def generate_shape(kind, seed):
    rng = random.Random(str(seed))
    mask = [False] * (WIDTH * HEIGHT)
    def rect(x,y,w,h):
        for yy in range(max(0,int(y)), min(HEIGHT,int(y+h))):
            for xx in range(max(0,int(x)), min(WIDTH,int(x+w))):
                mask[yy*WIDTH+xx] = True
    # Geometry uses a common normalized 150 x 90 drawing coordinate system.
    def box(x,y,w,h): rect(x*.8,y*.8,w*.8,h*.8)
    if kind == 'random':
        x,y=65,38
        box(x,y,20,15)
        for _ in range(25):
            nx=max(8,min(125,x+rng.randint(-18,18)))
            ny=max(8,min(68,y+rng.randint(-15,15)))
            box(min(x,nx),y,abs(nx-x)+10,10)
            box(nx,min(y,ny),10,abs(ny-y)+10)
            box(nx,ny,rng.randint(8,24),rng.randint(7,21))
            x,y=nx,ny
    elif kind == 'stadium':
        for b in [(12,12,126,12),(12,66,126,12),(12,12,13,66),(125,12,13,66)]: box(*b)
        for x in range(33,120,25): box(x,21,8,rng.randint(10,19))
    elif kind == 'concert':
        for y in range(12,80):
            half=22+int((y-12)*.57)
            box(75-half,y,half*2,2)
        box(10,68,130,11)
    elif kind == 'patio':
        for b in [(15,15,105,58),(110,35,25,38),(30,8,35,10)]: box(*b)
    elif kind == 'terminal':
        box(12,34,126,23)
        for x in range(20,130,30):
            box(x,10,12,30); box(x+4,52,13,25)
        box(50,28,40,35)
    elif kind == 'mall':
        for b in [(10,36,130,14),(65,10,16,70),(40,27,63,35)]: box(*b)
        for x in range(15,135,28): box(x,22,12,20)
    elif kind == 'transit':
        for b in [(15,32,120,25),(20,18,26,20),(65,52,25,22),(110,20,20,18)]: box(*b)
    else: raise ValueError('Unknown preset')
    if kind != 'random':
        for _ in range(6):
            i=rng.choice([i for i,v in enumerate(mask) if v])
            rect(i%WIDTH-3,i//WIDTH-3,rng.randint(7,13),rng.randint(7,12))
    return mask

DENSITY_MODES = {'auto':'시드에 따라 선택', 'uniform':'균일 분포', 'smooth':'완만한 분포', 'clustered':'군집 분포', 'empty':'사람 없음'}

def generate_density(mask, seed, mode='auto', level=0.2):
    """Continuous fields with independent crowd level; never force a density peak.

    level is a nominal density scale, not a prescribed maximum or population.
    Smooth saturation limits densities below 5 without creating hard plateaus.
    """
    if mode not in DENSITY_MODES: raise ValueError('잘못된 분포 유형입니다.')
    level=float(level)
    if not math.isfinite(level) or not 0<=level<=5: raise ValueError('혼잡도는 0–5 범위여야 합니다.')
    rng=random.Random(str(seed)); points=[i for i,v in enumerate(mask) if v]
    density=[0.] * len(mask)
    if mode=='auto': mode=rng.choice(['uniform','smooth','clustered','empty'])
    if not points or mode=='empty' or level==0: return density
    phases=[rng.uniform(0,2*math.pi) for _ in range(3)]
    peaks=[]
    if mode=='clustered':
        for _ in range(rng.randint(1,6)):
            p=rng.choice(points)
            peaks.append((p%WIDTH+.5,p//WIDTH+.5,rng.uniform(6,20),
                          rng.uniform(5,16),rng.uniform(0,math.pi),rng.uniform(.4,2.5)))
    for i in points:
        x,y=i%WIDTH+.5,i//WIDTH+.5
        if mode=='uniform':
            density[i]=level
            continue
        if mode=='smooth':
            field=level*(1+.22*math.sin(x/WIDTH*2*math.pi+phases[0])
                           +.18*math.cos(y/HEIGHT*2*math.pi+phases[1]))
        else:
            field=level*.15
            for cx,cy,sx,sy,angle,amplitude in peaks:
                dx,dy=x-cx,y-cy
                u=dx*math.cos(angle)+dy*math.sin(angle)
                v=-dx*math.sin(angle)+dy*math.cos(angle)
                field+=level*amplitude*math.exp(-.5*((u/sx)**2+(v/sy)**2))
        density[i]=5*(-math.expm1(-field/5))
    return density


def components(mask):
    seen=set(); count=0
    for i,on in enumerate(mask):
        if not on or i in seen: continue
        count+=1; seen.add(i); todo=[i]
        while todo:
            p=todo.pop(); x,y=p%WIDTH,p//WIDTH
            for j in (p-1 if x else -1,p+1 if x<WIDTH-1 else -1,p-WIDTH if y else -1,p+WIDTH if y<HEIGHT-1 else -1):
                if j>=0 and mask[j] and j not in seen: seen.add(j); todo.append(j)
    return count

def quantize(mask,density,width=30,step=.5,threshold=.5):
    height=round(width*.6); cells=[]
    for y in range(height):
        for x in range(width):
            indices=[yy*WIDTH+xx for yy in range(y*HEIGHT//height,(y+1)*HEIGHT//height) for xx in range(x*WIDTH//width,(x+1)*WIDTH//width)]
            inside=[i for i in indices if mask[i]]
            fraction=len(inside)/len(indices); mean=sum(density[i] for i in inside)/len(inside) if inside else 0
            value=int(mean/step+.5)*step
            cells.append({'on':bool(inside) and fraction>=threshold,'fraction':fraction,'raw':mean,'density':value})
    return {'width':width,'height':height,'area':WORLD_WIDTH*WORLD_HEIGHT/(width*height),'cells':cells}

@dataclass
class Rectangle:
    x:int; y:int; width:int; height:int; area:float; mean:float; error:float; score:float

def objective(area,error,weight):
    """Modify this function to change the model's objective."""
    return area-weight*error/25

def overlaps(a,b):
    return a.x<b.x+b.width and b.x<a.x+a.width and a.y<b.y+b.height and b.y<a.y+a.height

def candidates(grid,weight):
    w,h,area=grid['width'],grid['height'],grid['area']; cells=grid['cells']; result=[]
    for top in range(h):
        valid=[True]*w; sums=[0.]*w; squares=[0.]*w
        for bottom in range(top,h):
            for x in range(w):
                c=cells[bottom*w+x]; valid[x] &= c['on']; sums[x]+=c['density']; squares[x]+=c['density']**2
            for left in range(w):
                total=sq=0.
                for right in range(left,w):
                    if not valid[right]: break
                    total+=sums[right]; sq+=squares[right]
                    rw,rh=right-left+1,bottom-top+1; n=rw*rh
                    error=max(0,sq-total*total/n)*area
                    result.append(Rectangle(left,top,rw,rh,n*area,total/n,error,objective(n*area,error,weight)))
    return sorted(result,key=lambda r:(-r.score,-r.area,r.y,r.x))

def solve(grid,k=5,weight=1):
    if sum(c['on'] for c in grid['cells'])<k: raise ValueError('공간 격자 수가 구역 수보다 적습니다.')
    all_rects=candidates(grid,weight); selected=[]
    for r in all_rects:
        if all(not overlaps(r,p) for p in selected): selected.append(r)
        if len(selected)==k: break
    lookup={(r.x,r.y,r.width,r.height):r for r in all_rects}
    while len(selected)<k:
        i=next(i for i,r in enumerate(selected) if r.width*r.height>1)
        r=selected.pop(i)
        if r.width>=r.height and r.width>1:
            n=r.width//2; keys=[(r.x,r.y,n,r.height),(r.x+n,r.y,r.width-n,r.height)]
        else:
            n=r.height//2; keys=[(r.x,r.y,r.width,n),(r.x,r.y+n,r.width,r.height-n)]
        selected.extend(lookup[key] for key in keys)
    for _ in range(5):
        changed=False
        for i,current in enumerate(selected):
            for r in all_rects:
                if r.score<=current.score+1e-8: break
                if all(j==i or not overlaps(r,p) for j,p in enumerate(selected)):
                    selected[i]=r; changed=True; break
        if not changed: break
    return selected

def boundary_errors(mask,grid,rectangles):
    w,h=grid['width'],grid['height']; covered=set()
    for r in rectangles:
        covered.update((x,y) for y in range(r.y,r.y+r.height) for x in range(r.x,r.x+r.width))
    missed=excess=0
    for y in range(HEIGHT):
        for x in range(WIDTH):
            selected=(int((x+.5)*w/WIDTH),int((y+.5)*h/HEIGHT)) in covered
            missed+=mask[y*WIDTH+x] and not selected
            excess+=selected and not mask[y*WIDTH+x]
    pixel_area=WORLD_WIDTH*WORLD_HEIGHT/(WIDTH*HEIGHT)
    return missed*pixel_area,excess*pixel_area
