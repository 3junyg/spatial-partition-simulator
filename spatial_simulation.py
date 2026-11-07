"""Desktop UI for the static spatial partition model. Run with Python 3.10+."""
import json, math, queue, random, threading, time, traceback
from dataclasses import asdict, replace
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import spatial_model as m

PALETTE=['#ffffff','#fff2a8','#ffc43d','#f48624','#d93c30','#8f174e','#410b65']
OUTSIDE='#abb5c2'
VIEWS={'밀도 (명/m²)':'density','유입 · 평활화 전':'fin0','유입 · 평활화 후':'fin','유출 · 평활화 전':'fout0','유출 · 평활화 후':'fout','환경 지표 d_E':'environment','기온 (°C)':'temperature','바람 크기 (m/s)':'wind','일사량 (W/m²)':'sun','상대습도 (%)':'humidity'}

class Application:
    def __init__(self,root):
        self.root=root;self.scene=None;self.solution=None;self.busy=False;self.messages=queue.Queue();self.hover=None;self.loaded_settings=None
        root.title('직사각형 분할 실험실 · 밀도 / 유입 / 유출 / 환경');root.geometry(f'{min(1450,root.winfo_screenwidth()-50)}x{min(900,root.winfo_screenheight()-90)}');root.minsize(1000,650)
        style=ttk.Style();style.theme_use('clam');style.configure('TFrame',background='#edf2f5');style.configure('TLabel',background='#edf2f5',foreground='#203447');style.configure('Title.TLabel',font=('맑은 고딕',16,'bold'));style.configure('TButton',padding=6);style.configure('Primary.TButton',background='#146b61',foreground='white');style.configure('TNotebook.Tab',padding=(12,6))
        self.status=tk.StringVar(value='입력 생성 후 계산하세요.');self.metrics=tk.StringVar();self.details=tk.StringVar();self.legend=tk.StringVar();self.info=tk.StringVar()
        defaults=asdict(m.Settings());self.vars={k:tk.StringVar(value=str(v)) for k,v in defaults.items()};self.vars['preset'].set(m.PRESETS[defaults['preset']]);self.vars['distribution'].set(m.DENSITY_MODES[defaults['distribution']])
        self.vars['size']=tk.StringVar(value='48 × 32 m');self.view=tk.StringVar(value=next(iter(VIEWS)))
        outer=ttk.Frame(root,padding=12);outer.pack(fill='both',expand=True)
        sidebar=ttk.Frame(outer);sidebar.pack(side='left',fill='y',padx=(0,12));scrollcanvas=tk.Canvas(sidebar,width=276,highlightthickness=0,bg='#edf2f5');bar=ttk.Scrollbar(sidebar,orient='vertical',command=scrollcanvas.yview);scrollcanvas.configure(yscrollcommand=bar.set);bar.pack(side='right',fill='y');scrollcanvas.pack(side='left',fill='both',expand=True)
        self.controls=ttk.Frame(scrollcanvas);scrollcanvas.create_window((0,0),window=self.controls,anchor='nw',width=267);self.controls.bind('<Configure>',lambda e:scrollcanvas.configure(scrollregion=scrollcanvas.bbox('all')))
        self.inputs=[];self.buttons=[]
        ttk.Label(self.controls,text='공간 분할 실험실',style='Title.TLabel').pack(anchor='w',pady=(0,10))
        self.heading('01 / 시드와 예시 관측')
        self.field('장소 유형','preset',list(m.PRESETS.values()));self.field('가상 공간 크기 · 기본 셀 2m × 2m','size',['32 × 20 m','48 × 32 m','64 × 40 m'])
        self.field('구조 시드','shape_seed');self.field('사람 시드','people_seed');self.field('유입·유출 시드','flow_seed');self.field('환경 시드','environment_seed');self.field('초기 인원 분포','distribution',list(m.DENSITY_MODES.values()));self.field('평균 인원 생성 기준 (명/m²)','density_level')
        self.field('대표 관측 시간 (분)','minutes');self.button('① 가상 관측 생성',self.generate,True);self.button('시드 모두 무작위로 변경',self.randomize)
        self.heading('02 / 평활화·분할 기준')
        self.field('유입·유출 평활화 λf (0–20)','smoothing');self.field('밀도 가중치 (0–1)','density_weight');self.field('특성 차이 경고 기준 (배율)','tolerance');self.field('구역 추가 비용 λM (직접 입력 가능)','rectangle_penalty');self.field('돌출 비용 β (0–10)','outside_penalty');self.field('최적화 시간 제한 (초, 1–300)','time_limit')
        self.button('② 설정 적용 · 격자 특성 갱신',self.refresh);self.button('③ 직사각형 최적화',self.solve,True);self.button('실험 JSON 저장',self.save);self.button('실험 JSON 열기',self.load);self.button('모델 설명 보기',self.explain)
        ttk.Label(self.controls,textvariable=self.status,wraplength=258,foreground='#076d56').pack(anchor='w',pady=10)
        ttk.Label(self.controls,text='정적인 가상 관측 데이터입니다.\n실제 시설 도면·실측값이 아닙니다.\n열·기류 물리 방정식을 풀지 않습니다.',wraplength=258,foreground='#617183').pack(anchor='w',pady=6)
        main=ttk.Frame(outer);main.pack(side='left',fill='both',expand=True)
        header=ttk.Frame(main);header.pack(fill='x');ttk.Label(header,text='특성이 비슷한 곳끼리, 필요한 수만큼.',style='Title.TLabel').pack(side='left');ttk.Label(main,textvariable=self.info).pack(anchor='w',pady=7)
        tools=ttk.Frame(main);tools.pack(fill='x');ttk.Label(tools,text='색상으로 볼 정보').pack(side='left');select=ttk.Combobox(tools,textvariable=self.view,values=list(VIEWS),state='readonly',width=24);select.pack(side='left',padx=8);select.bind('<<ComboboxSelected>>',lambda e:self.draw_all());self.viewbox=select
        quicksolve=ttk.Button(tools,text='직사각형 계산 →',command=self.solve,style='Primary.TButton');quicksolve.pack(side='left',padx=10);self.buttons.append(quicksolve)
        lambdabox=ttk.LabelFrame(main,text='분할 세밀함 조절 · λM',padding=(10,5));lambdabox.pack(fill='x',pady=6)
        lambdarow=ttk.Frame(lambdabox);lambdarow.pack(fill='x')
        ttk.Label(lambdarow,text='람다 직접 입력').pack(side='left')
        le=ttk.Entry(lambdarow,textvariable=self.vars['rectangle_penalty'],width=9);le.pack(side='left',padx=7);self.inputs.append(le)
        ttk.Label(lambdarow,text='낮을수록 세밀하게 / 높을수록 단순하게').pack(side='left',padx=5)
        for label,value in [('세밀 0.02',.02),('중간 0.08',.08),('단순 0.20',.20)]:
            b=ttk.Button(lambdarow,text=label,command=lambda value=value:self.set_lambda(value));b.pack(side='left',padx=3);self.buttons.append(b)
        b=ttk.Button(lambdarow,text='람다 3개 비교',command=self.compare_lambdas);b.pack(side='left',padx=5);self.buttons.append(b)
        self.lambda_position=tk.DoubleVar(value=math.log10(float(self.vars['rectangle_penalty'].get())))
        self._lambda_sync=False
        self.lambda_slider=ttk.Scale(lambdabox,from_=-3,to=0,orient='horizontal',variable=self.lambda_position,command=self.lambda_slid);self.lambda_slider.pack(fill='x',pady=3);self.inputs.append(self.lambda_slider)
        self.vars['rectangle_penalty'].trace_add('write',self.lambda_edited)
        ttk.Label(main,textvariable=self.legend,foreground='#4f6476').pack(anchor='w',pady=5)
        self.notebook=ttk.Notebook(main);self.notebook.pack(fill='both',expand=True,pady=6);self.canvases={}
        for key,title in [('raw','01 원본 공간·인원'),('grid','02 4m² 격자 특성'),('result','03 최적화 분할')]:
            frame=ttk.Frame(self.notebook);self.notebook.add(frame,text=title);c=tk.Canvas(frame,width=800,height=210,bg=OUTSIDE,highlightthickness=0);c.pack(fill='both',expand=True);c.bind('<Configure>',lambda e,key=key:self.draw(key));c.bind('<Motion>',self.inspect);self.canvases[key]=c
        ttk.Label(main,text='흰 테두리+검은 실선: 실제 공간 / 색 점선+번호: 직사각형 / 사선: 공간 밖 돌출',foreground='#53677a').pack(anchor='w')
        ttk.Label(main,textvariable=self.metrics,font=('맑은 고딕',10,'bold'),wraplength=1050).pack(anchor='w',pady=8)
        tableframe=ttk.Frame(main);tableframe.pack(fill='x')
        self.table=ttk.Treeview(tableframe,columns=('id','area','outside','density','in','out','env','error'),show='headings',height=5)
        for key,label,width in [('id','구역',50),('area','실제 면적 m²',110),('outside','돌출 m²',90),('density','밀도 명/m²',110),('in','평활 유입',95),('out','평활 유출',95),('env','환경 지표',95),('error','정규화 오차',110)]:self.table.heading(key,text=label);self.table.column(key,width=width,anchor='center',stretch=True)
        tablescroll=ttk.Scrollbar(tableframe,orient='vertical',command=self.table.yview);self.table.configure(yscrollcommand=tablescroll.set);tablescroll.pack(side='right',fill='y');self.table.pack(side='left',fill='x',expand=True);self.table.bind('<<TreeviewSelect>>',self.select_region)
        ttk.Label(main,textvariable=self.details,wraplength=1050,foreground='#334e65').pack(anchor='w',pady=8)
        root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close);self.generate()

    def heading(self,text):ttk.Label(self.controls,text=text,font=('맑은 고딕',10,'bold')).pack(anchor='w',pady=(12,5))
    def field(self,label,key,choices=None):
        ttk.Label(self.controls,text=label).pack(anchor='w',pady=(5,1));c=ttk.Combobox(self.controls,textvariable=self.vars[key],values=choices,state='readonly') if choices else ttk.Entry(self.controls,textvariable=self.vars[key]);c.pack(fill='x');self.inputs.append(c)
    def button(self,text,action,primary=False):
        b=ttk.Button(self.controls,text=text,command=action,style='Primary.TButton' if primary else 'TButton');b.pack(fill='x',pady=3);self.buttons.append(b)
    def settings(self):
        d={k:v.get() for k,v in self.vars.items() if k!='size'};d['preset']=next(k for k,v in m.PRESETS.items() if v==d['preset']);d['distribution']=next(k for k,v in m.DENSITY_MODES.items() if v==d['distribution']);d['width_m'],d['height_m']=map(int,self.vars['size'].get().replace(' m','').split(' × '))
        for key in ['density_level','minutes','smoothing','density_weight','tolerance','rectangle_penalty','outside_penalty','time_limit']:d[key]=float(d[key])
        s=m.Settings(**d);m.validate(s);return s
    def update_vars(self,s):
        for k,v in asdict(s).items():self.vars[k].set(str(v))
        self.vars['preset'].set(m.PRESETS[s.preset]);self.vars['distribution'].set(m.DENSITY_MODES[s.distribution]);self.vars['size'].set(f'{s.width_m} × {s.height_m} m')
    def generate(self):
        if self.busy:return
        try:s=self.settings();self.scene=m.generate(s);self.loaded_settings=s;self.scene_settings=asdict(s);self.clear_result();self.status.set('가상 인원·관측 생성 및 평활화 완료.');self.info.set(f"{s.width_m}m × {s.height_m}m 가상 영역 · {self.scene['w']} × {self.scene['h']} 격자 · 실제 면적 {self.scene['area'].sum():.1f}m² · {self.scene['counts'].sum()}명");self.draw_all()
        except Exception as e:messagebox.showerror('입력 오류',str(e))
    def refresh(self):
        # Regenerate with the same seeds: observations stay identical if only lambda changes.
        self.generate()
    def randomize(self):
        for k in ['shape_seed','people_seed','flow_seed','environment_seed']:self.vars[k].set(str(random.SystemRandom().randrange(2**32)))
        self.generate()
    def clear_result(self):
        self.solution=None;self.hover=None;self.table.delete(*self.table.get_children());self.metrics.set('구역 수 제한 없음 · λM으로 오차와 구역 수를 절충합니다. 기본 λM=0.08');self.details.set('격자 위에 마우스를 올리면 개별 정보를 볼 수 있습니다.')
    def solve(self):
        if self.busy:return
        try:
            s=self.settings()
            if self.scene is None or asdict(s)!=self.scene_settings:self.generate()
            if self.scene is None or asdict(s)!=self.scene_settings:return
        except Exception as e:messagebox.showerror('설정 오류',str(e));return
        self.busy=True
        for c in self.inputs:c.configure(state='disabled')
        for b in self.buttons:b.state(['disabled'])
        self.status.set('후보 생성 중…');self.clear_result()
        def worker():
            try:r=m.optimize(self.scene,s,lambda msg:self.messages.put(('status',msg)));self.messages.put(('done',r))
            except Exception:self.messages.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        last=None
        while not self.messages.empty():
            typ,value=self.messages.get()
            if typ=='status':last=value
            else:
                last=None;self.busy=False
                for c in self.inputs:c.configure(state='readonly' if isinstance(c,ttk.Combobox) else 'normal')
                for b in self.buttons:b.state(['!disabled'])
                if typ=='error':self.status.set('계산 오류');messagebox.showerror('계산 오류',value)
                elif typ=='comparison':self.show_comparison(value)
                else:self.finished(value)
        if last:self.status.set(last)
        self.root.after(100,self.poll)
    def finished(self,result):
        self.solution=result;self.table.delete(*self.table.get_children())
        for i,r in enumerate(result['rectangles']):self.table.insert('', 'end',iid=str(i),values=(i+1,f"{r['inside']:.2f}",f"{r['outside']:.2f}",*(f'{v:.3f}' for v in r['mean']),f"{r['error']/result['baseline'] if result['baseline']>1e-12 else 0:.5f}"))
        gap='미확보' if result['gap'] is None else f"{100*result['gap']:.2f}%"
        self.status.set(f"{result['status']} · {result['seconds']:.2f}초 · 최적성 간격 {gap}")
        self.metrics.set(f"구역 {len(result['rectangles'])}개 | 실제 공간 커버 100% | 잔여 특성오차 {result['error_ratio']:.2%} | 돌출 {result['outside']:.2f}m² | 목적함수 {result['objective']:.4f} | 후보 {result['candidate_count']:,}개")
        self.details.set(f"큰 특성 차이가 남은 구역: {result.get('heterogeneity_warnings',0)}개. 적은 구역 수에 따른 근사 오차이며 기존의 엄격한 차이 제한을 만족한다는 뜻은 아닙니다.");self.notebook.select(2);self.draw_all()
    def set_lambda(self,value):
        self.vars['rectangle_penalty'].set(f'{value:.4f}')
        self.status.set(f'λM={value:.4f} 설정. 계산 버튼을 누르면 적용됩니다.')

    def lambda_slid(self,value):
        if self._lambda_sync:return
        self._lambda_sync=True
        try:self.vars['rectangle_penalty'].set(f'{10**float(value):.4f}')
        finally:self._lambda_sync=False
        self.status.set('람다 설정 변경 · 계산 버튼으로 다시 계산하세요.')

    def lambda_edited(self,*_):
        if self._lambda_sync:return
        try:value=float(self.vars['rectangle_penalty'].get())
        except ValueError:return
        if .001<=value<=1:
            self._lambda_sync=True
            try:self.lambda_position.set(math.log10(value))
            finally:self._lambda_sync=False

    def compare_lambdas(self):
        if self.busy:return
        try:
            s=self.settings()
            if self.scene is None or asdict(s)!=self.scene_settings:self.generate()
            if self.scene is None or asdict(s)!=self.scene_settings:return
        except Exception as e:messagebox.showerror('설정 오류',str(e));return
        center=s.rectangle_penalty
        values=sorted(set([max(.001,center/2),center,min(1,center*2)]))
        self.busy=True
        for c in self.inputs:c.configure(state='disabled')
        for b in self.buttons:b.state(['disabled'])
        def worker():
            try:
                results=[]
                for value in values:
                    self.messages.put(('status',f'같은 원본으로 λ={value:.4f} 비교 중…'))
                    result=m.optimize(self.scene,replace(s,rectangle_penalty=value,time_limit=min(s.time_limit,8)),lambda msg:self.messages.put(('status',f'λ={value:.4f} · '+msg)))
                    results.append(result)
                self.messages.put(('comparison',results))
            except Exception:self.messages.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def show_comparison(self,results):
        win=tk.Toplevel(self.root);win.title('같은 공간 · 람다별 분할 비교');win.geometry('850x340');win.transient(self.root);win.grab_set()
        ttk.Label(win,text='행을 선택한 다음 적용하면 해당 분할 결과를 바로 표시합니다. 비교 계산은 값당 최대 8초입니다.',wraplength=800).pack(padx=12,pady=12)
        table=ttk.Treeview(win,columns=('lambda','count','error','outside','gap'),show='headings',height=4)
        for key,label in [('lambda','람다 λM'),('count','구역 수'),('error','잔여 특성오차'),('outside','돌출 면적 m²'),('gap','최적성 간격')]:table.heading(key,text=label);table.column(key,width=150,anchor='center')
        table.pack(fill='x',padx=12)
        for i,r in enumerate(results):table.insert('', 'end',iid=str(i),values=(f"{r['settings']['rectangle_penalty']:.4f}",len(r['rectangles']),f"{r['error_ratio']:.1%}",f"{r['outside']:.1f}",'미확보' if r['gap'] is None else f"{r['gap']:.1%}"))
        table.selection_set(str(min(range(len(results)),key=lambda i:abs(results[i]['settings']['rectangle_penalty']-float(self.vars['rectangle_penalty'].get())))))
        def apply():
            selected=table.selection()
            if not selected:return
            r=results[int(selected[0])]
            # The dialog is modal, so the source scene cannot change while choosing.
            self.update_vars(m.Settings(**r['settings']));self.scene_settings=r['settings'];self.finished(r);win.destroy()
        ttk.Button(win,text='선택한 람다와 결과 적용',command=apply,style='Primary.TButton').pack(pady=12)
        ttk.Label(win,text='구역 수 제한은 없습니다. 숫자는 이번 원본과 설정에 대한 결과이며, 값이 작을 때도 단순한 공간은 적게 나뉠 수 있습니다.',wraplength=800).pack(padx=12)
        self.status.set('람다 비교 완료 · 비교 창에서 결과를 선택하세요.')

    def values(self):
        key=VIEWS[self.view.get()]
        if key=='temperature':return self.scene['env'][:,0]
        if key=='wind':return np.linalg.norm(self.scene['env'][:,1:3],axis=1)
        if key=='sun':return self.scene['env'][:,3]
        if key=='humidity':return self.scene['env'][:,4]
        return self.scene[key]
    @staticmethod
    def color(t):
        t=max(0,min(1,float(t)))*(len(PALETTE)-1);i=min(int(t),len(PALETTE)-2);f=t-i;a=PALETTE[i];b=PALETTE[i+1];return '#'+''.join(f'{round(int(a[j:j+2],16)*(1-f)+int(b[j:j+2],16)*f):02x}' for j in (1,3,5))
    def transform(self,c):
        W=max(1,c.winfo_width());H=max(1,c.winfo_height());sx=min((W-32)/self.scene['w'],(H-32)/self.scene['h']);return sx,(W-self.scene['w']*sx)/2,(H-self.scene['h']*sx)/2
    def draw_all(self):
        if self.scene is None:return
        a=self.values()[self.scene['active']];lo=float(a.min());hi=float(a.max());self.legend.set(f'{self.view.get()} · 색상 범위 {lo:.3f} → {hi:.3f} (현재 공간 기준) · 흰색=낮음 / 진한 보라=높음')
        for key in self.canvases:self.draw(key)
    def draw(self,key):
        if self.scene is None:return
        c=self.canvases[key];c.delete('all');sc,ox,oy=self.transform(c);w,h=self.scene['w'],self.scene['h'];vals=self.values();active=self.scene['active'];lo=float(vals[active].min());hi=float(vals[active].max());span=hi-lo
        if key=='raw':
            mask=self.scene['mask'];rh,rw=mask.shape;ss=sc/6
            for yy in range(rh):
                x=0
                while x<rw:
                    if not mask[yy,x]:x+=1;continue
                    start=x
                    while x<rw and mask[yy,x]:x+=1
                    c.create_rectangle(ox+start*ss,oy+yy*ss,ox+x*ss,oy+(yy+1)*ss,fill='#f9fcfa',outline='')
            for x,y in self.scene['people']:c.create_oval(ox+x*w*sc-2,oy+y*h*sc-2,ox+x*w*sc+2,oy+y*h*sc+2,fill='#172c42',outline='')
        else:
            for i,on in enumerate(active):
                if not on:continue
                yy,xx=divmod(i,w);v=(vals[i]-lo)/span if span>1e-10 else 0
                c.create_rectangle(ox+xx*sc,oy+yy*sc,ox+(xx+1)*sc,oy+(yy+1)*sc,fill=self.color(v),outline='#65768b',width=.5)
                if key=='grid' and sc>=24:
                    label=str(self.scene['counts'][i]) if VIEWS[self.view.get()]=='density' else f'{vals[i]:.1f}'
                    c.create_text(ox+(xx+.5)*sc,oy+(yy+.5)*sc,text=label,fill='white' if v>.6 else '#14283b',font=('맑은 고딕',8))
            if key=='result' and self.solution:
                for j,r in enumerate(self.solution['rectangles']):
                    x=ox+r['x']*sc;y=oy+r['y']*sc;right=x+r['w']*sc;bottom=y+r['h']*sc;color='#%02x%02x%02x'%tuple(int(v*255) for v in __import__('colorsys').hsv_to_rgb((j*.618)%1,.85,.65))
                    indices=[yy*w+xx for yy in range(r['y'],r['y']+r['h']) for xx in range(r['x'],r['x']+r['w'])]
                    avg=float(np.dot(self.scene['area'][indices],vals[indices])/r['inside'])
                    extension=self.color((avg-lo)/span if span>1e-10 else 0)
                    # Hatch only outside cells, never treat them as observed data.
                    for yy in range(r['y'],r['y']+r['h']):
                        for xx in range(r['x'],r['x']+r['w']):
                            if not active[yy*w+xx]:
                                c.create_rectangle(ox+xx*sc,oy+yy*sc,ox+(xx+1)*sc,oy+(yy+1)*sc,fill=extension,outline='')
                                c.create_line(ox+xx*sc+2,oy+yy*sc+2,ox+(xx+1)*sc-2,oy+(yy+1)*sc-2,fill='#67768b')
                    c.create_rectangle(x+2,y+2,right-2,bottom-2,outline='white',width=5)
                    c.create_rectangle(x+2,y+2,right-2,bottom-2,outline=color,width=2,dash=(6,3))
                    c.create_rectangle(x+2,y+2,x+23,y+20,fill=color,outline='');c.create_text(x+12,y+11,text=str(j+1),fill='white',font=('맑은 고딕',8,'bold'))
                    if self.hover==j:c.create_rectangle(x+1,y+1,right-1,bottom-1,outline='#00e5ff',width=5)
        self.boundary(c,sc,ox,oy)
    def boundary(self,c,sc,ox,oy):
        mask=self.scene['mask'];h,w=mask.shape;ss=sc/6;segments=[]
        for y,x in np.argwhere(mask):
            if y==0 or not mask[y-1,x]:segments.append((x,y,x+1,y))
            if y==h-1 or not mask[y+1,x]:segments.append((x,y+1,x+1,y+1))
            if x==0 or not mask[y,x-1]:segments.append((x,y,x,y+1))
            if x==w-1 or not mask[y,x+1]:segments.append((x+1,y,x+1,y+1))
        for color,width in [('white',5),('#101c2b',3)]:
            for a,b,d,e in segments:c.create_line(ox+a*ss,oy+b*ss,ox+d*ss,oy+e*ss,fill=color,width=width,capstyle='projecting',tags='boundary')
    def inspect(self,event):
        if self.scene is None:return
        sc,ox,oy=self.transform(event.widget);x=int(math.floor((event.x-ox)/sc));y=int(math.floor((event.y-oy)/sc));w,h=self.scene['w'],self.scene['h']
        if not 0<=x<w or not 0<=y<h:return
        i=y*w+x
        if not self.scene['active'][i]:self.details.set(f'격자 ({x+1}, {y+1}) · 실제 공간 밖');return
        z=self.scene['z'][i];self.details.set(f"격자 ({x+1}, {y+1}) | 실제 면적 {self.scene['area'][i]:.2f}m² | {self.scene['counts'][i]}명 | 밀도 {z[0]:.3f}명/m² | 평활 유입 {z[1]:.3f} / 유출 {z[2]:.3f}명/(m²·분) | 환경 d_E {z[3]:.3f}")
    def select_region(self,event):
        selected=self.table.selection();self.hover=int(selected[0]) if selected else None;self.draw('result')
    def save(self):
        if self.scene is None:return
        path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('실험 JSON','*.json')])
        if not path:return
        data={'version':3,'settings':self.scene_settings,'observations':{key:self.scene[key].tolist() for key in ['mask','counts','cin','cout','env']},'solution':self.solution}
        Path(path).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8');self.status.set('실험과 관측값·계산 결과를 저장했습니다.')
    def load(self):
        path=filedialog.askopenfilename(filetypes=[('실험 JSON','*.json')])
        if not path:return
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if data.get('version')!=3:raise ValueError('이번 버전에서 저장한 v3 파일만 열 수 있습니다.')
            s=m.Settings(**data['settings']);m.validate(s);self.update_vars(s);self.generate()
            obs=data['observations']
            for key in ['mask','counts','cin','cout','env']:
                a=np.array(obs[key]);expected=self.scene[key]
                if a.shape!=expected.shape or not np.array_equal(a,expected):raise ValueError('저장된 관측값이 시드·설정과 일치하지 않습니다.')
            if data.get('solution'):
                r=data['solution']
                if r.get('unrestricted'):
                    m.verify(self.scene,r['rectangles']);self.finished(r)
                else:self.status.set('이전 버전 결과입니다. 계산 버튼으로 개수 제한 없이 다시 구하세요.')
        except Exception as e:messagebox.showerror('열기 오류',str(e))
    def explain(self):
        win=tk.Toplevel(self.root);win.title('모델 정의와 계산 방법');win.geometry('820x700');txt=tk.Text(win,wrap='word',font=('맑은 고딕',10),padx=16,pady=16);bar=ttk.Scrollbar(win,command=txt.yview);txt.configure(yscrollcommand=bar.set);bar.pack(side='right',fill='y');txt.pack(fill='both',expand=True);txt.insert('1.0',Path(__file__).with_name('MODEL.md').read_text(encoding='utf-8'));txt.configure(state='disabled')
    def close(self):
        if self.busy and not messagebox.askyesno('계산 중','창을 닫으면 진행 중인 결과가 저장되지 않습니다. 닫을까요?'):return
        self.root.destroy()

if __name__=='__main__':
    root=tk.Tk();Application(root);root.mainloop()
