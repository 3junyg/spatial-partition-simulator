from dataclasses import replace
import numpy as np
import spatial_model as m

def run():
    s=m.Settings(time_limit=5);sc=m.generate(s);r=m.optimize(sc,s);m.verify(sc,r['rectangles']);assert r['unrestricted'];print('default',len(r['rectangles']),r['gap'])
    assert (sc['w'],sc['h'],sc['cell_m'])==(s.width_m,s.height_m,1)
    a=np.ones(8);z=np.zeros((8,4));z[:,0]=[0,2,0,2,0,2,0,2];y,_,_=m.standardize(z,a);toy=dict(w=4,h=2,area=a,active=a>0,z=z,y=y)
    low=m.optimize(toy,replace(s,rectangle_penalty=.001));high=m.optimize(toy,replace(s,rectangle_penalty=1))
    assert len(low['rectangles'])==4 and len(high['rectangles'])==1
    print('PASS: no 2–3-region constraint, lambda selects 4 or 1; valid cover, no overlaps')

if __name__=='__main__':run()
