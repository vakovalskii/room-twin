"""Карта высот для расстановки мебели агентом: сверху, клетка 5 см, цвет — самая высокая точка в клетке.

  python scripts/heightmap.py out/<скан>      # → out/<скан>/height.png

Синее — низкое (пол, кровать ~0.5), зелёное — 1 м (столы с мониторами), оранжевое — 2 м (шкафы, стены).
Оси и стены как на plan_room.png: x вдоль S, y вдоль W, красный кружок — точка съёмки.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import numpy as np, json
from PIL import Image, ImageDraw
from pathlib import Path
from cloud import font
o=Path(sys.argv[1]); outp=sys.argv[2] if len(sys.argv)>2 else str(o/'height.png')
zlo=float(sys.argv[3]) if len(sys.argv)>3 else 0.08; zhi=float(sys.argv[4]) if len(sys.argv)>4 else 2.0
raw=open(o/'cloud.ply','rb').read(); i=raw.index(b'end_header\n')+11
rec=np.frombuffer(raw[i:],dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
P=json.load(open(o/'poses.json')); W,D=P['room']['width'],P['room']['depth']
print(P['room'])
cell=0.05; nx,ny=int(W/cell)+1,int(D/cell)+1
H=np.zeros((ny,nx))
x,y,z=rec['x'],rec['y'],rec['z']
sel=(z>zlo)&(z<zhi)&(x>=0)&(y>=0)&(x<W)&(y<D)
np.maximum.at(H,((y[sel]/cell).astype(int),(x[sel]/cell).astype(int)),z[sel])
s=12; pad=50
img=Image.new('RGB',(nx*s+2*pad,ny*s+2*pad),(252,252,250)); dr=ImageDraw.Draw(img)
cm=lambda t:(int(30+220*t),int(120+100*(1-abs(t-.5)*2)),int(230-200*t))
for j in range(ny):
    for k in range(nx):
        if H[j,k]>0:
            t=min(H[j,k]/2.0,1); dr.rectangle([pad+k*s,pad+(ny-1-j)*s,pad+k*s+s-1,pad+(ny-1-j)*s+s-1],fill=cm(t))
for g in np.arange(0,max(W,D)+0.01,0.5):
    c=(120,120,120) if g%1==0 else (190,190,190)
    if g<=W: dr.line([(pad+g/cell*s,pad),(pad+g/cell*s,pad+ny*s)],fill=c)
    if g<=D: dr.line([(pad,pad+ny*s-g/cell*s),(pad+nx*s,pad+ny*s-g/cell*s)],fill=c)
    if g%1==0:
        if g<=W: dr.text((pad+g/cell*s-4,pad+ny*s+8),f"{g:.0f}",fill=(0,0,0),font=font(16))
        if g<=D: dr.text((pad-25,pad+ny*s-g/cell*s-8),f"{g:.0f}",fill=(0,0,0),font=font(16))
cx,cy=P['room']['camera_xy']; dr.ellipse([pad+cx/cell*s-7,pad+ny*s-cy/cell*s-7,pad+cx/cell*s+7,pad+ny*s-cy/cell*s+7],outline=(255,0,0),width=3)
for t,l in [(0,'0'),(0.25,'0.5'),(0.5,'1.0'),(0.75,'1.5'),(1,'2.0 м')]:
    dr.rectangle([pad+int(t*300),8,pad+int(t*300)+20,28],fill=cm(t)); dr.text((pad+int(t*300)+24,10),l,fill=(0,0,0),font=font(14))
for lab,(px_,py_) in {'S':(nx*s/2,ny*s+30),'N':(nx*s/2,-40),'W':(-45,ny*s/2),'E':(nx*s+10,ny*s/2)}.items():
    dr.text((pad+px_,pad+py_),lab,fill=(0,0,0),font=font(20))
img.save(outp)
print(outp)
