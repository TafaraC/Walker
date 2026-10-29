"""Portable PDF fallback for the focused Marp source; no browser/TeX required.

Same slide text/images, with a simple measured 16:9 layout (not a Marp renderer).
Supports this deck's headings, paragraphs, lists, tables and standalone images.
Fails on overflowing text; previews are rendered from the same figure as PDF.
"""
import argparse
import json
import os
import re
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/walker-final-slides')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
W,H=1280,720
INK,TEAL,MUTED='#172B3A','#007C83','#536874'

def plain(s):
    s=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',s)
    return s.replace('**','').replace('`','')

def render(preview=None):
    raw=(HERE/'CONSOLIDATED_PAPER_MARP.md').read_text()
    raw=re.sub(r'^---\n.*?\n---\n','',raw,count=1,flags=re.S)
    raw=re.sub(r'<!--.*?-->','',raw,flags=re.S)
    slides=re.split(r'\n---\n',raw.strip())
    titles=[re.search(r'^# (.+)',s,re.M).group(1) for s in slides]
    assert [int(t.split(' · ')[0]) for t in titles]==list(range(len(slides)))
    if preview:preview.mkdir(parents=True,exist_ok=True)
    layout=[]
    matplotlib.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    target=HERE/'CONSOLIDATED_PAPER_SLIDES.pdf'
    with PdfPages(target,metadata={'Title':'Imitating RoboCup locomotion: consolidated evidence and controls'}) as pdf:
        for index,slide in enumerate(slides):
            fig=plt.figure(figsize=(W/72,H/72),dpi=90,facecolor='white')
            renderer=fig.canvas.get_renderer();artists=[]
            def width(s,size,weight='normal'):
                return renderer.get_text_width_height_descent(s,FontProperties(size=size,weight=weight),False)[0]*72/fig.dpi
            def wrap(s,size,limit,weight='normal'):
                lines=[];line=''
                for word in s.split():
                    test=(line+' '+word).strip()
                    if width(test,size,weight)>limit and line:lines.append(line);line=word
                    else:line=test
                if line:lines.append(line)
                return lines
            def put(s,x,y,size=25,color=INK,weight='normal',limit=1150,leading=1.35):
                lines=wrap(plain(s),size,limit,weight)
                for line in lines:
                    a=fig.text(x/W,y/H,line,fontsize=size,ha='left',va='top',color=color,weight=weight,parse_math=False)
                    artists.append(a);y-=size*leading
                return y
            fig.patches.append(Rectangle((0,0.97),1,.03,transform=fig.transFigure,color=TEAL))
            title=titles[index]
            y=put(title,60,662,34,weight='bold')-16
            blocks=re.split(r'\n\s*\n',slide.strip())
            for b in blocks[1:]:
                b=b.strip()
                if not b:continue
                if b.startswith('!['):
                    path=re.fullmatch(r'!\[[^\]]*\]\(([^)]+)\)',b).group(1)
                    img=plt.imread(HERE/path)
                    height=1120*img.shape[0]/img.shape[1]
                    ax=fig.add_axes([80/W,(y-height)/H,1120/W,height/H])
                    ax.imshow(img);ax.axis('off');y-=height+12
                elif b.startswith('|'):
                    rows=[line.strip().strip('|').split('|') for line in b.splitlines()]
                    rows=[r for i,r in enumerate(rows) if i!=1]
                    cols=len(rows[0]);cw=1150/cols
                    for ri,row in enumerate(rows):
                        size=20 if cols>=4 else 23
                        wrapped=[wrap(plain(c.strip()),size,cw-22,'bold' if ri==0 else 'normal') for c in row]
                        height=max(map(len,wrapped))*size*1.2+18
                        fig.patches.append(Rectangle((60/W,(y-height)/H),1150/W,height/H,
                            transform=fig.transFigure,color='#DAECEC' if ri==0 else ('#F0F4F6' if ri%2 else '#FFFFFF')))
                        for j,ls in enumerate(wrapped):
                            for k,line in enumerate(ls):
                                artists.append(fig.text((71+j*cw)/W,(y-9-k*size*1.2)/H,line,fontsize=size,
                                    va='top',color=INK,weight='bold' if ri==0 else 'normal',parse_math=False))
                        y-=height
                    y-=18
                else:
                    if b.startswith('## '):y=put(b[3:],60,y,27,color=TEAL,weight='bold')-14
                    elif b.startswith('- '):
                        for line in b.splitlines():
                            y=put('• '+line[2:],70,y,23,limit=1130)-8
                    else:
                        size=22 if '![' in slide else 25
                        weight='bold' if b.startswith('**') and b.endswith('**') else 'normal'
                        y=put(' '.join(b.splitlines()),60,y,size,weight=weight)-13
            if y<50:raise ValueError(f'Slide {index} content too low: {y}')
            fig.text(60/W,25/H,'Consolidated study · 28 October 2026 · See companion notes for sources and limits',fontsize=12,color=MUTED)
            fig.text(1210/W,25/H,f'{index+1} / {len(slides)}',ha='right',fontsize=12,color=MUTED)
            fig.canvas.draw()
            for a in artists:
                box=a.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
                if box.x0<0 or box.x1>1 or box.y0<.06 or box.y1>1:
                    raise ValueError(f'Slide {index} text outside content bounds: {a.get_text()}')
            pdf.savefig(fig)
            if preview:fig.savefig(preview/f'slide_{index:02d}.png',dpi=90)
            layout.append({'slide':index,'title':title,'bottom_y':round(y,1)})
            plt.close(fig)
    (HERE/'CONSOLIDATED_PAPER_SLIDES.layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    print(f'Rendered {len(slides)} slides: {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',type=Path)
    render(p.parse_args().preview)
