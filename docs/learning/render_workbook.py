"""Render this page-structured Markdown workbook to vector PDF using Matplotlib.

No TeX, browser or extra PDF package is required. This intentionally supports
only the headings, paragraphs, lists and fenced code used by the workbook.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

WIDTH, HEIGHT = 595.28, 841.89
LEFT, RIGHT, BOTTOM = 48., 547.28, 69.
INK, MUTED, ACCENT = '#172B3A', '#536874', '#007C83'


def clean(text):
    return text.replace('**', '').replace('`', '')


def blocks(page):
    lines = page.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError('Unclosed code fence')
            yield 'code', '\n'.join(code)
        elif line.startswith('# '):
            yield 'title', line[2:]
        elif line.startswith('## '):
            yield 'heading', line[3:]
        elif line.startswith('- ') or re.match(r'^\d+\. ', line):
            yield 'item', line
        else:
            paragraph = [line]
            while i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','```','- ','https://')):
                i += 1
                paragraph.append(lines[i])
            yield 'body', ' '.join(paragraph)
        i += 1


def render(source, output, preview=None):
    pages = source.read_text().split('<!-- PAGE -->')
    matplotlib.rcParams.update({'pdf.fonttype':42, 'font.family':'DejaVu Sans'})
    output.parent.mkdir(parents=True, exist_ok=True)
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
    inventory = []
    with PdfPages(output, metadata={'Title':'Walker experiments: environment and Python workbook',
                                   'Author':'FC Portugal Walker research project',
                                   'Subject':'Setup, reproducibility and six guided learning exercises'}) as pdf:
        for number, page in enumerate(pages, 1):
            fig = plt.figure(figsize=(WIDTH/72, HEIGHT/72), dpi=144, facecolor='white')
            renderer = fig.canvas.get_renderer()
            y = 764.
            texts = []

            def width(value, size, family='DejaVu Sans', weight='normal'):
                prop = FontProperties(family=family, size=size, weight=weight)
                return renderer.get_text_width_height_descent(value, prop, False)[0]*72/fig.dpi

            def wrap(value, size, family='DejaVu Sans', weight='normal', limit=RIGHT-LEFT):
                result, current = [], ''
                for word in value.split():
                    trial = (current+' '+word).strip()
                    if width(trial,size,family,weight) <= limit:
                        current = trial
                        continue
                    if current:
                        result.append(current)
                        current = ''
                    # Split unusually long URLs/paths, but never silently wrap code.
                    while width(word,size,family,weight) > limit:
                        count = 1
                        while count < len(word) and width(word[:count+1],size,family,weight) <= limit:
                            count += 1
                        result.append(word[:count])
                        word = word[count:]
                    current = word
                if current:
                    result.append(current)
                return result or ['']

            def draw(line, size, x=LEFT, family='DejaVu Sans', weight='normal', color=INK, url=None):
                text = fig.text(x/WIDTH, y/HEIGHT, line, fontsize=size, family=family,
                                weight=weight, color=color, va='top', url=url,
                                parse_math=False)
                texts.append(text)

            fig.patches.append(Rectangle((0, 817/HEIGHT), 1, 25/HEIGHT,
                               transform=fig.transFigure, facecolor=ACCENT, edgecolor='none'))
            fig.text(LEFT/WIDTH, 798/HEIGHT, 'WALKER  /  PRACTICAL RESEARCH WORKBOOK',
                     fontsize=8, color=ACCENT, weight='bold')
            for kind, value in blocks(page):
                if kind == 'code':
                    code = value.splitlines()
                    for line in code:
                        if width(line,8.5,'DejaVu Sans Mono') > RIGHT-LEFT-18:
                            raise ValueError(f'Code line too wide on page {number}: {line}')
                    height = 11.8*len(code)+15
                    fig.patches.append(Rectangle(((LEFT-5)/WIDTH,(y-height+3)/HEIGHT),
                                       (RIGHT-LEFT+10)/WIDTH,height/HEIGHT,
                                       transform=fig.transFigure,facecolor='#EFF5F6',edgecolor='none'))
                    y -= 6
                    for line in code:
                        draw(line,8.5,x=LEFT+5,family='DejaVu Sans Mono')
                        y -= 11.8
                    y -= 12
                else:
                    value = clean(value)
                    size, leading, gap, weight, color = {
                        'title':(23,29,14,'bold',INK),
                        'heading':(12.2,17,8,'bold',ACCENT),
                        'body':(10.2,14.3,9,'normal',INK),
                        'item':(10.2,14.3,5,'normal',INK),
                    }[kind]
                    url = value if value.startswith('https://') and ' ' not in value else None
                    for line in wrap(value,size,weight=weight):
                        draw(line,size,weight=weight,color=color,url=url)
                        y -= leading
                    y -= gap
                if y < BOTTOM:
                    raise ValueError(f'Page {number} overflow: y={y:.1f}; shorten or split the source page')
            fig.text(LEFT/WIDTH, 34/HEIGHT, '28 OCT 2026  |  Repository-specific setup; preserve historical results',
                     fontsize=7.1,color=MUTED)
            fig.text(RIGHT/WIDTH, 34/HEIGHT, f'{number:02d} / {len(pages):02d}',
                     fontsize=8,color=MUTED,ha='right')
            fig.canvas.draw()
            # Check rendered text boundaries, not merely estimated line counts.
            for text in texts:
                box = text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
                if box.x1*72 > WIDTH-35 or box.y0*72 < 49:
                    raise ValueError(f'Rendered text outside content bounds on page {number}: {text.get_text()}')
            pdf.savefig(fig)
            if preview:
                fig.savefig(preview/f'page_{number:02d}.png',dpi=120)
            inventory.append(dict(page=number, title=next(v for k,v in blocks(page) if k=='title'),
                                  bottom_content_y=y))
            plt.close(fig)
    output.with_suffix('.layout.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(f'Rendered {len(pages)} pages: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).with_name('EXPERIMENT_SETUP_WORKBOOK.md'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('EXPERIMENT_SETUP_WORKBOOK.pdf'))
    parser.add_argument('--preview',type=Path)
    args = parser.parse_args()
    render(args.source,args.output,args.preview)
