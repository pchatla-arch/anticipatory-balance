#!/usr/bin/env python3
"""Build the review PDF from paper.md with ReportLab and typeset math assets.

Documentation-only dependencies: Python reportlab, matplotlib, Pillow, pypdf;
Pandoc must be on PATH. No research-environment installation is performed.
"""
from pathlib import Path
import hashlib, html, io, json, math, re, subprocess, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import mathtext, font_manager
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
 Spacer, Image, Table, TableStyle, KeepTogether, PageBreak, CondPageBreak, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'.build'
CACHE.mkdir(exist_ok=True)
(ROOT/'figures').mkdir(exist_ok=True)
WIDTH=528
FONTDIR=Path('/usr/share/fonts/truetype/liberation')
for name,file in [('Body','LiberationSans-Regular.ttf'),('Body-Bold','LiberationSans-Bold.ttf'),('Body-Italic','LiberationSans-Italic.ttf'),('Body-BoldItalic','LiberationSans-BoldItalic.ttf'),('Mono','LiberationMono-Regular.ttf')]:
 pdfmetrics.registerFont(TTFont(name,str(FONTDIR/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic',boldItalic='Body-BoldItalic')
INK=colors.HexColor('#17252B')
MUTED=colors.HexColor('#59646B')
BLUE=colors.HexColor('#1B5367')
LINE=colors.HexColor('#D9D9D9')
styles={
 'title':ParagraphStyle('Title',fontName='Body-Bold',fontSize=24,leading=27,textColor=colors.black,spaceAfter=9),
 'subtitle':ParagraphStyle('Subtitle',fontName='Body',fontSize=14.2,leading=18.3,textColor=colors.black,spaceAfter=13),
 'body':ParagraphStyle('Body',fontName='Body',fontSize=10.6,leading=14.4,textColor=INK,spaceAfter=7,autoLeading='max',splitLongWords=True,allowWidows=0,allowOrphans=0),
 'meta':ParagraphStyle('Meta',fontName='Body',fontSize=9.4,leading=12.5,textColor=MUTED,spaceAfter=9),
 'h2':ParagraphStyle('Heading2',fontName='Body-Bold',fontSize=15,leading=18.5,textColor=colors.black,spaceBefore=17,spaceAfter=8,keepWithNext=True),
 'h3':ParagraphStyle('Heading3',fontName='Body-Bold',fontSize=11.5,leading=14.8,textColor=colors.black,spaceBefore=11,spaceAfter=6,keepWithNext=True),
 'table':ParagraphStyle('Table',fontName='Body',fontSize=8.9,leading=11.6,textColor=INK,spaceAfter=0,autoLeading='max',splitLongWords=True),
 'tablehead':ParagraphStyle('TableHead',fontName='Body-Bold',fontSize=9,leading=11.7,textColor=colors.white,spaceAfter=0),
 'caption':ParagraphStyle('Caption',fontName='Body-Italic',fontSize=9,leading=12,textColor=MUTED,spaceBefore=5,spaceAfter=12),
 'code':ParagraphStyle('Code',fontName='Mono',fontSize=8.6,leading=11.3,textColor=INK,backColor=colors.HexColor('#F3F5F6'),borderPadding=8,spaceBefore=5,spaceAfter=12),
 'ref':ParagraphStyle('Reference',fontName='Body',fontSize=9,leading=12.1,textColor=INK,spaceAfter=8,splitLongWords=True),
}
math_warnings=[]
def clean_math(s):
 s=' '.join(s.split())
 s=re.sub(r'\\(?:big|Big|bigg|Bigg)[lr]?', '', s)
 s=re.sub(r'\\le(?![A-Za-z])', lambda m:r'\leq', s)
 s=re.sub(r'\\ge(?![A-Za-z])', lambda m:r'\geq', s)
 s=re.sub(r'\\ne(?![A-Za-z])', lambda m:r'\neq', s)
 s=re.sub(r'\\(mathbb|widehat)\s+([A-Za-z])', lambda m:'\\'+m.group(1)+'{'+m.group(2)+'}', s)
 s=re.sub(r'\\mathcal\s+([A-Za-z])',r'\\mathcal{\1}',s)
 return s

def math_asset(s,size=10.6):
 s=clean_math(s)
 key=hashlib.sha256((str(size)+s).encode()).hexdigest()[:20]
 path=CACHE/(key+'.png')
 if not path.exists():
  try:
   mathtext.math_to_image('$'+s+'$',str(path),prop=font_manager.FontProperties(size=size),dpi=300,format='png',color='#111111')
  except Exception as exc:
   raise RuntimeError(f'Math rendering failed for {s!r}: {exc}') from exc
 im=PILImage.open(path)
 w,h=im.size
 return path,w*72/300,h*72/300

def text_inlines(items):
 out=[]
 for x in items:
  t=x['t'];c=x.get('c')
  if t=='Str':out.append(c)
  elif t in ('Space','SoftBreak'):out.append(' ')
  elif t=='LineBreak':out.append(' ')
  elif t in ('Strong','Emph','Strikeout','SmallCaps','Superscript','Subscript'):out.append(text_inlines(c))
  elif t=='Code':out.append(c[1])
  elif t=='Math':out.append(c[1])
  elif t in ('Link','Image'):out.append(text_inlines(c[1]))
  elif t=='Quoted':out.append(text_inlines(c[1]))
 return ''.join(out)

def rich(items,size=10.6):
 out=[]
 for x in items:
  t=x['t'];c=x.get('c')
  if t=='Str':out.append(html.escape(c))
  elif t in ('Space','SoftBreak'):out.append(' ')
  elif t=='LineBreak':out.append('<br/>')
  elif t=='Strong':out.append('<b>'+rich(c,size)+'</b>')
  elif t=='Emph':out.append('<i>'+rich(c,size)+'</i>')
  elif t=='Code':out.append('<font name="Mono" size="'+str(size*.91)+'">'+html.escape(c[1])+'</font>')
  elif t=='Math':
   path,w,h=math_asset(c[1],size)
   if h>size*2.6:
    scale=size*2.6/h;w*=scale;h*=scale
   out.append(f'<img src="{path}" width="{w:.2f}" height="{h:.2f}" valign="middle"/>')
  elif t=='Link':
   target=c[2][0]
   if target.startswith(('https://','http://','#')):
    out.append('<a href="'+html.escape(target,quote=True)+'" color="#1B5367">'+rich(c[1],size)+'</a>')
   else:out.append(rich(c[1],size))
  elif t=='Quoted':out.append('“'+rich(c[1],size)+'”')
  elif t in ('Superscript','Subscript'):
   tag='super' if t=='Superscript' else 'sub';out.append('<'+tag+'>'+rich(c,size*.8)+'</'+tag+'>')
  elif t in ('Span','Cite'):out.append(rich(c[-1],size))
  elif t=='RawInline':out.append(html.escape(c[1]))
  elif t=='Image':out.append(rich(c[1],size))
 result=''.join(out)
 return result

def p(items,key='body'):
 return Paragraph(rich(items,styles[key].fontSize),styles[key])
def display_math(s):
 # Separate complete expressions at explicit quad separators, preserving math syntax.
 chunks=[x.strip() for x in re.split(r'\\qquad|\\quad',s) if x.strip()]
 imgs=[]
 for chunk in chunks:
  path,w,h=math_asset(chunk,12.0)
  scale=min(1, (WIDTH-20)/w)
  if scale<.80:math_warnings.append({'equation':chunk,'scale':scale})
  im=Image(str(path),width=w*scale,height=h*scale)
  im.hAlign='CENTER'
  imgs.extend([im,Spacer(1,6)])
 return [KeepTogether([Spacer(1,4)]+imgs+[Spacer(1,4)])]

def node(ax,x,y,w,h,label,fc='#F1F5F6',fontsize=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',linewidth=1,edgecolor='#9AADB4',facecolor=fc))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fontsize,color='#173747')
def arrow(ax,pts,label=None,lpos=None):
 for a,b in zip(pts[:-2],pts[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color='#326C7C',lw=1.1)
 ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=12,lw=1.1,color='#326C7C'))
 if label and lpos:ax.text(*lpos,label,ha='center',va='center',fontsize=8.3,color='#43565E')
def figures():
 plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
 fig,ax=plt.subplots(figsize=(10.5,5.6));ax.set_xlim(0,10.5);ax.set_ylim(0,5.6);ax.axis('off')
 ax.add_patch(FancyBboxPatch((3.45,.6),3.3,4.4,boxstyle='round,pad=0.08',facecolor='#EEF5F8',edgecolor='#7296A3'))
 ax.text(5.1,4.73,'EXECUTION LAYER',ha='center',fontsize=11,fontweight='bold')
 node(ax,.2,3.35,2.2,1.0,'Frozen PPO walker\n50 Hz actor\nweights + normalization fixed',fc='#F4F4F4',fontsize=9.3)
 node(ax,3.8,3.35,2.6,1.0,'Bounded target correction\nconfidence + fault status',fontsize=9.5)
 node(ax,7.9,3.35,2.35,1.0,'G1 servos and dynamics\n500 Hz physics\nhidden injected parameters',fc='#F4F4F4',fontsize=9.2)
 node(ax,3.8,1.15,2.6,1.2,'Online estimation\nalignment + validity\nproposed module',fontsize=9.5)
 node(ax,.2,1.15,2.2,1.2,'Initial calibration\noptional initialization\nproposed baseline',fontsize=9.3)
 arrow(ax,[(2.4,3.85),(3.8,3.85)],'nominal target',(3.1,4.05))
 arrow(ax,[(6.4,3.85),(7.9,3.85)],'issued target',(7.16,4.05))
 arrow(ax,[(9.1,3.35),(9.1,1.75),(6.4,1.75)],'state, effort, timestamps',(8.0,1.98))
 arrow(ax,[(5.1,2.35),(5.1,3.35)],'estimate + confidence',(5.85,2.86))
 arrow(ax,[(2.4,1.75),(3.8,1.75)],'initial model',(3.1,1.96))
 arrow(ax,[(9.1,4.35),(9.1,5.32),(1.3,5.32),(1.3,4.35)],'actor observations',(7.6,5.46))
 ax.text(5.25,.22,'Known gain feeds only the diagnostic oracle arm. C++ replay is a proposed implementation of the layer.',ha='center',fontsize=8.3,color='#59646B')
 fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
 for ext in ['svg','png']:fig.savefig(ROOT/'figures'/('architecture.'+ext),dpi=220,facecolor='white')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(10.5,4.3));ax.set_xlim(0,10.5);ax.set_ylim(0,4.3);ax.axis('off')
 node(ax,.15,2.65,2.6,1.0,'NOMINAL\nno supported correction',fc='#F4F4F4')
 node(ax,3.95,2.65,2.6,1.0,'ESTIMATING\ncollect and validate evidence')
 node(ax,7.75,2.65,2.6,1.0,'ADAPTING\nfeasible bounded correction')
 node(ax,2.0,.5,2.8,1.0,'LIMITED OR REJECTED\nlow information / model mismatch',fontsize=9.5)
 node(ax,6.15,.5,2.8,1.0,'CONSTRAINT CONFLICT\nlogged decay or override',fontsize=9.5)
 arrow(ax,[(2.75,3.15),(3.95,3.15)],'new evidence',(3.35,3.43))
 arrow(ax,[(6.55,3.15),(7.75,3.15)],'valid and feasible',(7.15,3.43))
 arrow(ax,[(4.7,2.65),(3.6,1.5)],'insufficient evidence',(3.05,2.12))
 arrow(ax,[(9.05,2.65),(7.55,1.5)],'no feasible offset',(9.0,2.02))
 arrow(ax,[(2.0,1.0),(.7,1.0),(.7,2.65)],'withdrawal',(1.1,1.78))
 arrow(ax,[(6.15,1.0),(4.8,1.0)],'specified fallback',(5.48,.73))
 ax.text(5.25,3.98,'PROPOSED STATUS LOGIC',ha='center',fontsize=11,fontweight='bold')
 ax.text(5.25,.13,'Status and fallback describe software behavior; physical recovery is measured separately.',ha='center',fontsize=8.5,color='#59646B')
 fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
 for ext in ['svg','png']:fig.savefig(ROOT/'figures'/('runtime_states.'+ext),dpi=220,facecolor='white')
 plt.close(fig)

class PaperDoc(BaseDocTemplate):
 def __init__(self,path):
  super().__init__(path,pagesize=letter,leftMargin=42,rightMargin=42,topMargin=43,bottomMargin=42,
   title='Fleet Robust Execution - An Actuator Aware Execution Layer for Learned Humanoid Locomotion',
   author='Prasuna Chatla',subject='Technical design, experimental protocol and independent review draft')
  f=Frame(42,42,528,707,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
  self.addPageTemplates(PageTemplate(id='body',frames=[f],onPage=self.chrome))
 def chrome(self,c,doc):
  c.saveState();c.setFillColor(MUTED);c.setFont('Body',8)
  c.drawString(42,768,'FLEET ROBUST EXECUTION  |  Technical design and experimental protocol')
  c.drawRightString(570,768,'Review draft 0.2')
  c.setStrokeColor(LINE);c.setLineWidth(.5);c.line(42,759,570,759)
  c.line(42,32,570,32);c.setFont('Body',7.8)
  c.drawString(42,20,'Prasuna Chatla  |  Evidence cutoff November 15, 2026')
  c.drawRightString(570,20,str(doc.page));c.restoreState()
 def afterFlowable(self,f):
  if hasattr(f,'bookmark'):
   self.canv.bookmarkPage(f.bookmark)
   self.canv.addOutlineEntry(f.heading_text,f.bookmark,level=0,closed=False)
   self.notify('TOCEntry',(0,f.heading_text,self.page,f.bookmark))

def table(block):
 c=block['c'];heads=c[3][1]
 bodyrows=[]
 for tbody in c[4]:bodyrows+=tbody[2]+tbody[3]
 rows=heads+bodyrows+c[5][1]
 cells=[]
 for ri,row in enumerate(rows):
  vals=[]
  for cell in row[1]:
   contents=cell[4]
   paras=[]
   for b in contents:
    if b['t'] in ('Plain','Para'):paras.append(p(b['c'],'tablehead' if ri<len(heads) else 'table'))
    else:paras.extend(convert([b],table_context=True))
   vals.append(paras or [Paragraph('',styles['table'])])
  cells.append(vals)
 n=len(cells[0])
 if n==2:widths=[WIDTH*.37,WIDTH*.63]
 elif n==3:widths=[WIDTH*.25,WIDTH*.405,WIDTH*.345]
 elif n==4:widths=[WIDTH*.18,WIDTH*.285,WIDTH*.27,WIDTH*.265]
 else:widths=[WIDTH/n]*n
 t=Table(cells,colWidths=widths,repeatRows=len(heads),hAlign='LEFT',splitByRow=1)
 cmds=[('GRID',(0,0),(-1,-1),.45,LINE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
 ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
 ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]
 if heads:cmds+=[('BACKGROUND',(0,0),(-1,len(heads)-1),BLUE)]
 for ri in range(len(heads),len(cells)):
  cmds.append(('BACKGROUND',(0,ri),(-1,ri),colors.white if (ri-len(heads))%2==0 else colors.HexColor('#F4F7F8')))
 t.setStyle(TableStyle(cmds))
 return [Spacer(1,4),t,Spacer(1,11)]

toc=TableOfContents()
toc.levelStyles=[ParagraphStyle('TOC',fontName='Body',fontSize=10.2,leading=15.7,leftIndent=0,firstLineIndent=0,spaceBefore=3,rightIndent=20,textColor=INK)]
section_count=0
in_refs=False
def convert(blocks,table_context=False):
 global section_count,in_refs
 flows=[]
 for b in blocks:
  t=b['t'];c=b.get('c')
  if t=='Header':
   level,attr,items=c;title=text_inlines(items)
   if level==1:
    flows.append(p(items,'title'));continue
   if title.startswith('An actuator aware'):
    flows.append(p(items,'subtitle'));continue
   if level==2 and title=='Problem and objectives':
    flows.extend([PageBreak(),Paragraph('Contents',styles['h2']),Spacer(1,8),toc,PageBreak()])
   if level==2:
    section_count+=1
    prefix='' if title=='Abstract' else f'{section_count-1}  '
    para=Paragraph(html.escape(prefix+title),styles['h2'])
    para.bookmark='section-'+str(section_count)
    para.heading_text=prefix+title
    flows.extend([CondPageBreak(85),para])
    in_refs=(title=='References')
   else:flows.extend([CondPageBreak(65),p(items,'h3')])
  elif t in ('Para','Plain'):
   # Standalone math is rendered as a display, not an image inside a sentence.
   if any(x['t']=='Math' and x['c'][0]['t']=='DisplayMath' for x in c):
    pending=[]
    for x in c:
     if x['t']=='Math' and x['c'][0]['t']=='DisplayMath':
      if text_inlines(pending).strip():flows.append(p(pending))
      pending=[]
      flows.extend(display_math(x['c'][1]))
     else:pending.append(x)
    if text_inlines(pending).strip():flows.append(p(pending))
    continue
   if len(c)==1 and c[0]['t']=='Image':
    img=c[0]['c'];src=ROOT/img[2][0]
    if src.suffix=='.svg':src=src.with_suffix('.png')
    pil=PILImage.open(src);w,h=pil.size;dh=WIDTH*h/w
    flows.append(KeepTogether([Image(str(src),width=WIDTH,height=dh),p(img[1],'caption')]))
    continue
   key='ref' if in_refs else ('meta' if text_inlines(c).startswith(('Prasuna Chatla |','Technical design and experimental protocol |')) else 'body')
   flows.append(p(c,key))
  elif t=='Table':flows.extend(table(b))
  elif t=='Figure':flows.extend(convert(c[2],table_context))
  elif t=='BulletList':
   for item in c:
    first=True
    for inner in item:
     if inner['t'] in ('Para','Plain'):
      st=ParagraphStyle('Bullet',parent=styles['body'],leftIndent=12,firstLineIndent=-9,spaceAfter=5)
      flows.append(Paragraph(('•  ' if first else '')+rich(inner['c']),st));first=False
     else:flows.extend(convert([inner],table_context))
   flows.append(Spacer(1,3))
  elif t=='OrderedList':
   start=c[0][0]
   for idx,item in enumerate(c[1],start):
    first=True
    for inner in item:
     if inner['t'] in ('Para','Plain'):
      st=ParagraphStyle('Ordered',parent=styles['body'],leftIndent=17,firstLineIndent=-17,spaceAfter=5)
      flows.append(Paragraph((f'{idx}. ' if first else '')+rich(inner['c']),st));first=False
     else:flows.extend(convert([inner],table_context))
  elif t=='CodeBlock':
   content='\n'.join(textwrap.fill(line,width=93,subsequent_indent='    ',replace_whitespace=False,drop_whitespace=False) if len(line)>93 else line for line in c[1].splitlines())
   flows.append(XPreformatted(html.escape(content),styles['code']))
  elif t=='BlockQuote':flows.extend(convert(c,table_context))
  elif t=='Div':flows.extend(convert(c[1],table_context))
  elif t=='HorizontalRule':flows.append(Spacer(1,10))
  elif t=='RawBlock':pass
  else:raise RuntimeError(f'Unhandled block {t}')
 return flows

def main():
 figures()
 ast=json.loads(subprocess.check_output(['pandoc',str(ROOT/'paper.md'),'-f','markdown+tex_math_dollars','-t','json']))
 story=convert(ast['blocks'])
 pdf=ROOT/'paper.pdf'
 doc=PaperDoc(str(pdf));doc.multiBuild(story)
 reader=PdfReader(pdf)
 stats={'pages':len(reader.pages),'bytes':pdf.stat().st_size,'words':len((ROOT/'paper.md').read_text().split()),'math_width_warnings':math_warnings}
 (CACHE/'build_report.json').write_text(json.dumps(stats,indent=2))
 (CACHE/'extracted.txt').write_text('\n\n'.join(p.extract_text() or '' for p in reader.pages))
 print(json.dumps(stats))
if __name__=='__main__':main()
