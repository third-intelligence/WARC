"""Join released event categories to geometry; ambiguous matches stay unresolved.

Names/locations are identification evidence, not a reviewed semantic pixel mask.
Never turn a bbox-only fallback into an exact object mask or a causal label.
"""
import csv
import importlib.util
import json
from pathlib import Path
import re
import numpy as np
from .mapping import indices,mask_from_indices,raster

from ..paths import DATA
REPO=DATA
PAT=re.compile(r'^\[([^]]+)\]:\[([^]]+)\]:\[([^]]*)\]:(.*)$')
from . import coverage_rules as coverage


class EventBinder:
    def __init__(self,game):
        self.game=game
        # data/event_maps: for ten games, a table that maps the program's logged event signatures to mechanics;
        # the other games use the rules of coverage_rules.prompt_category
        cell=json.loads((REPO/'event_maps/index.json').read_text()).get(game,{})
        self.source={'mapping':cell.get('mapping'),'variant':cell.get('variant')}
        self.rows=None
        if cell.get('mapping'):
            with (REPO/'event_maps'/cell['mapping']).open() as f:self.rows=list(csv.DictReader(f,delimiter='\t'))
            required={f'canonical_{cell["variant"]}',f'status_{cell["variant"]}',f'scope_{cell["variant"]}'}
            if not self.rows or not required<=self.rows[0].keys():raise ValueError('mapping columns missing')

    def category(self,line):
        m=PAT.match(line)
        if not m:return None,'malformed'
        verb,target,details,text=m.groups()
        if self.rows is not None:
            try:
                ev,audit=coverage.common.resolve_gt([line],self.rows,self.source['variant'])
                return (ev[0]['category'],'mapped') if ev else (None,'excluded')
            except ValueError:return None,'unmapped_signature'
        cat=coverage.prompt_category(self.game,verb,target,details,text)
        return cat,'mapped' if cat else 'unmapped_or_out_of_scope'

    def bind(self,lines,before,after,mapped):
        h,w=np.asarray(before['frame']).shape
        ra,rb=raster(before),raster(after)
        geometry={o['id']:o for o in mapped['objects'] if o['kind']=='object'}
        out=[]
        for eid,line in enumerate(lines):
            cat,status=self.category(line)
            row={'id':f'event:{eid}','kind':'event','name':cat or 'unresolved event',
                 'gimmick_id':cat,'event':line,'category_status':status,
                 'mapping_verified':False,'source_rule':self.source,'roi':[],'changed':[],
                 'binding_status':'unresolved','candidates':[]}
            m=PAT.match(line)
            if not m or not cat:out.append(row);continue
            verb,target,details,text=m.groups()
            # Structured identity names are authoritative candidate selectors.
            name=re.search(r'(?:^|;)sname\(([^)]+)\)',details)
            poss=re.findall(r'(?:position\(|_\()(-?\d+)_(-?\d+)\)',details)
            bbox=re.search(r'bbox\((-?\d+)_(-?\d+)\)',details)
            rect=np.zeros((h,w),bool)
            if bbox:
                bw,bh=map(int,bbox.groups())
                if bw>0 and bh>0:
                    for x,y in poss:
                        x,y=int(x),int(y)
                        x0,y0,x1,y1=max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
                        if x1>x0 and y1>y0:rect[y0:y1,x0:x1]=True
            identities=[];candidates=[]
            for uid,o in geometry.items():
                snapshots=[r['objects'][uid]['object'] for r in (ra,rb) if uid in r['objects']]
                identity=any(s.get('name')==name[1] for s in snapshots) if name else any(s.get('name')==target or target in s.get('tags',[]) for s in snapshots)
                if identity:identities.append(uid)
                # Position disambiguates duplicate names; it never invents identity.
                if identity and (not rect.any() or np.any(mask_from_indices(o['amodal_roi'],(h,w))&rect)):
                    candidates.append(uid)
            # A unique named endpoint instance outranks a stale animation
            # bbox. For duplicate names, keep position-based disambiguation.
            if name and len(identities)==1:candidates=identities
            row['candidates']=candidates
            if len(candidates)==1:
                uid=candidates[0];obj=geometry[uid]
                if self.game=='ka59' and cat=='player_center_selection':
                    part=next((o for o in mapped['objects'] if o['id']==uid+'/center'),None)
                    if part is None:
                        row['binding_status']='missing_center_part';out.append(row);continue
                    row.update(roi=part['roi'],changed=part['changed'],parent=uid,binding_status='candidate_part')
                elif cat=='edge_indicator':
                    # Edge strips are not the player body. Keep separate bbox
                    # candidate until their drawing has been reviewed.
                    row.update(roi=indices(rect),changed=indices(rect&(np.asarray(before['frame'])!=np.asarray(after['frame']))),parent=uid,binding_status='bbox_only_not_exact')
                else:
                    row.update(roi=obj['roi'],changed=obj['changed'],parent=uid,binding_status='candidate_whole_object')
            elif len(candidates)>1:
                row['binding_status']='ambiguous_instances'
            elif rect.any():
                row.update(roi=indices(rect),changed=indices(rect&(np.asarray(before['frame'])!=np.asarray(after['frame']))),binding_status='bbox_only_not_exact')
            out.append(row)
        return out
