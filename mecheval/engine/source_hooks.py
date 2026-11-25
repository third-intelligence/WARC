"""Narrow, fail-closed runtime instrumentation of the pinned GT source.

Only private game instances are patched; source files/classes are untouched.
The game's own arithmetic/branches execute. Added observers copy coordinates,
never recompute gameplay. Structural guards fail if upstream code changes.
"""
import ast
import copy
import inspect
import textwrap
import types
import numpy as np


def instrument(bound, transformer, observer):
    original=bound.__func__
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=transformer.visit(tree)
    if not transformer.hits: raise ValueError('source hook no longer matches: '+original.__name__)
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__,_pixel_observer=observer)
    exec(compile(tree,inspect.getsourcefile(original)+':pixel-observer','exec'),namespace)
    return types.MethodType(namespace[original.__name__],bound.__self__)


class ReflectionObserver(ast.NodeTransformer):
    def __init__(self): self.hits=0
    def visit_If(self,node):
        node=self.generic_visit(node)
        if ast.unparse(node.test)=='mqrjslqocj[dhxrbswmvp, vqnsvotjkg] == -1':
            self.hits+=1
            call=ast.parse('_pixel_observer(sprite, vqnsvotjkg, dhxrbswmvp)').body[0]
            return [call,node]
        return node


class PaintObserver(ast.NodeTransformer):
    def __init__(self):self.hits=0
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if len(node.targets)==1:
            target=node.targets[0]
            if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='pixels':
                self.hits+=1
                # Reuse the actual evaluated slice on a boolean array with the
                # same shape. Observer runs inside the actual branch, including
                # same-color writes that deliberately emit no GT log.
                call=ast.Expr(ast.Call(func=ast.Name('_pixel_observer',ast.Load()),args=[
                    ast.Name('canvas',ast.Load()),ast.Attribute(ast.Name('pixels',ast.Load()),'shape',ast.Load()),
                    slice_expression(copy.deepcopy(target.slice))],keywords=[]))
                return [node,call]
        return node


def slice_expression(node):
    if isinstance(node,ast.Slice):
        return ast.Call(ast.Name('slice',ast.Load()),[node.lower or ast.Constant(None),node.upper or ast.Constant(None),node.step or ast.Constant(None)],[])
    if isinstance(node,ast.Tuple):return ast.Tuple([slice_expression(x) for x in node.elts],ast.Load())
    return node


class SourceHooks:
    def __init__(self,game_id,game):
        self.game_id=game_id;self.game=game;self.reflections={};self.paint=[];self.originals={}
        if game_id=='ar25':
            original=game._compute_all_reflection_maps
            def observe(sprite,x,y):
                key=id(sprite)
                self.reflections.setdefault(key,{'sprite':sprite,'cells':set()})['cells'].add((int(x),int(y)))
            transformer=ReflectionObserver();traced=instrument(original,transformer,observe)
            if transformer.hits!=1:raise ValueError('unexpected ar25 reflection hook count')
            def call():
                self.reflections={}
                return traced()
            self.originals['_compute_all_reflection_maps']=original
            game._compute_all_reflection_maps=call
        if game_id=='cd82':
            for name in ('_paint_canvas','_paint_canvas_crane'):
                original=getattr(game,name)
                def observe(canvas,shape,index,source=name):
                    mask=np.zeros(shape,bool);mask[index]=True
                    self.paint.append({'sprite':canvas,'mask':mask,'x':canvas.x,'y':canvas.y,
                                       'rotation':canvas.rotation,'scale':canvas.scale,'source':source})
                setattr(game,name,instrument(original,PaintObserver(),observe))
                self.originals[name]=original

    def begin_action(self):self.paint=[]

    def close(self):
        for name,method in self.originals.items():setattr(self.game,name,method)
