import numpy as np
from PIL import Image, ImageDraw
from core.row_boundary import propose_row_refinement, mask_qc, ink_evidence, qc_overlay
from core.multi_layout import propose_boxes
from core.signal_artifact import separate


def scene():
    im=Image.new('RGB',(800,600),'white'); d=ImageDraw.Draw(im)
    for y in [170,280,390,500]:d.line((30,y,750,y),fill=(20,20,20),width=3)
    return im


def test_rows_preserve_count_and_column_boundaries():
    im=scene();p=propose_boxes(im,'6x2',has_rhythm=False)
    result=propose_row_refinement(im,p)
    assert len(result['row_boundaries_suggested'])==7
    assert result['row_boundaries_suggested'][0]==p['row_boundaries'][0]
    assert result['row_boundaries_suggested'][-1]==p['row_boundaries'][-1]
    assert result['status']=='REVIEW'
    assert all(a<b for a,b in zip(result['row_boundaries_suggested'],result['row_boundaries_suggested'][1:]))


def test_never_moves_more_than_limit():
    im=scene();p=propose_boxes(im,'3x4',has_rhythm=True)
    r=propose_row_refinement(im,p,max_shift_fraction=.028)
    for a,b in zip(p['row_boundaries'],r['row_boundaries_suggested']):assert abs(a-b)<=round(im.height*.028)+1


def test_blank_is_review():
    im=Image.new('RGB',(500,110),'white');e=separate(im);q=mask_qc(im,e,lead='I')
    assert q['status']=='REVIEW'
    assert 'insufficient_ink_candidate' in q['reasons']
    assert qc_overlay(im,e,q).size==im.size


def test_colored_grid_without_waveform_not_pass():
    im=Image.new('RGB',(500,180),'white'); d=ImageDraw.Draw(im)
    for x in range(0,500,14):d.line((x,0,x,180),fill=(210,160,168))
    for y in range(0,180,14):d.line((0,y,500,y),fill=(210,160,168))
    ev=separate(im);q=mask_qc(im,ev,lead='V2')
    assert q['status']=='REVIEW'
    assert set(q['metrics'])>= {'uncertain_fraction','edge_candidate_fraction','straight_candidate_fraction'}


def test_repeatability():
    im=scene();p=propose_boxes(im,'6x2',has_rhythm=False)
    assert propose_row_refinement(im,p)==propose_row_refinement(im,p)


def test_ink_evidence_shapes():
    im=scene();a,b=ink_evidence(im)
    assert a.shape==b.shape==(600,800)
    assert a.dtype==np.bool_
