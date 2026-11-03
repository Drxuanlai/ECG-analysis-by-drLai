import numpy as np
from PIL import Image
from core.lead_layout import detect_layout, resolve_layout
from core.waveform_roi import refine_waveform_rois
from core.lead_isolation import isolate_leads, isolation_overlay


def _fixture():
    a=np.full((700,1400,3),235,np.uint8)
    # Black strokes: use multiple widths to emulate printed ink.
    for y in (135,275,415):
        a[y-2:y+3,45:1350]=18
    im=Image.fromarray(a)
    rois=refine_waveform_rois(im,resolve_layout(im,detect_layout(im,ocr=False)))
    return im,rois


def test_twelve_unique_masks_and_nonoverlap():
    im,rois=_fixture()
    r=isolate_leads(im,rois)
    assert len(r['lead_masks'])==12
    count=np.zeros(im.size[::-1],dtype=np.uint8)
    for mask in r['lead_masks'].values():count+=(mask>0).astype(np.uint8)
    assert count.max()<=1
    assert r['ambiguous_mask'].shape==count.shape


def test_geometry_preserved_and_review_status():
    im,rois=_fixture()
    prior=[(r['lead'],r['layout_box'][:],r['waveform_box'][:]) for r in rois]
    r=isolate_leads(im,rois)
    assert prior==[(q['lead'],q['layout_box'],q['waveform_box']) for q in rois]
    assert all(q['status']=='REVIEW' for q in r['diagnostics'].values())
    assert isolation_overlay(im,r,rois).size==im.size


def test_invalid_margin_rejected():
    import pytest
    im,rois=_fixture()
    with pytest.raises(ValueError):isolate_leads(im,rois,ambiguous_margin_ratio=.5)
