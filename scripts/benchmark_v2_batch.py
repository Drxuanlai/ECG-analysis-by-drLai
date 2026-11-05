"""Batch QA of legacy validation ZIPs, intentionally never claims clinical accuracy."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from core.benchmark_v2 import import_validation_zip,evaluate
from core.qrs_detector_v2 import detect_qrs_candidates

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',help='Directory of validation ZIP archives')
    parser.add_argument('--output',default='benchmark_v2_summary.csv')
    a=parser.parse_args()
    out=[]
    for p in sorted(Path(a.folder).glob('*.zip')):
        try:
            data=import_validation_zip(p.read_bytes())
            r=data['record'];prediction=detect_qrs_candidates(data['roi_image'])
            report=evaluate(r,prediction['candidates'])
            out.append({'filename':p.name,'case_id':r['case_id'],'lead':r['lead'],
                        'reference_count':report['reference_count'],
                        'candidate_count':report['prediction_count'],
                        'matched_x_only_for_review':report['matched_count'],
                        'scoring_status':report['status'],
                        'sensitivity':report['sensitivity'],'precision':report['precision'],
                        'error':''})
        except Exception as e:
            out.append({'filename':p.name,'error':f'{type(e).__name__}: {e}'})
    fields=['filename','case_id','lead','reference_count','candidate_count','matched_x_only_for_review','scoring_status','sensitivity','precision','error']
    with open(a.output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    print(f'Wrote {len(out)} records to {a.output}')

if __name__=='__main__':main()
