from pathlib import Path
import re

import numpy as np
import pandas as pd
from docx import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]


def main():
    text = (ROOT/'manuscript/compact.md').read_text()
    figures = re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
    assert len(figures) == 4
    for path in figures:
        assert (ROOT/'manuscript'/path).is_file(), path
    assert not re.search(r'Raw MiniLM|raw_minilm|нескольких компактных моделей',text)
    data=ROOT/'build/compact/data'
    trajectory=pd.read_csv(data/'geometry_trajectories.csv')
    geometry=pd.read_csv(data/'geometry_quality.csv')
    assert set(trajectory.model)==set(geometry.model)=={'all_minilm'}
    assert len(trajectory)==48 and len(geometry)==12
    assert np.allclose(trajectory[trajectory.fraction==0].effective_rank,203.610870,atol=.001)
    q=pd.read_csv(data/'quality_geometry_by_seed.csv')
    assert len(q)==69 and set(q.seed)=={42,43,44}
    pdf=PdfReader(ROOT/'build/compact/cross_task_interference_compact.pdf')
    doc=Document(ROOT/'build/compact/cross_task_interference_compact.docx')
    assert len(doc.inline_shapes)==4
    assert len(list((ROOT/'build/compact/pages').glob('page-*.png')))==len(pdf.pages)
    assert 'Raw MiniLM' not in '\n'.join(p.extract_text() for p in pdf.pages)
    words=len(re.findall(r'\b[\w-]+\b',text))
    print(f'Validated: {words} words, {len(pdf.pages)} pages, 4 figures, {len(doc.tables)} tables')


if __name__=='__main__':
    main()
