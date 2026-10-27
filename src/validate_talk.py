"""Validate generated deck, script and their shared scientific provenance."""
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from docx import Document
from pypdf import PdfReader
from publication import conference, unified_assets as shared

ROOT=shared.ROOT
OUT=ROOT/'build/presentation'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}


def main():
    for path,digest in json.loads((OUT/'provenance.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    pdf=PdfReader(OUT/'cross_task_interference_talk.pdf')
    assert len(pdf.pages)==6
    with ZipFile(OUT/'cross_task_interference_talk.pptx') as z:
        slides=sorted(n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n))
        assert len(slides)==6
        size=ET.fromstring(z.read('ppt/presentation.xml')).find('p:sldSz',NS)
        assert size.attrib=={'cx':'12192000','cy':'6858000'} or (size.get('cx'),size.get('cy'))==('12192000','6858000')
        texts=[]
        for i,name in enumerate(slides):
            page=ET.fromstring(z.read(name))
            texts.append(' '.join(t.text or '' for t in page.findall('.//a:t',NS)))
            if i:
                assert page.findall('.//p:pic',NS), f'Missing figure on slide {i+1}'
        joined=' '.join(texts)
        assert conference.CONFIG['author_ru'] in joined
        assert conference.CONFIG['email'] in joined
        assert not any(x in joined for x in ['Название презентации','Область расположения','ЗАГОЛОВОК СЛАЙДА','{{'])
    article=json.loads((ROOT/'build/ultracompact/data/figure_manifest.json').read_text())
    slides=json.loads((OUT/'figure_manifest.json').read_text(encoding='utf-8'))
    assert len(slides)==len(article)==5
    for s,a in zip(slides,article):
        assert s['labels']==a['texts'],f'Figure {s["figure"]}: numeric labels changed'
        if s['figure'] in (2,3,5):
            assert s['edge_percentages']==a['edge_percentages']
            assert s['cameras']==a['cameras']
            assert all(c['elevation']==conference.CONFIG['surface_elevation'] for c in s['cameras'])
            assert all(c['azimuth']==conference.CONFIG['surface_azimuth'] for c in s['cameras'])
    script=Document(OUT/'speaker_script.docx')
    words=' '.join(p.text for p in script.paragraphs)
    assert words.count('Слайд ')==6
    assert '{{' not in words
    assert 1100 <= len(words.split()) <= 1600
    state={'slides':6,'figures':5,'script_words':len(words.split()),'numeric_labels_match_article':True}
    (OUT/'validation.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
