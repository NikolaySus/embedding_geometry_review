"""Six vector PDF pages comparing the three conference surfaces at fixed elevation."""
import io
import json
from copy import copy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from pypdf import PdfReader, PdfWriter, Transformation

from gallery.bundles import load_bundle
from gallery.data import G, Q, project
from gallery.surfaces import compact_surface
from publication import conference, unified_assets as shared, ultracompact

OUT = shared.ROOT / 'build/camera-comparison'
A4 = (595.276, 841.89)


def vertical_edge(ax):
    names = ['STS', 'Cls', 'Ret']
    xyz = np.array([[*project(w), 0] for w in np.eye(3)])
    x, y, _ = proj3d.proj_transform(*xyz.T, ax.get_proj())
    pairs = [(0, 1), (1, 2), (0, 2)]
    i, j = min(pairs, key=lambda ij: abs(x[ij[0]]-x[ij[1]]))
    assert abs(x[i]-x[j]) < 1e-10
    assert abs(y[i]-y[j]) > 1e-5
    third = next(k for k in range(3) if k not in (i, j))
    return f'{names[i]}–{names[j]}', 'справа' if x[i] > x[third] else 'слева'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(shared.SETTINGS['experiment'], shared.SETTINGS['model'])
    delta = bundle['deltas']
    mean = delta[delta.branch != 'control'].groupby('branch')[Q+G].mean()
    figures = [('2D', 2, 'Геометрия пространства', mean[G]),
               ('3E', 3, 'Качество на независимых задачах', mean[Q[1:]]),
               ('5A', 5, 'Остатки смешивания', bundle['residual'])]
    writer, manifest, reference_labels = PdfWriter(), [], {}
    # Three opposite-view pairs; first page reproduces the current azimuth.
    for number, angle in enumerate([-60, 120, 0, 180, 60, 240], 1):
        rows, edge = [], None
        for kind, figure_number, title, values in figures:
            with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':8}):
                fig, labels = compact_surface(values, bundle['triangles'], kind, dense=True, azim=angle)
                before = [[t.get_text() for t in ax.texts] for ax in fig.axes]
                ultracompact.surface_row(fig, outside_labels=False)
                # Equal zoom across all views leaves room for outward-facing labels.
                for ax in fig.axes:
                    ax.set_box_aspect(ax._box_aspect, zoom=1.05)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                def label_box(ax,label):
                    x,y,_ = proj3d.proj_transform(*label.get_position_3d(),ax.get_proj())
                    projected = copy(label)
                    projected.set_position((x,y))
                    return projected.get_window_extent(renderer)
                for panel, ax in enumerate(fig.axes):
                    boxes = [label_box(ax,label) for label in ax.texts]
                    left, right = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
                    target = (panel+.5)*fig.bbox.width/4
                    pos = ax.get_position()
                    ax.set_position([pos.x0+(target-(left+right)/2)/fig.bbox.width,pos.y0,pos.width,pos.height])
                fig.canvas.draw()
                for ax in fig.axes:
                    for label in ax.texts:
                        bounds = label_box(ax,label)
                        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width, (number,label.get_text())
                        assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.height, (number,label.get_text())
                edges = [vertical_edge(ax) for ax in fig.axes]
                assert len(set(edges)) == 1
                if edge is None:
                    edge = edges[0]
                assert edges[0] == edge
                assert len(labels) == 88
                assert before == [[t.get_text() for t in ax.texts] for ax in fig.axes]
                reference_labels.setdefault(kind, before)
                assert before == reference_labels[kind]
                path = OUT / f'view-{number}-figure-{figure_number}.pdf'
                fig.savefig(path)
                plt.close(fig)
                rows.append((figure_number, title, path))
        buffer = io.BytesIO()
        header = plt.figure(figsize=(A4[0]/72,A4[1]/72))
        def text(x,y,value,size=10,bold=False,align='left'):
            header.text(x/A4[0],y/A4[1],value,fontsize=size,fontfamily='Times New Roman',
                        fontweight='bold' if bold else 'normal',ha=align)
        text(22,805,f'Вариант {number} из 6 | Азимут {angle}°',17,True)
        text(22,782,f'Вертикальная сторона: {edge[0]}, {edge[1]}. Высота камеры: {conference.CONFIG["surface_elevation"]}°.',12)
        text(22,764,'Одинаковые данные, шкалы и числовые метки; меняется только направление обзора.')
        for (figure_number, title, _), top in zip(rows, [728, 503, 278]):
            text(22,top,f'Рисунок {figure_number} – {title}',12,True)
        text(22,35,'Верхние числа – значения 22 измеренных точек. Грани соединяют измерения; SD не показано.',9)
        text(A4[0]-22,19,str(number),9,align='right')
        header.savefig(buffer,format='pdf')
        plt.close(header)
        page = PdfReader(buffer).pages[0]
        for (_, _, path), bottom in zip(rows, [551, 326, 101]):
            row = PdfReader(path).pages[0]
            scale = (A4[0]-44)/float(row.mediabox.width)
            page.merge_transformed_page(row, Transformation().scale(scale).translate(22,bottom))
        writer.add_page(page)
        manifest.append({'variant':number, 'azimuth':angle, 'elevation':conference.CONFIG['surface_elevation'],
                         'vertical_edge':edge[0], 'side':edge[1], 'figures':[2,3,5], 'roof_labels_per_figure':88})
    assert len({(v['vertical_edge'],v['side']) for v in manifest}) == 6
    result = OUT / 'surface_camera_comparison.pdf'
    writer.write(result)
    assert len(PdfReader(result).pages) == 6
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
