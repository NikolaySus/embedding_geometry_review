"""In-memory Plotly scenes and the curated interactive camera controls."""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .data import OUT, Q, G, LABEL, xy, short, ternary_grid, project


def measured_scene(mean, triangles, key, height, color, panel):
    quality = key in ["3C", "3D"]
    wire = key in ["2E", "3D"]
    x, y = xy(mean.index)
    coords = (
        mean[["sts", "AG News", "retrieval"]].to_numpy()
        if quality
        else np.column_stack([x, y, mean[height]])
    )
    values = mean[color].to_numpy()
    limit = max(abs(values))
    f = go.Figure()
    f.add_trace(
        go.Mesh3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=values,
            colorscale="RdBu",
            cmin=-limit,
            cmax=limit,
            opacity=0.55,
            flatshading=True,
            visible=not wire,
            showscale=False,
            hoverinfo="skip",
            name="surface",
        )
    )
    edges = sorted(
        {
            tuple(sorted((int(a), int(b))))
            for tri in triangles
            for a, b in zip(tri, np.roll(tri, -1))
        }
    )
    lines = []
    for a, b in edges:
        lines.extend([coords[a].tolist(), coords[b].tolist(), [None, None, None]])
    lines = np.array(lines, dtype=object)
    f.add_trace(
        go.Scatter3d(
            x=lines[:, 0],
            y=lines[:, 1],
            z=lines[:, 2],
            mode="lines",
            line=dict(color="#777", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    f.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=values,
                colorscale="RdBu",
                cmin=-limit,
                cmax=limit,
                colorbar=dict(title="Δ " + LABEL[color]),
            ),
            text=[short(b) for b in mean.index],
            customdata=mean[Q + G].to_numpy(),
            hovertemplate="STS:Cls:Ret %{text}<br>"
            + "<br>".join(
                LABEL[c] + ": %{customdata[" + str(i) + "]:.5f}"
                for i, c in enumerate(Q + G)
            )
            + "<extra></extra>",
            showlegend=False,
        )
    )
    if quality:
        front = np.array(
            [
                not np.any(np.all(coords >= pt, axis=1) & np.any(coords > pt, axis=1))
                for pt in coords
            ]
        )
        f.add_trace(
            go.Scatter3d(
                x=coords[front, 0],
                y=coords[front, 1],
                z=coords[front, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=values[front],
                    colorscale="RdBu",
                    cmin=-limit,
                    cmax=limit,
                    line=dict(color="black", width=3),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if not quality:
        floor = min(0, coords[:, 2].min()) - max(np.ptp(coords[:, 2]) * 0.15, 0.001)
        for a, b in ternary_grid():
            ends = np.array([project(a), project(b)])
            f.add_trace(
                go.Scatter3d(
                    x=ends[:, 0],
                    y=ends[:, 1],
                    z=[floor] * 2,
                    mode="lines",
                    line=dict(color="#aaa", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        annotations = []
        for w, name in [
            (np.array([1, 0, 0]), "STS"),
            (np.array([0, 1, 0]), "Cls"),
            (np.array([0, 0, 1]), "Ret"),
        ]:
            xx, yy = project(w)
            annotations.append(dict(x=xx, y=yy, z=floor, text=name, showarrow=False))
        for axis in range(3):
            for t in [0.2, 0.4, 0.6, 0.8]:
                w = np.zeros(3)
                w[axis] = t
                w[(axis + 1) % 3] = 1 - t
                xx, yy = project(w)
                annotations.append(
                    dict(
                        x=xx,
                        y=yy,
                        z=floor,
                        text=str(int(t * 100)),
                        showarrow=False,
                        font=dict(size=10),
                    )
                )
        scene = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Δ " + LABEL[height], showbackground=False),
            annotations=annotations,
        )
    else:
        scene = dict(
            xaxis=dict(title="Δ STS"),
            yaxis=dict(title="Δ AG News"),
            zaxis=dict(title="Δ Ret"),
        )
    for axis, column in zip(["xaxis", "yaxis", "zaxis"], coords.T):
        low = float(column.min())
        high = float(column.max())
        if not quality and axis == "zaxis":
            low = floor
        pad = max((high - low) * 0.04, 1e-6)
        scene[axis].update(range=[low - pad, high + pad], autorange=False)
    scene.update(
        camera=dict(
            eye=dict(x=1.5, y=-1.8, z=1.2), projection=dict(type="orthographic")
        ),
        aspectmode="manual",
        aspectratio=dict(x=1, y=0.866, z=0.8),
    )
    identity = f"{key}-{panel}"
    f.update_layout(
        title=f"{identity}: {LABEL[height]} / {LABEL[color]}",
        scene=scene,
        width=1000,
        height=800,
        margin=dict(l=30, r=70, t=100, b=45),
        font=dict(family="DejaVu Sans", size=14),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Поверхность",
                        method="restyle",
                        args=[{"visible": True}, [0]],
                    ),
                    dict(
                        label="Каркас", method="restyle", args=[{"visible": False}, [0]]
                    ),
                ],
            )
        ],
    )
    return f


def roof_scene(m, tri, kind, panel, azim, font):
    style = "roof-gradient"
    c = m.columns[panel - 1]
    identity = f"{kind}-{style}-{panel}"
    padded = pd.DataFrame(0.0, index=m.index, columns=Q + G)
    padded.loc[:, m.columns] = m
    f = measured_scene(padded, tri, f"{kind}-{style}", c, c, panel)
    z = m[c].to_numpy()
    limit = np.abs(m.to_numpy()).max() if kind == "5A" else abs(z).max()
    low = min(0, z.min())
    high = max(0, z.max())
    span = max(high - low, 1e-6)
    floor = low - 0.13 * span
    top = high + 0.45 * span
    for t in f.data:
        if t.type == "mesh3d":
            t.update(cmin=-limit, cmax=limit)
        elif t.type == "scatter3d" and t.mode == "markers":
            t.marker.update(cmin=-limit, cmax=limit)
            t.customdata = None
            t.hovertemplate = (
                "STS:Cls:Ret %{text}<br>"
                + ("R " if kind == "5A" else "Δ ")
                + LABEL[c]
                + ": %{z:.6f}<extra></extra>"
            )
        elif t.type == "scatter3d" and t.mode == "lines" and len(t.z) == 2:
            t.z = [floor, floor]
    for a in f.layout.scene.annotations:
        a.z = floor
    f.layout.scene.zaxis.title.text = ("R " if kind == "5A" else "Δ ") + LABEL[c]
    f.layout.title.text = identity
    x, y = xy(m.index)
    if "roof" in style:
        coords = []
        for xx, yy in zip(x, y):
            coords.extend([[xx, yy, floor], [xx, yy, top], [None, None, None]])
        a = np.array(coords, dtype=object)
        f.add_trace(
            go.Scatter3d(
                x=a[:, 0],
                y=a[:, 1],
                z=a[:, 2],
                mode="lines",
                line=dict(color="rgba(90,90,90,.25)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        f.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=[top] * len(z),
                mode="text",
                text=[f"{v:+.1f}" if c == "effective_rank" else f"{v:+.3f}" for v in z],
                textfont=dict(size=font * 1000 / (6.2 * 72)),
                showlegend=False,
                hoverinfo="skip",
                name="roof_values",
            )
        )
    if "gradient" in style:
        for t in f.data:
            if t.type == "scatter3d" and t.mode == "markers":
                t.marker.showscale = False
        gx = 0.5 + 0.9 * np.sin(np.deg2rad(azim))
        gy = 0.29 - 0.9 * np.cos(np.deg2rad(azim))
        zz = np.linspace(low, high, 80)
        f.add_trace(
            go.Scatter3d(
                x=[gx] * 80,
                y=[gy] * 80,
                z=zz,
                mode="lines",
                line=dict(
                    color=zz, colorscale="RdBu", cmin=-limit, cmax=limit, width=9
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        f.layout.scene.zaxis.visible = False
        annotations = list(f.layout.scene.annotations)
        for v in np.linspace(low, high, 4):
            annotations.append(
                dict(
                    x=gx,
                    y=gy,
                    z=float(v),
                    text=f"{v:.3g}",
                    showarrow=False,
                    xanchor="right",
                )
            )
        annotations.append(
            dict(
                x=gx,
                y=gy,
                z=high + 0.2 * span,
                text=("R " if kind == "5A" else "Δ ") + LABEL[c],
                showarrow=False,
            )
        )
        f.layout.scene.annotations = annotations
        f.layout.scene.xaxis.range = [min(-0.08, gx - 0.10), max(1.08, gx + 0.10)]
        f.layout.scene.yaxis.range = [min(-0.08, gy - 0.10), max(0.95, gy + 0.10)]
    else:
        for t in f.data:
            if t.type == "scatter3d" and t.mode == "markers":
                t.marker.colorbar.title.text = ""
    f.layout.scene.zaxis.range = [
        floor - 0.03 * span,
        (top if "roof" in style else high) + 0.15 * span,
    ]
    f.layout.scene.camera.eye = dict(
        x=2 * np.cos(np.deg2rad(azim)),
        y=2 * np.sin(np.deg2rad(azim)),
        z=2 * np.tan(np.deg2rad(55 if "roof" in style else 30)),
    )
    return f


def apply_roof(f, ratio):
    meta = f.layout.meta
    top = meta["high"] + ratio * meta["span"]
    roof = next(t for t in f.data if t.name == "roof_values")
    roof.z = [top] * 22
    roof.textposition = "top center"
    lines = f.data[meta["roof_line_index"]]
    lines.z = [
        top if i % 3 == 1 else meta["floor"] if i % 3 == 0 else None for i in range(66)
    ]
    f.layout.scene.zaxis.range = [
        meta["floor"] - 0.03 * meta["span"],
        top + 0.2 * meta["span"],
    ]


def large_scene(m, tri, kind, panel, azim, font):
    f = roof_scene(m, tri, kind, panel, azim, font)
    identity = f"{kind}-large-{panel}"
    z = np.asarray(f.data[0].z, dtype=float)
    low = min(0, z.min())
    high = max(0, z.max())
    span = max(high - low, 1e-6)
    floor = low - 0.13 * span
    roofi = next(i for i, t in enumerate(f.data) if t.name == "roof_values")
    linei = roofi - 1
    f.layout.meta = dict(
        high=float(high),
        span=float(span),
        floor=float(floor),
        roof_line_index=linei,
        roof_text_index=roofi,
    )
    f.layout.title.text = identity
    f.layout.scene.camera.eye = dict(
        x=1.65 * np.cos(np.deg2rad(azim)),
        y=1.65 * np.sin(np.deg2rad(azim)),
        z=1.65 * np.tan(np.deg2rad(55)),
    )
    gx = 0.5 + 0.9 * np.sin(np.deg2rad(azim))
    gy = 0.29 - 0.9 * np.cos(np.deg2rad(azim))
    dx = 0.065 * np.sin(np.deg2rad(azim))
    dy = -0.065 * np.cos(np.deg2rad(azim))
    annotations = list(f.layout.scene.annotations)
    for a, v in zip(annotations[-5:-1], np.linspace(low, high, 4)):
        a.x = gx + 1.8 * dx
        a.y = gy + 1.8 * dy
        a.z = v
        f.add_trace(
            go.Scatter3d(
                x=[gx, gx + dx],
                y=[gy, gy + dy],
                z=[v, v],
                mode="lines",
                line=dict(color="#333", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    annotations[-1].z = high + 0.32 * span
    f.layout.scene.annotations = annotations
    apply_roof(f, 0.45)
    return f


def snap_camera(camera):
    eye = camera.get("eye", {})
    x = float(eye.get("x", 1))
    y = float(eye.get("y", 1))
    z = float(eye.get("z", 1))
    angle = int(np.floor((np.degrees(np.arctan2(y, x)) % 360) / 60 + 0.5)) % 6 * 60
    radius = max(np.hypot(x, y), 1e-4)
    return dict(
        eye=dict(
            x=float(radius * np.cos(np.deg2rad(angle))),
            y=float(radius * np.sin(np.deg2rad(angle))),
            z=z,
        ),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
        projection=dict(type="orthographic"),
    ), angle


def apply_font(f, points):
    f.data[f.layout.meta["roof_text_index"]].textfont.size = points * 1000 / (6.2 * 72)


def compact_scene(m, tri, kind, panel, azim, font):
    f = large_scene(m, tri, kind, panel, azim, font)
    identity = f"{kind}-compact-{panel}"
    f.layout.title.text = identity
    f.layout.scene.camera, angle = snap_camera(f.layout.scene.camera.to_plotly_json())
    sx = f.layout.scene.xaxis.range
    sy = f.layout.scene.yaxis.range
    f.layout.scene.aspectratio = dict(x=sx[1] - sx[0], y=sy[1] - sy[0], z=1.2)
    f.layout.scene.aspectmode = "manual"
    f.layout.scene.dragmode = "turntable"
    default_font = 6.5
    apply_font(f, default_font)
    return f


def interactive(m, tri, kind, panel, azim, font):
    f = compact_scene(m, tri, kind, panel, azim, font)
    identity = f"{kind}-compact-{panel}"
    _, angle = snap_camera(f.layout.scene.camera.to_plotly_json())
    default_font = 6.5
    f.write_json(OUT / "interactive" / f"{identity}.figure.json")
    script = """
const gd=document.getElementById('{plot_id}'),meta=gd.layout.meta;
const bar=document.createElement('div');bar.style.cssText='padding:10px;font:14px sans-serif;display:flex;gap:15px;align-items:center;flex-wrap:wrap';
bar.innerHTML='<label>Высота слоя <input id="roof" type="range" min=".15" max="1.5" step=".05" value=".45"><output id="height">0.45</output></label><label>Размер чисел (пт) <input id="font" type="number" min="5" max="12" step=".5" value="6.5"></label><label>Азимут <select id="az">'+[0,60,120,180,240,300].map(v=>'<option value="'+v+'">'+v+'°</option>').join('')+'</select></label><button id="save">Скачать ракурс JSON</button>';
document.body.prepend(bar);const roof=bar.querySelector('#roof'),font=bar.querySelector('#font'),az=bar.querySelector('#az');az.value=ANGLE;
let adjusting=false;
function aligned(camera,forced){const eye=camera.eye,r=Math.max(Math.hypot(eye.x,eye.y),.0001);let a=forced===undefined?((Math.round((Math.atan2(eye.y,eye.x)*180/Math.PI+360)%360/60)%6)*60):forced;return {camera:{eye:{x:r*Math.cos(a*Math.PI/180),y:r*Math.sin(a*Math.PI/180),z:eye.z},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1},projection:{type:'orthographic'}},angle:a};}
async function setAngle(forced){if(adjusting)return;const current=gd._fullLayout.scene.camera,v=aligned(current,forced);az.value=v.angle;
if(Math.abs(current.eye.x-v.camera.eye.x)<1e-8&&Math.abs(current.eye.y-v.camera.eye.y)<1e-8&&Math.abs(current.center.x)+Math.abs(current.center.y)+Math.abs(current.center.z)<1e-8&&Math.abs(current.up.x)+Math.abs(current.up.y)+Math.abs(current.up.z-1)<1e-8&&current.projection.type==='orthographic')return;
adjusting=true;try{await Plotly.relayout(gd,{'scene.camera':v.camera});}finally{adjusting=false;}}
gd.on('plotly_relayout',ev=>{if(Object.keys(ev).some(k=>k.startsWith('scene.camera')))setAngle();});
az.onchange=()=>setAngle(Number(az.value));
roof.oninput=()=>{const r=Number(roof.value),top=meta.high+r*meta.span;bar.querySelector('#height').textContent=r.toFixed(2);Plotly.restyle(gd,{z:[Array.from({length:66},(_,i)=>i%3===1?top:i%3===0?meta.floor:null),Array(22).fill(top)]},[meta.roof_line_index,meta.roof_text_index]);Plotly.relayout(gd,{'scene.zaxis.range':[meta.floor-.03*meta.span,top+.2*meta.span]});};
font.oninput=()=>{const value=Number(font.value);if(Number.isFinite(value)&&value>=5&&value<=12)Plotly.restyle(gd,{'textfont.size':value*1000/(6.2*72)},[meta.roof_text_index]);};
bar.querySelector('#save').onclick=async()=>{await setAngle();const s=gd._fullLayout.scene;const state={variant:IDENTITY,roof_ratio:Number(roof.value),label_font:Number(font.value),azimuth:Number(az.value),camera:s.camera,aspectratio:s.aspectratio,xaxis:gd.layout.scene.xaxis,yaxis:gd.layout.scene.yaxis,zaxis:gd.layout.scene.zaxis,surface_visible:gd.data[0].visible!==false};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{type:'application/json'}));a.download=IDENTITY+'.camera.json';a.click();URL.revokeObjectURL(a.href);};
""".replace("IDENTITY", json.dumps(identity)).replace("ANGLE", str(angle))
    f.write_html(
        OUT / "interactive" / f"{identity}.html",
        include_plotlyjs=True,
        post_script=script,
    )
    (OUT / "interactive" / f"{identity}.camera.json").write_text(
        json.dumps(
            dict(
                variant=identity,
                roof_ratio=0.45,
                label_font=default_font,
                azimuth=angle,
                camera=f.layout.scene.camera.to_plotly_json(),
            ),
            indent=2,
        )
    )
