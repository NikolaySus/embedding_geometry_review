from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/compact'
DATA = OUT / 'data'
FIGURES = OUT / 'figures'
SOURCE_DATA = ROOT / 'assets/full/data'
MIX = ROOT / 'reports/pilot_v2_mixtures'
COLORS = {'control': '#687782', 'sts': '#2878B5', 'classification': '#C54E52', 'retrieval': '#2A9D6F'}
LABELS = {'control': 'Контроль', 'sts': 'STS', 'classification': 'Классификация', 'retrieval': 'Поиск'}
SELECTED = ['control', 'mix_s100_c000_r000', 'mix_s000_c100_r000', 'mix_s000_c000_r100', 'mix_s020_c000_r080', 'mix_s040_c000_r060', 'mix_s033_c033_r033']


def label(branch):
    if branch == 'control':
        return 'Контроль'
    if branch == 'mix_s033_c033_r033':
        return '1:1:1'
    return ':'.join(str(int(x[1:])) for x in branch.removeprefix('mix_').split('_'))


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.png', dpi=260, bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def design():
    fig, ax = plt.subplots(figsize=(9, 2.9))
    ax.set(xlim=(0, 9), ylim=(0, 3))
    ax.axis('off')
    boxes = [
        (.05, 1.05, 1.65, .8, 'All-MiniLM\nисходная M0'),
        (2.1, 1.7, 3.45, 1.05, 'Отдельные цели\nWikiText / SimCSE · STS / CoSENT\nAG News / SupCon · MS MARCO / ranking'),
        (2.1, .35, 3.45, 1.05, 'Смешивание STS:Cls:Ret\n21 точка (шаг 0,20) + центр\n+ SimCSE-контроль'),
        (6.0, .85, 2.85, 1.25, 'Качество + геометрия\nSTS · классификация\nкластеризация · поиск\nспектр · соседства · зазоры'),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor='#EFF3F5', edgecolor='#596674'))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=9)
    for y in (2.2, .9):
        ax.annotate('', xy=(2.05, y), xytext=(1.75, 1.45), arrowprops={'arrowstyle': '->'})
        ax.annotate('', xy=(5.95, 1.45), xytext=(5.6, y), arrowprops={'arrowstyle': '->'})
    ax.text(4.5, .05, 'Каждая ветвь: 270 шагов · пакет 64 · seed 42 / 43 / 44 · общая M0', ha='center', fontsize=9)
    (DATA/'design.json').write_text(json.dumps({'model': 'sentence-transformers/all-MiniLM-L6-v2', 'steps':270, 'batch_size':64, 'seeds':[42,43,44], 'simplex_points':22}, indent=2))
    save(fig, 'design')


def matrix():
    family = pd.read_csv(MIX/'family_deltas.csv')
    scores = pd.read_csv(MIX/'score_deltas.csv')
    geometry = pd.read_csv(MIX/'geometry.csv')
    quality = family.pivot(index=['branch','seed'], columns='family', values='delta')
    ag = scores[scores.task == 'AGNewsClassification'].set_index(['branch','seed']).delta_vs_m0
    quality['AG News'] = ag
    qcols = ['AG News', 'sts', 'classification_transfer', 'clustering', 'retrieval']
    gcols = ['effective_rank', 'top10_eigenvalue_share', 'neighborhood_preservation', 'retrieval_margin']
    baseline = geometry[geometry.branch == 'm0'].set_index('seed')
    g = geometry[geometry.branch != 'm0'].copy()
    for c in gcols:
        g[c] = g[c] - g.seed.map(baseline[c])
    combined = quality[qcols].join(g.set_index(['branch','seed'])[gcols])
    combined.to_csv(DATA/'quality_geometry_by_seed.csv')
    stats = combined.groupby('branch').agg(['mean','std'])
    stats.to_csv(DATA/'quality_geometry_summary.csv')
    means = stats.xs('mean', level=1, axis=1).loc[SELECTED]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={'width_ratios':[5,4]}, layout='constrained')
    blocks = [(qcols, ['AG News','STS','Перенос\nклассиф.','Кластеры','Поиск'], 'а) Прикладные дельты'),
              (gcols, ['Ранг','Доля\nтоп-10','Соседства','Поиск.\nзазор'], 'б) Геометрические дельты')]
    for j, (cols, names, title) in enumerate(blocks):
        raw = means[cols].to_numpy()
        scale = np.max(np.abs(raw)) if j == 0 else np.max(np.abs(raw), axis=0)
        im = axes[j].imshow(raw/scale, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        axes[j].set_xticks(range(len(cols)), names)
        axes[j].set_yticks(range(len(SELECTED)), [label(b) for b in SELECTED] if j == 0 else ['']*len(SELECTED))
        axes[j].set_title(title)
        for row in range(len(raw)):
            for col in range(len(cols)):
                norm = raw[row,col]/(scale if j == 0 else scale[col])
                axes[j].text(col, row, f'{raw[row,col]:+.1f}' if j == 1 and col == 0 else f'{raw[row,col]:+.3f}', ha='center',va='center',fontsize=10,color='white' if abs(norm)>.55 else '#111111')
    axes[0].set_ylabel('STS:Cls:Ret')
    save(fig, 'quality_geometry')


def diagnostics():
    trajectories = pd.read_csv(SOURCE_DATA/'geometry_trajectories.csv')
    trajectories = trajectories[trajectories.model == 'all_minilm'].copy()
    merged = pd.read_csv(SOURCE_DATA/'geometry_and_quality_deltas.csv')
    assert set(merged.model) == {'all_minilm'}
    trajectories.to_csv(DATA/'geometry_trajectories.csv',index=False)
    merged.to_csv(DATA/'geometry_quality.csv',index=False)
    fig, axes = plt.subplots(2,2,figsize=(8.5,5.5),layout='constrained')
    for ax, metric, title in [(axes[0,0],'effective_rank','а) Эффективный ранг'),(axes[0,1],'neighborhood_preservation','б) Сохранение соседств')]:
        for branch, color in COLORS.items():
            stats = trajectories[trajectories.branch == branch].groupby('fraction')[metric].agg(['mean','std'])
            ax.plot(stats.index,stats['mean'],color=color,label=LABELS[branch])
            ax.fill_between(stats.index,stats['mean']-stats['std'],stats['mean']+stats['std'],color=color,alpha=.15)
        ax.set(title=title,xlabel='Доля шагов обучения')
    rows=[]
    for ax,x,y,title,xlabel in [(axes[1,0],'delta_effective_rank','STS','в) Геометрия и STS','Δ эффективного ранга'),(axes[1,1],'delta_retrieval_margin','Поиск','г) Геометрия и поиск','Δ поискового зазора')]:
        for b,color in COLORS.items():
            d=merged[merged.branch==b]
            ax.scatter(d[x],d[y],c=color,s=24)
        r=merged[x].corr(merged[y])
        rows.append({'x':x,'y':y,'r':r,'n':len(merged)})
        ax.text(.04,.94,f'r={r:.3f}; n={len(merged)}',transform=ax.transAxes,va='top')
        ax.set(title=title,xlabel=xlabel,ylabel=f'Δ {y}')
    for ax in axes.flat:
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8,ncol=2)
    pd.DataFrame(rows).to_csv(DATA/'correlations.csv',index=False)
    save(fig,'diagnostics')


def pareto():
    f=pd.read_csv(MIX/'family_deltas.csv')
    s=pd.read_csv(MIX/'score_deltas.csv')
    d=f.pivot(index=['branch','seed'],columns='family',values='delta')
    d['ag']=s[s.task=='AGNewsClassification'].set_index(['branch','seed']).delta_vs_m0
    d=d.drop(index='control',level='branch')
    stats=d.groupby('branch').agg(['mean','std'])
    stats.to_csv(DATA/'pareto_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,x,title in [(axes[0],'sts','а) STS и поиск'),(axes[1],'ag','б) AG News и поиск')]:
        vals=stats[[(x,'mean'),('retrieval','mean')]].to_numpy()
        front=[]
        for i,b in enumerate(stats.index):
            dom=np.any(np.all(vals>=vals[i],axis=1)&np.any(vals>vals[i],axis=1))
            if not dom: front.append(i)
            chosen=b in SELECTED
            ax.errorbar(*vals[i],xerr=stats.loc[b,(x,'std')],yerr=stats.loc[b,('retrieval','std')],fmt='o',ms=4 if chosen else 2.5,color='#C54E52' if not dom else '#788B99',alpha=.9 if chosen else .5,lw=.5)
            if b in ['mix_s100_c000_r000','mix_s000_c100_r000','mix_s020_c000_r080']:
                ax.annotate(label(b),vals[i],xytext=(3,5),textcoords='offset points',fontsize=8)
        front=sorted(front,key=lambda i:vals[i,0])
        ax.plot(vals[front,0],vals[front,1],color='#C54E52',lw=.8)
        ax.axhline(0,color='gray',lw=.6)
        ax.axvline(0,color='gray',lw=.6)
        ax.set(title=title,xlabel='Δ STS' if x=='sts' else 'Δ AG News',ylabel='Δ поиска')
        ax.margins(.18)
    residual=pd.read_csv(MIX/'interaction_residuals.csv')
    subset=residual[(residual.family=='retrieval')&residual.branch.isin(['mix_s000_c080_r020','mix_s000_c060_r040','mix_s033_c033_r033'])]
    rs=subset.groupby('branch')[['observed_delta','expected_linear_delta','interaction_residual']].agg(['mean','std'])
    rs.to_csv(DATA/'interaction_examples.csv')
    ypos=np.arange(len(rs))
    for offset,c,color,name in [(-.16,'expected_linear_delta','#AFB8BF','Линейное ожидание'),(.16,'observed_delta','#2878B5','Наблюдение')]:
        axes[2].barh(ypos+offset,rs[(c,'mean')],height=.28,xerr=rs[(c,'std')],color=color,label=name,error_kw={'elinewidth':.6})
    axes[2].set_yticks(ypos,[label(b) for b in rs.index])
    axes[2].set(title='в) Нелинейный эффект',xlabel='Δ поиска')
    axes[2].axvline(0,color='gray',lw=.6)
    axes[2].legend(fontsize=8,loc='lower left')
    for ax in axes: ax.grid(alpha=.15)
    save(fig,'tradeoffs')


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False})
    for name in ['geometry_correlations.csv', 'family_deltas_by_seed.csv', 'selected_task_deltas.csv', 'exact_text_overlaps.csv', 'clean_sts_pair_counts.csv']:
        pd.read_csv(SOURCE_DATA/name).to_csv(DATA/name,index=False)
    for name in ['interaction_residuals.csv', 'attenuation_criteria.csv']:
        pd.read_csv(MIX/name).to_csv(DATA/name,index=False)
    design()
    matrix()
    diagnostics()
    pareto()


if __name__=='__main__':
    main()
