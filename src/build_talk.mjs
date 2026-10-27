// Import the supplied deck, reuse its cover and infographic layout, then export.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';
const root=path.resolve(import.meta.dirname,'..');
const out=path.join(root,'build/presentation');
const runtime=process.env.PRESENTATION_MODULES || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES=runtime;
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
try {
const original=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'rules/of.prezentaciya_russkiy_inzhener_shablon.pptx')));
const spec=JSON.parse(await fs.readFile(path.join(out,'deck.json'),'utf8'));
const proto=original.toProto();
proto.fonts=[]; // no unavailable embedded fonts are needed by the authored Arial text
const cover=structuredClone(proto.slides[0]);
const base=proto.slides[9];
proto.slides=[cover,...Array.from({length:5},(_,i)=>{
  const s=structuredClone(base); s.id=`conference-${i+2}`; s.index=i+1;
  s.elements=s.elements.filter(e=>e.id==='9'); // retain the template's pale-blue sidebar
  delete s.notesSlide;
  return s;
})];
delete cover.notesSlide;
const p=Presentation.load(proto);
const text=(slide,value,box,size,bold=false,color='#333333')=>{
  const sh=slide.shapes.add({geometry:'textbox',position:box,fill:'none',line:{fill:'none',width:0}});
  sh.text=value; sh.text.style={fontFamily:'Arial',typeface:'Arial',fontSize:size,bold,color,verticalAlignment:'middle'};
  return sh;
};
const first=p.slides.items[0];
first.shapes.deleteAll(); // cover branding is in its preserved background image
text(first,'Конгресс «Русский инженер»',{left:470,top:62,width:475,height:65},27,false,'#FFFFFF');
text(first,spec.slides[0].title,{left:155,top:225,width:930,height:200},43,true,'#FFFFFF');
text(first,spec.slides[0].subtitle,{left:155,top:425,width:890,height:90},25,false,'#FFFFFF');
text(first,spec.author.author_ru,{left:155,top:570,width:520,height:45},26,false,'#FFFFFF');
text(first,spec.author.email,{left:155,top:615,width:520,height:35},22,false,'#FFFFFF');
text(first,spec.author.affiliation_ru,{left:710,top:575,width:420,height:75},23,false,'#FFFFFF');
for(let i=1;i<6;i++){
  const s=p.slides.items[i], item=spec.slides[i];
  s.shapes.items[0].position.left=1030;
  s.shapes.items[0].position.width=250;
  text(s,item.title,{left:1050,top:190,width:210,height:130},i===1 ? 23 : 28,true,'#084D96');
  text(s,item.takeaway,{left:1050,top:340,width:210,height:210},22);
  text(s,String(i+1).padStart(2,'0'),{left:1170,top:645,width:70,height:40},20,false,'#084D96');
  const figureBox=[2,3,5].includes(i) ? {left:41,top:25,width:938,height:670} : {left:10,top:25,width:1000,height:670};
  s.images.add({blob:new Uint8Array(await fs.readFile(path.join(out,`figures/figure-${i}.png`))),contentType:'image/png',alt:`Рисунок ${i} статьи; все измеренные точки сохранены`,fit:'contain',position:figureBox});
}
await fs.writeFile(path.join(out,'layout.json'),JSON.stringify((await p.inspect({kind:'slide,textbox,image',maxChars:100000})).records,null,2));
const pptx=await PresentationFile.exportPptx(p);
const staging=path.join(out,'staging');
await fs.mkdir(staging,{recursive:true});
const candidate=path.join(staging,'candidate.pptx');
await pptx.save(candidate);
const skill=process.env.PRESENTATION_SKILL || path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.22136/skills/presentations');
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const finalPath=path.join(out,'cross_task_interference_talk.pptx');
// Only the named generated output is replaced, never the reference template.
await fs.rm(finalPath,{force:true});
await fs.mkdir(path.join(root,'build/presentation-validation'),{recursive:true});
await fs.rm(path.join(root,'build/presentation-validation/receipt.json'),{force:true});
await finalizePresentation({
 workspaceDir:root,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.PRESENTATION_PYTHON || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'),
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:6,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:path.join(root,'build/presentation-validation/receipt.json'),
});
await fs.mkdir(path.join(out,'previews'),{recursive:true});
for(let i=0;i<6;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(out,`previews/slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('Exported six slides using template slides 1 and 10; font fallback Arial.');
} catch(error){ console.error(error.stack?.split('\n').slice(0,8).join('\n') || error.message);process.exitCode=1; }
