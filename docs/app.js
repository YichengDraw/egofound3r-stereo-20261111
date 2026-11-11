import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { frameMatrix, projectPoint, geometryFrame, jointPixels, frameFromMediaTime } from './projection.js';

const $ = (id) => document.getElementById(id);
const COLORS = ['#64b6ff', '#f5b54f'];
const ACCENTS = ['#2c70d9', '#d38b2c', '#8c65b7'];
const EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
const TYPES = {float32:Float32Array,float64:Float64Array,uint8:Uint8Array,uint16:Uint16Array,uint32:Uint32Array,int32:Int32Array};
const state = {manifest:null,sequence:null,frame:0,playing:false,clock:0,last:0,panels:[],controller:null,loadId:0,updatingCamera:false};
const legacyCache=new Map();
const LEGACY_COLORS={hggt_native:'#fa74b8',hggt_adapted:'#b69bff',hawor:'#66dda5',gt:'#f3f6ff',ego:'#f4b24b'};
async function loadLegacy(entry,signal){
  if(legacyCache.has(entry.id))return legacyCache.get(entry.id);
  const job=(async()=>{const response=await fetch(entry.geometry_url,{signal});if(!response.ok)throw new Error(`HGGT 数据 ${response.status}`);const text=await response.text();
    const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text)),hash=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
    if(hash!==entry.geometry_sha256)throw new Error('HGGT 原始材料校验失败，已停止叠加');
    const source=JSON.parse(text);if(JSON.stringify(source.frame_ids)!==JSON.stringify(entry.frame_ids))throw new Error('HGGT 帧号不匹配');
    const result={};for(const key of ['hggt_native','hggt_adapted','hawor']){const m=source.methods[key];if(!m?.available)continue;
      result[key]={frame_count:source.frame_count,hand_count:2,joint_count:21,vertex_count:0,units:'m',coordinate_space:'camera',coordinate_convention:'opencv',hand_order:['left','right'],joint_edges:source.joint_edges,joints:new Float32Array(m.joints.flatMap(frame=>frame.flatMap(hand=>hand.flatMap(p=>p||[NaN,NaN,NaN])))),valid:typed(m.valid,'uint8'),joint_valid:typed(m.joint_valid,'uint8'),fill_mask:m.infilled?typed(m.infilled,'uint8'):null,source:m};
    }return result;})();legacyCache.set(entry.id,job);try{return await job;}catch(e){legacyCache.delete(entry.id);throw e;}
}
function deduplicateGTProjections(){
  const assigned=new Set(state.panels.filter(p=>p.method?.kind==='reference').map(p=>p.method.view));
  for(const p of state.panels){const hide=p.method?.kind==='reference'||assigned.has(p.method?.view);p.q('.projection-gt').hidden=hide;p.q('.projection-row').classList.toggle('single-projection',hide);if(!hide)assigned.add(p.method?.view);p.drawProjections();}
}
function renderLegacyMaterials(){
  const target=$('legacy-materials'),legacy=state.legacyManifest,entry=legacy?.samples.find(x=>x.id===state.sequence?.id);target.replaceChildren();
  const title=node('div','section-title');title.append(node('h2','','HGGT × HaWoR · 昨日原始材料'));target.append(title);
  const open=node('a','','打开完整原站 ↗');open.href=legacy?.source_site||'https://yichengdraw.github.io/hggt-hawor-singleview-20261110/';open.target='_blank';open.rel='noopener';title.append(open);
  if(entry){
    target.append(node('p','legacy-note',`同源对应：${entry.view==='cam_a'?'仅相机 A · 836212060125':'Aria RGB · 214-1'}。原视频、原始几何、指标与评价口径保持不变。HGGT 适配仅改变平移，不使用 GT；不把昨日共同有效帧指标与 EgoFound3R 全 GT 分母合并。`));
    const table=node('table'),thead=node('thead'),tr=node('tr');tr.append(node('th','','原方法'));for(const d of legacy.original_manifest.metric_definitions)tr.append(node('th','',d.label+(d.unit?` (${d.unit})`:'')));thead.append(tr);table.append(thead);const body=node('tbody');
    for(const m of entry.sample.metrics){const row=node('tr');row.append(node('td','',m.method));for(const d of legacy.original_manifest.metric_definitions)row.append(node('td','',fmt(m.values[d.key],d.decimals??2)));body.append(row);}table.append(body);const wrap=node('div','table-scroll');wrap.append(table);target.append(wrap,node('p','legacy-note',entry.sample.metrics_note||''));
    const links=node('div','downloads');for(const x of entry.sample.links){const a=node('a','',x.label);a.href=new URL(x.href,legacy.source_site).href;a.target='_blank';a.rel='noopener';links.append(a);}target.append(links);
  }else target.append(node('p','legacy-note','昨日没有这一片段的 HGGT / HaWoR 结果，因此没有补造视频、几何或指标。完整四段原材料仍可在下面查看。'));
  const details=node('details','legacy-original'),summary=node('summary','','展开昨日完整页面（原样保留全部材料与说明）');details.append(summary);details.addEventListener('toggle',()=>{if(details.open&&!details.querySelector('iframe')){const frame=node('iframe');frame.src=legacy.source_site;frame.title='HGGT × HaWoR 昨日完整原页面';frame.loading='lazy';details.append(frame);}});target.append(details);
}

const assetCache = new Map();
const settings = () => ({scene:$('scene').checked,hands:$('hands').checked,joints:$('joints').checked,trajectory:$('trajectory').checked,wireframe:$('wireframe').checked,color:$('color-mode').value,overlay:$('overlay').checked,filled:$('filled').checked});
const fmt = (v, digits=2) => v != null && !(typeof v==='string' && v.trim()==='') && Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : '—';
const finite = (v) => Number.isFinite(v);
const flatten = (v) => Array.isArray(v) ? v.flat(Infinity) : v;
const typed = (raw,dtype='float32') => {const Type=TYPES[dtype]||Float32Array;const vals=flatten(raw);return new Type((Type===Float32Array||Type===Float64Array)?Array.from(vals,v=>v==null?NaN:v):vals);};
const urlFor = (path,base=document.baseURI) => new URL(path,base).href;
const setText = (el,text) => {el.textContent = text ?? '';};

function node(tag, cls, text) {
  const el=document.createElement(tag); if(cls) el.className=cls; if(text!=null) el.textContent=text; return el;
}
function notice(message,error=false) {
  $('notice').hidden=!message; $('notice').classList.toggle('error',error); setText($('notice'),message);
}
async function fetchJSON(path,signal) {
  const r=await fetch(path,{signal,cache:'no-cache'}); if(!r.ok) throw new Error(`${r.status} · ${new URL(path).pathname}`); return r.json();
}
async function loadArray(spec,base,signal) {
  if(spec==null) return null;
  if(Array.isArray(spec)) return typed(spec);
  if(ArrayBuffer.isView(spec)) return spec;
  if(spec.data) return typed(spec.data,spec.dtype);
  if(typeof spec==='string') spec={url:spec};
  if(!spec.url) throw new Error('数组缺少 data 或 url');
  const u=urlFor(spec.url,base), key=`${u}|${spec.dtype||'float32'}`;
  if(assetCache.has(key)) return assetCache.get(key);
  const r=await fetch(u,{signal}); if(!r.ok) throw new Error(`数组读取失败 ${r.status} · ${new URL(u).pathname}`);
  let arr;
  if(new URL(u).pathname.endsWith('.json')) arr=typed(await r.json(),spec.dtype);
  else {const bytes=await r.arrayBuffer(); arr=new (TYPES[spec.dtype]||Float32Array)(bytes);}
  const expected=spec.shape?.reduce((a,b)=>a*b,1);
  if(expected!=null && expected!==arr.length) throw new Error(`数组尺寸不符：${new URL(u).pathname}，需要 ${expected}，得到 ${arr.length}`);
  assetCache.set(key,arr); return arr;
}
function resolveHandLayout(g) {
  const count=g.hand_count||2;
  const order=Array.isArray(g.hand_order)?g.hand_order:typeof g.hand_order==='string'?g.hand_order.split(',').map(s=>s.trim()):[];
  const sides=Array.from({length:count},(_,i)=>['left','right'].includes(order[i])?order[i]:null);
  const sameIndices=(a,b)=>ArrayBuffer.isView(a)&&ArrayBuffer.isView(b)&&a.length===b.length&&a.every((v,i)=>v===b[i]);
  // An unknown hand side is compatible with an explicitly shared topology.
  // Legacy left/right keys establish shared topology only if every index agrees.
  const common=ArrayBuffer.isView(g.faces)?g.faces:sameIndices(g.faces?.left,g.faces?.right)?g.faces.left:null;
  const faces=Array.from({length:count},(_,i)=>g.faces_by_slot?.[i]||common||(sides[i]?g.faces?.[sides[i]]:null)||null);
  const labels=Array.from({length:count},(_,i)=>g.hand_labels?.[i]||(sides[i]==='left'?'左手':sides[i]==='right'?'右手':`native slot ${i}`));
  const colorIndices=sides.map((side,i)=>side==='left'?0:side==='right'?1:i%COLORS.length);
  return {sides,faces,labels,colorIndices,missingFaceSlots:faces.flatMap((face,i)=>face?[]:[i])};
}
async function loadGeometry(path,signal) {
  const u=urlFor(path), g=await fetchJSON(u,signal);
  const out={...g,_base:u,_warnings:[]};
    const fields=['vertices','joints','valid','display_valid','official_valid','contact','visibility','joints2d','observed','network_anchor','fill_mask','timestamps','projection_K','joint_valid'];
  const jobs=fields.map(async k=>{if(g[k]!=null) out[k]=await loadArray(g[k],u,signal);});
  if(g.faces) jobs.push((async()=>{
    if(g.faces.left || g.faces.right) out.faces={left:await loadArray(g.faces.left,u,signal),right:await loadArray(g.faces.right,u,signal)};
    else out.faces=await loadArray(g.faces,u,signal);
  })());
  if(g.faces_by_slot) jobs.push((async()=>{out.faces_by_slot=await Promise.all(g.faces_by_slot.map(spec=>loadArray(spec,u,signal)));})());
  if(g.scene) jobs.push((async()=>{const shape=g.scene.points?.shape;out.scene={...g.scene,point_count:g.scene.point_count||shape?.[shape.length-2],frame_count:g.scene.frame_count||g.scene.frame_indices?.length||(shape?.length===3?shape[0]:1),points:await loadArray(g.scene.points,u,signal),colors:await loadArray(g.scene.colors,u,signal),valid:await loadArray(g.scene.valid,u,signal)};})());
  if(g.series) jobs.push((async()=>{out.series=await Promise.all(g.series.map(async s=>({...s,values:await loadArray(s.values,u,signal)})));})());
  await Promise.all(jobs);
  out.frame_count=Number(g.frame_count||g.vertices?.shape?.[0]||g.joints?.shape?.[0]||state.sequence.frames);
  out.hand_count=Number(g.hand_count??2); out.vertex_count=Number(g.vertex_count??778); out.joint_count=Number(g.joint_count??21);
  // Preserve strict validity; filled display is an explicit user toggle.
  out._handLayout=resolveHandLayout(out);
  if(!out.vertices && !out.joints && !out.scene?.points) throw new Error('该结果没有可显示的网格、关节或场景点云。');
  return out;
}

async function loadProjectionGeometry(path,signal){
  const u=urlFor(path),g=await fetchJSON(u,signal),out={...g,_base:u};
  const fields=['vertices','joints','valid','display_valid','fill_mask','joints2d','projection_K','joint_valid'];
  await Promise.all(fields.map(async k=>{if(g[k]!=null)out[k]=await loadArray(g[k],u,signal);}));
  if(g.faces)out.faces=g.faces.left||g.faces.right?{left:await loadArray(g.faces.left,u,signal),right:await loadArray(g.faces.right,u,signal)}:await loadArray(g.faces,u,signal);
  out._handLayout=resolveHandLayout(out);return out;
}

class SoftwareRenderer {
  constructor(){
    this.domElement=document.createElement('canvas');this.context=this.domElement.getContext('2d');
    if(!this.context)throw new Error('Canvas 2D 不可用');
    this.isSoftwareRenderer=true;this.pixelRatio=1;this.width=1;this.height=1;this.sampled=false;
  }
  setPixelRatio(value){this.pixelRatio=value;}
  setSize(width,height){this.width=width;this.height=height;this.domElement.width=Math.round(width*this.pixelRatio);this.domElement.height=Math.round(height*this.pixelRatio);}
  render(scene,camera){
    const ctx=this.context,w=this.width,h=this.height,dpr=this.pixelRatio;
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.globalAlpha=1;ctx.fillStyle=scene.background?.isColor?scene.background.getStyle(THREE.SRGBColorSpace):'#15202d';ctx.fillRect(0,0,w,h);
    scene.updateMatrixWorld(true);camera.updateMatrixWorld(true);
    const commands=[],tmp=new THREE.Vector3(),view=new THREE.Vector3(),color=new THREE.Color();this.sampled=false;
    const project=(position,index,matrix)=>{
      const x=position.getX(index),y=position.getY(index),z=position.getZ(index);if(![x,y,z].every(finite))return null;
      tmp.set(x,y,z).applyMatrix4(matrix);view.copy(tmp).applyMatrix4(camera.matrixWorldInverse);tmp.project(camera);
      if(![tmp.x,tmp.y,tmp.z].every(finite)||tmp.z < -1||tmp.z > 1||view.z>=0)return null;
      return {x:(tmp.x+1)*w/2,y:(1-tmp.y)*h/2,z:tmp.z,depth:-view.z};
    };
    scene.traverseVisible(object=>{
      if(!(object.isPoints||object.isLine||object.isMesh))return;
      const geometry=object.geometry,position=geometry?.attributes.position,material=Array.isArray(object.material)?object.material[0]:object.material;if(!position||!material?.visible)return;
      const opacity=material.opacity??1,overlay=material.depthTest===false,colors=geometry.attributes.color;
      const cssColor=index=>{if(material.vertexColors&&colors){color.setRGB(colors.getX(index),colors.getY(index),colors.getZ(index));return color.getStyle(THREE.SRGBColorSpace);}return material.color?.getStyle(THREE.SRGBColorSpace)||'#cad8e8';};
      const cache=new Map(),point=index=>{if(!cache.has(index))cache.set(index,project(position,index,object.matrixWorld));return cache.get(index);};
      const start=Math.max(0,geometry.drawRange.start||0),maxCount=geometry.index?.count??position.count,end=Math.min(maxCount,start+(geometry.drawRange.count??Infinity));
      const at=i=>geometry.index?geometry.index.getX(i):i;
      const segment=(a,b,alpha=opacity)=>{const p=point(a),q=point(b);if(!p||!q)return;commands.push({kind:'line',p,q,z:(p.z+q.z)/2,color:cssColor(a),alpha,overlay,width:object.isMesh?.8:Math.max(1.4,material.linewidth||1)});};
      if(object.isPoints){
        const stride=Math.max(1,Math.ceil((end-start)/16000));if(stride>1)this.sampled=true;
        for(let i=start;i<end;i+=stride){const index=at(i),p=point(index);if(!p||p.x < -20||p.x>w+20||p.y < -20||p.y>h+20)continue;
          const radius=material.sizeAttenuation===false?(material.size||2)/2:(material.size||.004)*h*camera.projectionMatrix.elements[5]/(4*p.depth);
          commands.push({kind:'point',p,z:p.z,color:cssColor(index),alpha:opacity,overlay,radius:Math.max(1.1,Math.min(14,radius))});
        }
      }else if(object.isMesh){
        const triangles=Math.floor((end-start)/3),stride=Math.max(1,Math.ceil(triangles/6000));if(stride>1)this.sampled=true;
        for(let t=0;t<triangles;t+=stride){const i=start+t*3,a=at(i),b=at(i+1),c=at(i+2);segment(a,b,opacity*.55);segment(b,c,opacity*.55);segment(c,a,opacity*.55);}
      }else{
        const step=object.isLineSegments?2:1,total=Math.max(0,Math.floor((end-start-1)/step)+1),stride=Math.max(1,Math.ceil(total/20000));if(stride>1)this.sampled=true;
        for(let i=start;i+1<end;i+=step*stride)segment(at(i),at(i+1));
      }
    });
    commands.sort((a,b)=>Number(a.overlay)-Number(b.overlay)||b.z-a.z);ctx.lineCap='round';ctx.lineJoin='round';
    for(const item of commands){ctx.globalAlpha=item.alpha;if(item.kind==='line'){ctx.strokeStyle=item.color;ctx.lineWidth=item.width;ctx.beginPath();ctx.moveTo(item.p.x,item.p.y);ctx.lineTo(item.q.x,item.q.y);ctx.stroke();}else{ctx.fillStyle=item.color;ctx.beginPath();ctx.arc(item.p.x,item.p.y,item.radius,0,Math.PI*2);ctx.fill();}}
    ctx.globalAlpha=1;
  }
  dispose(){this.context.clearRect(0,0,this.width,this.height);}
  forceContextLoss(){}
}

class SceneViewer {
  constructor(host,panel) {
    this.host=host;this.panel=panel;this.data=null;this.meshes=[];this.skeletons=[];this.jointPoints=[];this.traces=[];this.disposed=false;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#15202d');
    this.camera=new THREE.PerspectiveCamera(48,1,.001,200);
    this.camera.position.set(0,.15,1);
    const canvas=document.createElement('canvas');let gl=null;
    try{gl=canvas.getContext('webgl2',{antialias:true,alpha:false,powerPreference:'low-power'});}catch(e){}
    try{this.renderer=gl?new THREE.WebGLRenderer({canvas,context:gl,antialias:true,alpha:false,powerPreference:'low-power'}):new SoftwareRenderer();}catch(e){this.renderer=new SoftwareRenderer();}
    this.isSoftware=!!this.renderer.isSoftwareRenderer;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    host.append(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=false;this.controls.minDistance=.015;this.controls.maxDistance=150;
    this.controls.addEventListener('change',()=>{this.render();if(!state.updatingCamera&&$('linked').checked)syncCameras(this);});
    this.root=new THREE.Group();this.scene.add(this.root);
    this.scene.add(new THREE.HemisphereLight(0xe9f1ff,0x61758c,2.7));
    const light=new THREE.DirectionalLight(0xffffff,2.2);light.position.set(1,2,3);this.scene.add(light);
    const back=new THREE.DirectionalLight(0x85b9ff,.8);back.position.set(-2,-1,-3);this.scene.add(back);
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(host);this.resize();
  }
  resize(){if(this.disposed)return;const w=this.host.clientWidth,h=this.host.clientHeight;if(!w||!h)return;this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h,false);this.render();this.panel.drawOverlay();this.panel.drawChart();}
  setData(data){
    this.data=data;this.root.scale.set(1,data.coordinate_convention==='opencv'?-1:1,data.coordinate_convention==='opencv'?-1:1);
    this.scale=data.units==='mm'?.001:1;
    this.buildHands();this.buildScene();this.buildTrajectories();this.update(state.frame);this.fit();
  }
  addComparisonLayer(key,data,label){
    this.layers=this.layers||new Map();if(this.layers.has(key))return;
    const group=new THREE.Group(),items=[];this.root.add(group);const edges=data.joint_edges||EDGES,color=LEGACY_COLORS[key]||'#ffffff';
    for(let h=0;h<data.hand_count;h++){
      const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(edges.length*6),3));
      const line=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color,transparent:true,opacity:.95,depthTest:false}));line.frustumCulled=false;group.add(line);
      const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(data.joint_count*3),3));const points=new THREE.Points(pg,new THREE.PointsMaterial({color,size:.005,sizeAttenuation:true,depthTest:false}));points.frustumCulled=false;group.add(points);items.push({line,points});
    }
    this.layers.set(key,{data,group,items,edges,label,enabled:false});group.visible=false;
  }
  updateComparisonLayers(frame){
    for(const layer of this.layers?.values()||[]){layer.group.visible=layer.enabled;if(!layer.enabled)continue;const g=layer.data,f=geometryFrame(g,frame);
      for(let h=0;h<g.hand_count;h++){const item=layer.items[h],valid=!g.valid||g.valid[f*g.hand_count+h]>.5;item.line.visible=item.points.visible=valid;if(!valid)continue;
        const base=(f*g.hand_count+h)*g.joint_count*3,values=item.points.geometry.attributes.position.array;
        for(let j=0;j<g.joint_count;j++)for(let k=0;k<3;k++)values[j*3+k]=g.joint_valid&&!g.joint_valid[(f*g.hand_count+h)*g.joint_count+j]?NaN:g.joints[base+j*3+k];
        const out=item.line.geometry.attributes.position.array;layer.edges.forEach(([a,b],i)=>{for(let k=0;k<3;k++){out[i*6+k]=values[a*3+k];out[i*6+3+k]=values[b*3+k];}});item.line.geometry.attributes.position.needsUpdate=true;item.points.geometry.attributes.position.needsUpdate=true;item.line.material.opacity=g.fill_mask?.[f*g.hand_count+h]?.45:.95;
      }
    }
    if(this.primaryVisible===false)for(const o of [...this.meshes,...this.skeletons,...this.jointPoints,...this.traces])o.visible=false;
  }
  buildHands(){
    const g=this.data, V=g.vertex_count, J=g.joint_count;this.edges=g.joint_edges||EDGES;this.layout=g._handLayout||resolveHandLayout(g);
    for(let h=0;h<g.hand_count;h++){
      const side=this.layout.sides[h],index=this.layout.colorIndices[h];
      if(g.vertices){
        const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(V*3),3));geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(V*3),3));
        const faces=this.layout.faces[h];
        if(faces) geo.setIndex(new THREE.BufferAttribute(new Uint32Array(faces),1));
        const mat=new THREE.MeshStandardMaterial({color:0xffffff,vertexColors:true,roughness:.68,metalness:.04,side:THREE.DoubleSide});
        const mesh=faces?new THREE.Mesh(geo,mat):new THREE.Points(geo,new THREE.PointsMaterial({size:.004,vertexColors:true,sizeAttenuation:true}));
        mesh.frustumCulled=false;mesh.userData={side,index};this.root.add(mesh);this.meshes.push(mesh);
      }
      if(g.joints){
        const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(this.edges.length*6),3));
        const lines=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color:COLORS[index],transparent:true,opacity:.95}));lines.frustumCulled=false;this.root.add(lines);this.skeletons.push(lines);
        const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(J*3),3));
        const jp=new THREE.Points(pg,new THREE.PointsMaterial({color:COLORS[index],size:.007,sizeAttenuation:true,depthTest:false}));jp.frustumCulled=false;this.root.add(jp);this.jointPoints.push(jp);
      }
    }
  }
  buildScene(){
    if(!this.data.scene?.points)return;
    const s=this.data.scene,shape=this.data.scene_shape||null;
    this.sceneCount=Number(s.point_count||s.points_per_frame||shape?.[1]||s.shape?.[1]||s.points.length/(3*(s.frame_count||s.frame_indices?.length||1)));
    this.sceneCount=Math.floor(this.sceneCount);
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(this.sceneCount*3),3));geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(this.sceneCount*3),3));
    this.cloud=new THREE.Points(geo,new THREE.PointsMaterial({size:s.point_size||.004,vertexColors:true,sizeAttenuation:true,transparent:true,opacity:.85}));this.cloud.frustumCulled=false;this.root.add(this.cloud);
  }
  buildTrajectories(){
    const g=this.data;if(!g.joints)return;
    for(let h=0;h<g.hand_count;h++){
      const pts=[];for(let f=1;f<g.frame_count;f++){
        if(g.valid&&(!g.valid[f*g.hand_count+h]||!g.valid[(f-1)*g.hand_count+h]))continue;
        const a=((f-1)*g.hand_count+h)*g.joint_count*3,b=(f*g.hand_count+h)*g.joint_count*3;
        if([...g.joints.subarray(a,a+3),...g.joints.subarray(b,b+3)].every(finite))for(const k of [a,b])pts.push(g.joints[k]*this.scale,g.joints[k+1]*this.scale,g.joints[k+2]*this.scale);
      }
      const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
      const line=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color:COLORS[this.layout.colorIndices[h]],transparent:true,opacity:.55}));this.root.add(line);this.traces.push(line);
    }
  }
  handValid(frame,h){const g=this.data,mask=settings().filled&&g.display_valid?g.display_valid:g.valid;return !mask || mask[frame*g.hand_count+h]>.5;}
  geometryFrame(frame){
    const g=this.data;if(g.frame_indices?.length){let best=0;while(best+1<g.frame_indices.length&&Math.abs(g.frame_indices[best+1]-frame)<Math.abs(g.frame_indices[best]-frame))best++;return best;}
    return Math.min(g.frame_count-1,frame);
  }
  update(frame){
    if(!this.data)return;const g=this.data,f=this.geometryFrame(frame),opts=settings(),V=g.vertex_count,J=g.joint_count,H=g.hand_count;
    this.frame=f;
    for(let h=0;h<H;h++){
      const valid=this.handValid(f,h),filled=!!g.fill_mask?.[f*H+h];
      if(this.meshes[h]){
        const mesh=this.meshes[h];mesh.visible=opts.hands&&valid;mesh.material.wireframe=opts.wireframe;mesh.material.transparent=filled;mesh.material.opacity=filled?.42:1;
        const positions=mesh.geometry.attributes.position.array,colors=mesh.geometry.attributes.color.array,base=(f*H+h)*V*3;
        for(let i=0;i<V*3;i++)positions[i]=g.vertices[base+i]*this.scale;
        const attr=opts.color==='contact'?g.contact:opts.color==='visibility'?g.visibility:null;
        const defaultColor=new THREE.Color(COLORS[mesh.userData.index]);
        for(let v=0;v<V;v++){
          const p=attr?.[(f*H+h)*V+v],c=attr&&finite(p)?heatColor(p):defaultColor;
          colors[v*3]=c.r;colors[v*3+1]=c.g;colors[v*3+2]=c.b;
        }
        mesh.geometry.attributes.position.needsUpdate=true;mesh.geometry.attributes.color.needsUpdate=true;
        if(mesh.isMesh&&!this.isSoftware)mesh.geometry.computeVertexNormals();
      }
      if(this.skeletons[h]){
        const line=this.skeletons[h],jp=this.jointPoints[h];line.visible=jp.visible=opts.joints&&valid;line.material.opacity=filled?.42:.95;
        const base=(f*H+h)*J*3,positions=line.geometry.attributes.position.array,jpa=jp.geometry.attributes.position.array;
        for(let i=0;i<J*3;i++)jpa[i]=g.joints[base+i]*this.scale;
        this.edges.forEach(([a,b],e)=>{for(let j=0;j<3;j++){positions[e*6+j]=jpa[a*3+j];positions[e*6+3+j]=jpa[b*3+j];}});
        line.geometry.attributes.position.needsUpdate=true;jp.geometry.attributes.position.needsUpdate=true;
      }
      if(this.traces[h])this.traces[h].visible=opts.trajectory;
    }
    if(this.cloud){
      const s=g.scene,frames=s.frame_count||s.frame_indices?.length||Math.max(1,s.points.length/(3*this.sceneCount));
      let sf=Math.min(frames-1,f);
      if(s.frame_indices?.length){sf=0;while(sf+1<s.frame_indices.length&&Math.abs(s.frame_indices[sf+1]-frame)<Math.abs(s.frame_indices[sf]-frame))sf++;}
      this.cloud.visible=opts.scene;
      if(this.lastSceneFrame!==sf){this.lastSceneFrame=sf;
      const positions=this.cloud.geometry.attributes.position.array,colors=this.cloud.geometry.attributes.color.array,base=sf*this.sceneCount*3;
      for(let i=0;i<this.sceneCount*3;i++){const isValid=!s.valid||s.valid[sf*this.sceneCount+Math.floor(i/3)]>.5;positions[i]=isValid?s.points[base+i]*this.scale:NaN;}
      const col=s.colors,colBase=col?.length===this.sceneCount*3?0:base;
      const div=col instanceof Uint8Array||s.colors_range==='0-255'?255:1;
      for(let i=0;i<this.sceneCount*3;i++)colors[i]=col?(col[colBase+i]/div):.5;
      this.cloud.visible=opts.scene;this.cloud.geometry.attributes.position.needsUpdate=true;this.cloud.geometry.attributes.color.needsUpdate=true;}
    }
    this.updateComparisonLayers(frame);this.render();
  }
  fit(){
    if(!this.data)return;const g=this.data,arr=g.vertices||g.joints,pts=[],f=this.frame||0;
    if(arr&&this.primaryVisible!==false){const count=g.vertices?g.vertex_count:g.joint_count;for(let h=0;h<g.hand_count;h++){if(!this.handValid(f,h))continue;const base=(f*g.hand_count+h)*count*3;for(let i=0;i<count;i+=Math.max(1,Math.floor(count/180))){const p=new THREE.Vector3(arr[base+i*3]*this.scale,arr[base+i*3+1]*this.scale,arr[base+i*3+2]*this.scale);if([p.x,p.y,p.z].every(finite))pts.push(p);}}}
    for(const layer of this.layers?.values()||[]){if(!layer.enabled)continue;const d=layer.data,lf=geometryFrame(d,this.panel.displayFrame??state.frame);for(let h=0;h<d.hand_count;h++){if(d.valid&&!d.valid[lf*d.hand_count+h])continue;for(let j=0;j<d.joint_count;j++){const b=((lf*d.hand_count+h)*d.joint_count+j)*3,p=new THREE.Vector3(d.joints[b],d.joints[b+1],d.joints[b+2]);if([p.x,p.y,p.z].every(finite))pts.push(p);}}}
    if(!pts.length && g.scene?.points){for(let i=0;i<Math.min(g.scene.points.length,6000);i+=30){const p=new THREE.Vector3(g.scene.points[i]*this.scale,g.scene.points[i+1]*this.scale,g.scene.points[i+2]*this.scale);if([p.x,p.y,p.z].every(finite))pts.push(p);}}
    if(!pts.length)return;
    const box=new THREE.Box3().setFromPoints(pts),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());center.multiply(this.root.scale);
    const span=Math.max(.15,Math.min(10,Math.max(size.x,size.y,size.z))),distance=span/(2*Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2)))*1.55;
    state.updatingCamera=true;this.controls.target.copy(center);this.camera.position.copy(center).add(new THREE.Vector3(span*.18,span*.06,distance));this.camera.up.set(0,1,0);this.camera.near=Math.max(.001,distance/1000);this.camera.far=Math.max(100,distance*100);this.camera.updateProjectionMatrix();this.controls.update();state.updatingCamera=false;this.render();
  }
  render(){if(!this.disposed)this.renderer.render(this.scene,this.camera);}
  dispose(){this.disposed=true;this.resizeObserver.disconnect();this.controls.dispose();this.scene.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});this.renderer.dispose();this.renderer.forceContextLoss();this.host.replaceChildren();}
}

function heatColor(p){
  const stops=[[.21,.48,.90],[.09,.71,.72],[.94,.82,.35],[.96,.43,.27]],q=Math.max(0,Math.min(1,p))*3,a=Math.min(2,Math.floor(q)),t=q-a;
  return {r:stops[a][0]*(1-t)+stops[a+1][0]*t,g:stops[a][1]*(1-t)+stops[a+1][1]*t,b:stops[a][2]*(1-t)+stops[a+1][2]*t};
}
function syncCameras(source){
  if(state.updatingCamera)return;state.updatingCamera=true;
  for(const p of state.panels){const v=p.viewer;if(!v||v===source)continue;v.camera.position.copy(source.camera.position);v.camera.quaternion.copy(source.camera.quaternion);v.camera.up.copy(source.camera.up);v.controls.target.copy(source.controls.target);v.camera.near=source.camera.near;v.camera.far=source.camera.far;v.camera.updateProjectionMatrix();v.controls.update();v.render();}
  state.updatingCamera=false;
}

class MethodPanel {
  constructor(index,methodId){
    this.index=index;this.el=$('method-template').content.firstElementChild.cloneNode(true);this.method=null;this.geometry=null;this.viewer=null;this.disposed=false;this.loadId=0;
    this.q=s=>this.el.querySelector(s);this.video=this.q('video');this.archiveVideo=this.q('.archive-video');this.canvas=this.q('.overlay-canvas');this.q('.method-accent').style.background=ACCENTS[index%3];
    const select=this.q('.method-select');for(const m of state.sequence.methods||[]){const op=node('option','',m.label||m.id);op.value=m.id;select.append(op);}select.value=methodId;
    select.addEventListener('change',()=>{this.load(select.value);saveQuery();});
    this.q('.fit-camera').addEventListener('click',()=>{this.viewer?.fit();if($('linked').checked&&this.viewer)syncCameras(this.viewer);});
    this.video.addEventListener('seeked',()=>{if(!this.video.seeking)this.present(this.mediaFrame());});
    this.video.addEventListener('loadeddata',()=>{if(!state.playing){this.seek(state.frame);this.present(state.frame);}});
    this.video.addEventListener('ended',()=>{if(state.playing&&state.panels[0]===this){setPlaying(false);if($('loop').checked){setFrame(0);setPlaying(true);}else setFrame(state.sequence.frames-1);}});
    this.video.addEventListener('seeking',()=>{this.seekCount=(this.seekCount||0)+1;this.el.dataset.seekCount=this.seekCount;});
    this.resizeObserver=new ResizeObserver(()=>{this.drawOverlay();this.drawProjections();});this.resizeObserver.observe(this.el);
    this.startVideoFrames();
    this.archiveVideo.addEventListener('error',()=>setText(this.q('.legacy-scope'),'昨日原视频加载失败，可从下方原材料链接打开；当前 RGB 与预测未受影响。'));
    this.video.addEventListener('error',()=>{const m=this.q('.video-missing');m.hidden=false;setText(m,'视频读取失败。请检查资源路径或在下载区打开原文件。');});
    $('comparison').append(this.el);
    this.load(methodId);
  }
  async load(id){
    setPlaying(false);const token=++this.loadId;this.projectionGeometry=null;this.referenceGeometry=null;this.meshPaths=null;this.method=(state.sequence.methods||[]).find(x=>x.id===id);this.geometry=null;
    const m=this.method;if(!m)return;this.q('.method-checks').replaceChildren();setText(this.q('.comparison-scope'),'正在加载同坐标比较几何…');this.setupVideoTabs();for(const pane of this.qAll('.projection-pane')){delete pane.dataset.validHands;delete pane.dataset.projected;}
    this.q('.method-select').value=id;const kind=this.q('.method-kind');kind.className='method-kind '+(m.kind||'');setText(kind,({official:'官方流程',adapted:'双目适配',reference:'参考真值',baseline:'已有基线'})[m.kind]||m.kind||'推理结果');
    setText(this.q('.method-description'),m.summary||m.notes?.[0]||'');
    setText(this.q('.coordinate-label'),m.coordinate_system||state.sequence.coordinate_system||'');
    setText(this.q('.video-view'),m.view_label||(m.view==='right'?'右相机':m.view==='left'?'左相机':m.view||''));
    this.q('.video-missing').hidden=true;this.video.pause();this.video.removeAttribute('src');
    const rgb=state.sequence.rgb?.[m.view||'left'],video=m.overlay_video||m.video||rgb?.video;
    this.rgb=rgb;this.bakedOverlay=!!m.overlay_video;setText(this.q('.rgb-title'),this.bakedOverlay?'RGB + 模型投影':'RGB + 2D 关节');
    if(m.kind==='reference')setText(this.q('.rgb-title'),'RGB + GT 投影');
    setText(this.q('.scene-panel .pane-label>span:first-child'),m.kind==='reference'?'GT 标注骨架':'可交互 3D');
    const poster=m.poster||rgb?.poster;if(poster)this.video.poster=urlFor(poster);else this.video.removeAttribute('poster');
    if(video){this.video.src=urlFor(video);this.video.load();}else{const missing=this.q('.video-missing');missing.hidden=false;setText(missing,m.video_missing_reason||'未导出该视角的视频。');}
    if(this.viewer){this.viewer.dispose();this.viewer=null;}
    const message=this.q('.scene-message');message.hidden=false;setText(message,m.status==='complete'||m.geometry?'正在加载几何结果…':m.missing_reason||'该方法尚无已完成的几何结果。');
    this.q('.frame-metrics').replaceChildren();this.q('.chart-container').classList.remove('has-data');
    if(!m.geometry){setText(this.q('.rgb-title'),'输入 RGB · 待推理');this.q('.scene-legend').hidden=true;this.q('.fit-camera').hidden=true;this.update(state.frame);if(!(state.sequence.methods||[]).some(method=>method.geometry))setText($('load-status'),'RGB 输入就绪 · 待推理');return;}
    this.q('.scene-legend').hidden=false;this.q('.fit-camera').hidden=false;
    try{
      const geom=await loadGeometry(m.geometry,state.controller?.signal);
      if(this.disposed||token!==this.loadId)return;this.geometry=geom;
      if(geom.joints&&!geom.vertices)$('joints').checked=true;
      const layout=geom._handLayout;
      this.qAll('.scene-legend>span:not(.heat-legend)').forEach((el,i)=>el.replaceChildren(node('i','dot '+(layout.colorIndices[i]?'right':'left')),document.createTextNode(layout.labels[i]||`native slot ${i}`)));
      try{this.viewer=new SceneViewer(this.q('.webgl-host'),this);this.viewer.setData(geom);message.hidden=true;if(this.viewer.isSoftware)setText(this.q('.scene-panel .pane-label>span:first-child'),m.kind==='reference'?'GT 标注骨架 · 软件 3D':'软件 3D · 网格线框');}catch(e){this.webglError=true;setText(message,`此浏览器暂时无法创建三维视图。\n${e.message}`);}
      if(!this.bakedOverlay&&!geom.joints2d&&geom.projection_space!=='camera')setText(this.q('.rgb-title'),'输入 RGB · 未导出 2D 投影');
      setText(this.q('.coordinate-label'),m.coordinate_system||geom.coordinate_system||state.sequence.coordinate_system||`${geom.coordinate_convention==='opencv'?'相机坐标':'三维坐标'} · ${geom.units||'m'}`);
      let projectionGeometry=geom;
      // World vertices must never be fed directly into the RGB camera K.
      if(geom.coordinate_space==='world'){
        const cameraMethod=state.sequence.methods.find(x=>x.kind!=='reference'&&x.view===m.view&&!x.id.endsWith('_world'));
        if(cameraMethod)projectionGeometry=await loadProjectionGeometry(cameraMethod.geometry,state.controller?.signal);
      }
      const reference=state.sequence.methods.find(x=>x.kind==='reference'&&x.view===m.view);
      const referenceGeometry=reference?(m.kind==='reference'?geom:await loadProjectionGeometry(reference.geometry,state.controller?.signal)):null;
      if(this.disposed||token!==this.loadId)return;
      this.projectionGeometry=projectionGeometry;this.referenceGeometry=referenceGeometry;
      this.q('.projection-gt').hidden=m.kind==='reference';
      this.q('.projection-row').classList.toggle('reference-only',m.kind==='reference');
      setText(this.q('.projection-pred .projection-title'),m.kind==='reference'?'GT 骨架投影 · 真实标注':'预测手网格投影 · 原始几何');
      this.update(state.frame);deduplicateGTProjections();this.setupComparisonLayers(token).catch(e=>{if(!this.disposed&&token===this.loadId&&e.name!=='AbortError')setText(this.q('.comparison-scope'),`比较几何加载失败：${e.message}`);});
      const first=state.panels.find(p=>p.viewer?.data)?.viewer;if($('linked').checked&&first)syncCameras(first);
      setText($('load-status'),'几何与视频按同一帧同步');
    }catch(e){if(e.name==='AbortError'||token!==this.loadId)return;setText(message,`几何结果读取失败：${e.message}`);this.el.dataset.error='true';setText($('load-status'),'部分资源读取失败，请查看对应面板');$('load-status').classList.add('error');}
  }
  setupVideoTabs(){
    const entry=state.legacyManifest?.samples.find(x=>x.id===state.sequence.id&&x.view===this.method.view);this.legacyEntry=entry;this.activeVideoTab='input';this.archiveVideo.pause();this.archiveVideo.removeAttribute('src');this.archiveVideo.load();this.archiveVideo.onloadedmetadata=null;this.archiveVideo.hidden=true;this.video.hidden=false;this.canvas.hidden=false;this.q('.video-panel').classList.remove('archive-active');
    const tabs=this.q('.video-tabs');tabs.replaceChildren();
    const choices=[{id:'input',label:this.method.kind==='reference'?'RGB + GT':'EgoFound3R / RGB'},...(entry?.videos||[])];
    for(const choice of choices){const b=node('button','video-tab',choice.label);b.type='button';b.setAttribute('role','tab');b.setAttribute('aria-selected',String(choice.id==='input'));b.addEventListener('click',()=>{
      setPlaying(false);this.activeVideoTab=choice.id;for(const el of tabs.children)el.setAttribute('aria-selected',String(el===b));const archived=choice.id!=='input';this.video.hidden=archived;this.canvas.hidden=archived;this.archiveVideo.hidden=!archived;this.q('.video-panel').classList.toggle('archive-active',archived);
      if(archived){this.archiveFPS=choice.fps;this.archiveVideo.src=choice.url;this.archiveVideo.load();setText(this.q('.rgb-title'),choice.label+' · 原片未改');this.archiveVideo.onloadedmetadata=()=>this.seekArchive(state.frame);}
      else{this.archiveVideo.pause();this.archiveVideo.removeAttribute('src');this.archiveVideo.load();setText(this.q('.rgb-title'),this.method.kind==='reference'?'RGB + GT 投影':'RGB + 2D 关节');}
    });tabs.append(b);}
    setText(this.q('.legacy-scope'),entry?'昨日视频与当前片段逐源帧对应；投影仍使用下方同帧 RGB。':state.legacyManifest?.samples.some(x=>x.id===state.sequence.id)?'昨日 HGGT / HaWoR 仅有相机 A；此相机没有对应结果。':'昨日没有此片段的 HGGT / HaWoR 结果。');
  }
  seekArchive(frame){if(this.activeVideoTab==='input'||this.archiveVideo.readyState<1)return;const t=(frame+.05)/(this.archiveFPS||30);if(Math.abs(this.archiveVideo.currentTime-t)>.1/(this.archiveFPS||30))this.archiveVideo.currentTime=Math.min(t,this.archiveVideo.duration-.001);}
  async setupComparisonLayers(token){
    const v=this.viewer,box=this.q('.method-checks');box.replaceChildren();if(!v)return;
    const add=(key,label,primary=false)=>{const wrap=node('label','method-check'),input=node('input');input.type='checkbox';input.checked=primary;input.setAttribute('aria-label',`3D 显示 ${label}`);const dot=node('i');dot.style.background=LEGACY_COLORS[key]||'#64b6ff';wrap.append(input,dot,document.createTextNode(label));input.addEventListener('change',()=>{if(primary)v.primaryVisible=input.checked;else if(v.layers?.has(key))v.layers.get(key).enabled=input.checked;v.update(this.displayFrame??state.frame);if(input.checked)v.fit();if($('linked').checked)syncCameras(v);});box.append(wrap);};
    add('primary',this.method.kind==='reference'?'GT（本列）':'EgoFound3R（本列）',true);
    if(this.geometry.coordinate_space==='world'){setText(this.q('.comparison-scope'),'本列是独立预测世界系，不能直接叠加昨日相机系。请选择相机坐标方法进行比较。');return;}
    if(this.method.kind!=='reference'&&this.referenceGeometry){v.addComparisonLayer('gt',this.referenceGeometry,'GT');add('gt','GT');}
    if(this.method.kind==='reference'){
      const source=state.sequence.methods.find(m=>m.kind!=='reference'&&m.view===this.method.view&&!m.id.endsWith('_world'));if(source){const ego=await loadProjectionGeometry(source.geometry,state.controller?.signal);if(this.disposed||token!==this.loadId)return;v.addComparisonLayer('ego',ego,'EgoFound3R');add('ego','EgoFound3R');}
    }
    if(this.legacyEntry){try{const legacy=await loadLegacy(this.legacyEntry,state.controller?.signal);if(this.disposed||token!==this.loadId)return;for(const [key,label] of [['hggt_native','HGGT 原生'],['hggt_adapted','HGGT 适配'],['hawor','HaWoR']])if(legacy[key]){v.addComparisonLayer(key,legacy[key],label);add(key,label);}}catch(e){if(e.name!=='AbortError')setText(this.q('.comparison-scope'),`昨日几何未加载：${e.message}`);return;}}
    setText(this.q('.comparison-scope'),'勾选可在同一相机坐标叠加真实 21 关节；颜色区分方法。无配准、缩放或 GT 拟合。HaWoR 补全沿用昨日标记；原站的示意壳与完整交互保留在下方原材料中。');v.update(this.displayFrame??state.frame);
  }
  mediaFrame(time=this.video.currentTime){return frameFromMediaTime(time,state.sequence.fps||30,state.sequence.frames,this.method?.video_offset_s||0);}
  startVideoFrames(){
    if(!this.video.requestVideoFrameCallback)return;
    const callback=(now,metadata)=>{
      if(this.disposed)return;
      this.presentedMediaTime=metadata.mediaTime;
      const frame=this.mediaFrame(metadata.mediaTime);
      this.el.dataset.mediaFrame=frame;
      if(state.playing){
        if(state.panels[0]===this)updateTimeline(frame);
        this.present(frame);
      }else if(!this.video.seeking)this.present(frame);
      this.videoCallback=this.video.requestVideoFrameCallback(callback);
    };
    this.videoCallback=this.video.requestVideoFrameCallback(callback);
  }
  seek(frame){
    this.seekArchive(frame);
    if(!this.method||!this.video.src||this.video.readyState<1)return;
    const t=(frame+.05)/(state.sequence.fps||30)+(this.method.video_offset_s||0);
    const target=Math.max(0,Math.min(this.video.duration-.001,t));
    if(Math.abs(this.video.currentTime-target)>.1/(state.sequence.fps||30))this.video.currentTime=target;
  }
  present(frame){
    if(!this.method)return;
    this.displayFrame=frame;this.el.dataset.frame=frame;
    // Use the actual presented video frame, not a free-running wall clock.
    const now=performance.now();
    const visible=this.q('.scene-panel').getBoundingClientRect();
    const onScreen=visible.bottom>0&&visible.top<innerHeight;
    if(!state.playing||(onScreen&&(!this.viewer?.isSoftware||now-(this.lastSceneDraw||0)>=100))){this.viewer?.update(frame);this.lastSceneDraw=now;}
    this.drawOverlay();this.drawProjections();
    if(!state.playing||now-(this.lastMetricDraw||0)>=150){this.drawMetrics(frame);this.drawChart();this.updateLabels(frame);this.lastMetricDraw=now;}
  }
  update(frame){
    if(!this.method)return;
    if(!state.playing)this.seek(frame);
    this.present(frame);
    this.updateLabels(frame);
  }
  updateLabels(frame){
    const g=this.geometry,f=this.geometry?geometryFrame(this.geometry,frame):frame,opts=settings();
    let text=this.method.kind==='reference'?['GT 参考 · 非模型预测']:[];if(g){let count=0;for(let h=0;h<g.hand_count;h++)if(!g.valid||g.valid[f*g.hand_count+h]>.5)count++;text.push(`严格有效 ${count}/${g.hand_count} 手`);if(g.fill_mask?.[f*g.hand_count]||g.fill_mask?.[f*g.hand_count+1])text.push(opts.filled?'填补：半透明显示':'填补：已隐藏');if(!g.scene?.points&&this.method.kind!=='reference')text.push('未导出场景点云');if(opts.color==='contact'&&!g.contact)text.push('缺少接触概率');if(opts.color==='visibility'&&!g.visibility)text.push('缺少可见概率');}
    if(g?.vertices&&g._handLayout?.missingFaceSlots.length)text.push(`手槽 ${g._handLayout.missingFaceSlots.join('/')} 未提供明确面片拓扑，显示顶点`);
    if(this.viewer?.isSoftware&&g?.vertices)text.push('软件视图：真实面片线框');
    if(this.viewer?.renderer.sampled)text.push('已按显示预算抽样点或线');
    setText(this.q('.frame-state'),text.join(' · '));
    const heat=this.q('.heat-legend');heat.hidden=opts.color==='hand'||!g?.[opts.color];
    this.qAll('.scene-legend>span:not(.heat-legend)').forEach((el,i)=>el.hidden=!heat.hidden||(g&&i>=g.hand_count));
    if(!heat.hidden)setText(heat.querySelector('span'),opts.color==='contact'?'接触概率 0 → 1':'可见概率 0 → 1');
    let caption=[];if(g?.observed){const n=Array.from({length:g.hand_count},(_,h)=>g.observed[f*g.hand_count+h]>.5).filter(Boolean).length;caption.push(n?`${n} 只手：网络输出帧`:'插值 / 填补帧');}else if(g?.network_anchor){caption.push(g.network_anchor[f]>.5?'网络采样帧':'时间插值帧');}if(g?.scene?.frame_indices?.length){const a=g.scene.frame_indices.reduce((x,y)=>Math.abs(y-frame)<Math.abs(x-frame)?y:x);caption.push(`场景 G 锚帧 ${a}`);}if(g?.fill_mask){const n=Array.from({length:g.hand_count},(_,h)=>g.fill_mask[f*g.hand_count+h]>.5).filter(Boolean).length;if(n)caption.push(`${n} 只手：官方填补结果`);}setText(this.q('.video-caption'),caption.join(' · '));
  }
  drawProjections(){
    if(!this.method)return;
    const frame=this.displayFrame??state.frame;
    this.drawProjectionPane(this.q('.projection-pred'),this.projectionGeometry,frame,this.method.kind==='reference');
    if(this.method.kind!=='reference'&&!this.q('.projection-gt').hidden)this.drawProjectionPane(this.q('.projection-gt'),this.referenceGeometry,frame,true);
  }
  drawProjectionPane(pane,g,frame,isGT){
    const canvas=pane.querySelector('canvas'),w=pane.clientWidth,h=pane.clientHeight;
    if(!w||!h)return;
    const dpr=Math.min(devicePixelRatio,1.5),W=Math.round(w*dpr),H=Math.round(h*dpr);
    if(canvas.width!==W||canvas.height!==H){canvas.width=W;canvas.height=H;}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.globalAlpha=1;ctx.setLineDash([]);ctx.fillStyle='#101b29';ctx.fillRect(0,0,w,h);
    const vw=this.video.videoWidth||this.rgb?.width||640,vh=this.video.videoHeight||this.rgb?.height||480;
    const scale=Math.min(w/vw,h/vh),ox=(w-vw*scale)/2,oy=(h-vh*scale)/2;
    if(this.video.readyState>=2&&!this.video.seeking)ctx.drawImage(this.video,ox,oy,vw*scale,vh*scale);
    // During an explicit seek do not paint mismatched geometry over the old video.
    if(this.video.seeking){setText(pane.querySelector('.projection-status'),'正在定位视频帧…');return;}
    setText(pane.querySelector('.projection-frame'),`帧 ${frame}`);pane.dataset.frame=frame;
    if(!g){setText(pane.querySelector('.projection-status'),isGT?'GT 正在加载，或该视角没有 GT':'正在加载投影几何…');return;}
    const f=geometryFrame(g,frame),opts=settings(),mask=!isGT&&opts.filled&&g.display_valid?g.display_valid:g.valid,K=frameMatrix(g.projection_K,f);
    const iw=g.projection_image_size_hw?.[1]||vw,ih=g.projection_image_size_hw?.[0]||vh,sx=vw/iw,sy=vh/ih;
    let visibleHands=0,filledHands=0,projected=0;
    ctx.save();ctx.beginPath();ctx.rect(ox,oy,vw*scale,vh*scale);ctx.clip();ctx.translate(ox,oy);ctx.scale(scale*sx,scale*sy);
    for(let hand=0;hand<g.hand_count;hand++){
      if(mask&&!mask[f*g.hand_count+hand])continue;
      visibleHands++;const filled=!isGT&&!!g.fill_mask?.[f*g.hand_count+hand];if(filled)filledHands++;
      const color=COLORS[g._handLayout?.colorIndices[hand]??hand%2];ctx.fillStyle=color;ctx.strokeStyle=color;
      const faces=g._handLayout?.faces[hand];
      if(!isGT&&opts.hands&&g.vertices&&faces&&K&&g.coordinate_space!=='world'){
        const cacheKey=`${g._base}|${f}|${hand}`;
        this.meshPaths=this.meshPaths||new Map();let cached=this.meshPaths.get(cacheKey);
        if(!cached){
          const path=new Path2D(),base=(f*g.hand_count+hand)*g.vertex_count*3,points=[];
          for(let v=0;v<g.vertex_count;v++){const i=base+v*3;points.push(projectPoint(g.vertices[i],g.vertices[i+1],g.vertices[i+2],K));}
          let triangles=0;
          for(let i=0;i<faces.length;i+=3){let a=points[faces[i]],b=points[faces[i+1]],c=points[faces[i+2]];if(!a||!b||!c)continue;
            // Give every triangle the same screen winding: one union fill avoids
            // thousands of alpha-overdraw calls and preserves the true silhouette.
            if((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])<0)[b,c]=[c,b];
            path.moveTo(...a);path.lineTo(...b);path.lineTo(...c);path.closePath();triangles++;
          }
          cached={path,triangles};this.meshPaths.set(cacheKey,cached);if(this.meshPaths.size>16)this.meshPaths.delete(this.meshPaths.keys().next().value);
        }
        ctx.globalAlpha=filled?.18:.40;ctx.fill(cached.path);projected+=cached.triangles;
        if(opts.wireframe){ctx.lineWidth=.45;ctx.globalAlpha=filled?.15:.30;ctx.stroke(cached.path);}
      }
      const coords=jointPixels(g,f,hand,iw,ih);
      if(isGT||opts.overlay){
        ctx.globalAlpha=filled?.45:1;ctx.lineWidth=2/Math.max(.5,scale);ctx.lineCap='round';ctx.setLineDash(filled?[6,4]:[]);
        ctx.beginPath();for(const [a,b] of (g.joint_edges||EDGES)){if(!coords[a]||!coords[b])continue;ctx.moveTo(...coords[a]);ctx.lineTo(...coords[b]);projected++;}ctx.stroke();
        ctx.setLineDash([]);for(const p of coords){if(!p)continue;ctx.beginPath();ctx.arc(...p,2.4/Math.max(.5,scale),0,Math.PI*2);ctx.fill();}
      }
    }
    ctx.restore();ctx.globalAlpha=1;
    let label=isGT?'GT：真实关节标注；未提供 GT 网格':'预测：真实网格 + 相机 K；不对齐 GT';
    if(!visibleHands)label=isGT?'本帧无有效 GT 标注':'本帧无严格有效预测'+(g.fill_mask?.slice(f*g.hand_count,(f+1)*g.hand_count).some(Boolean)?' · 填补已隐藏':'');
    else if(!projected)label='手部未投影到画面内，或投影显示已关闭';
    else if(filledHands)label+=` · ${filledHands} 只填补手半透明`;
    pane.dataset.validHands=visibleHands;pane.dataset.projected=projected;pane.dataset.filledHands=filledHands;
    setText(pane.querySelector('.projection-status'),label);
  }
  qAll(s){return this.el.querySelectorAll(s);}
  drawOverlay(){
    const canvas=this.canvas,box=this.q('.video-panel'),w=box.clientWidth,h=box.clientHeight,dpr=Math.min(devicePixelRatio,2);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const g=this.geometry;if(!g||!$('overlay').checked||this.bakedOverlay||!w)return;
    const vw=this.video.videoWidth||this.rgb?.width||g.image_size_hw?.[1]||640,vh=this.video.videoHeight||this.rgb?.height||g.image_size_hw?.[0]||480,scale=Math.min(w/vw,h/vh),ox=(w-vw*scale)/2,oy=(h-vh*scale)/2;
    const f=geometryFrame(g,this.displayFrame??state.frame),H=g.hand_count,J=g.joint_count,K=flatten(g.projection_K||this.rgb?.K||[]),projectable=g.joints2d||(g.joints&&g.projection_space==='camera'&&K.length===9);
    if(!projectable)return;
    ctx.lineWidth=1.6;ctx.lineCap='round';
    for(let hand=0;hand<H;hand++){
      const mask=settings().filled&&g.display_valid?g.display_valid:g.valid;if(mask&&mask[f*H+hand]<.5)continue;ctx.globalAlpha=g.fill_mask?.[f*H+hand]?.42:1;ctx.setLineDash(g.fill_mask?.[f*H+hand]?[4,3]:[]);const coords=[];
      for(const p of jointPixels(g,f,hand,vw,vh))coords.push(p?[p[0]*scale+ox,p[1]*scale+oy]:null);
      const color=COLORS[g._handLayout?.colorIndices[hand]??hand%2];ctx.strokeStyle=color;ctx.fillStyle=color;ctx.shadowColor='#000';ctx.shadowBlur=2;
      for(const [a,b] of (g.joint_edges||EDGES)){if(!coords[a]||!coords[b])continue;ctx.beginPath();ctx.moveTo(...coords[a]);ctx.lineTo(...coords[b]);ctx.stroke();}
      for(const p of coords){if(!p)continue;ctx.beginPath();ctx.arc(...p,2,0,Math.PI*2);ctx.fill();}
    }
  }
  drawMetrics(frame){
    const target=this.q('.frame-metrics');target.replaceChildren();const g=this.geometry,f=this.geometry?geometryFrame(this.geometry,frame):frame;
    const add=(label,value,unit='')=>{const d=node('div','frame-metric'),l=node('span','',label),v=node('strong','',value);if(unit)v.append(node('small','',unit));d.append(l,v);target.append(d);};
    if(!g){add('几何状态',this.method.status==='complete'?'未提供':this.method.status==='blocked'?'待权重授权':'等待结果');return;}
    if(g.series?.length){for(const s of g.series.slice(0,3))add(s.label,fmt(s.values[f],s.digits??2),s.unit||'');}
    if(g.contact){let sum=0,n=0;for(let h=0;h<g.hand_count;h++){if(g.valid&&!g.valid[f*g.hand_count+h])continue;const start=(f*g.hand_count+h)*g.vertex_count;for(let i=0;i<g.vertex_count;i++){const v=g.contact[start+i];if(finite(v)){sum+=v;n++;}}}add('有效顶点平均接触概率',n?fmt(sum/n,3):'—');}
    if(!target.childElementCount){add('当前帧',String(frame));add('几何单位',g.units||'m');}
    if(g.observed){let num=0;for(let h=0;h<g.hand_count;h++)num+=g.observed[f*g.hand_count+h]>.5;add('网络输出 / 有效手',`${num} / ${g.hand_count}`);}
  }
  drawChart(){
    const g=this.geometry,chart=g?.series?.find(s=>s.values?.length>1),holder=this.q('.chart-container');holder.classList.toggle('has-data',!!chart);if(!chart)return;
    const canvas=this.q('.series-chart'),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2),w=rect.width,h=rect.height;if(!w||!h)return;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const vals=chart.values,good=Array.from(vals).filter(finite);if(!good.length)return;
    let lo=Math.min(...good),hi=Math.max(...good);if(hi-lo<1e-8){lo-=.5;hi+=.5;}const x=i=>4+i/(vals.length-1)*(w-8),y=v=>h-5-(v-lo)/(hi-lo)*(h-10);
    ctx.strokeStyle='#e5eaf1';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,h-4);ctx.lineTo(w,h-4);ctx.stroke();
    ctx.strokeStyle=ACCENTS[this.index%3];ctx.lineWidth=1.5;ctx.beginPath();let pen=false;for(let i=0;i<vals.length;i++){if(!finite(vals[i])){pen=false;continue;}if(!pen)ctx.moveTo(x(i),y(vals[i]));else ctx.lineTo(x(i),y(vals[i]));pen=true;}ctx.stroke();
    const f=geometryFrame(g,this.displayFrame??state.frame);ctx.strokeStyle='#283e5b';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(f),1);ctx.lineTo(x(f),h);ctx.stroke();if(finite(vals[f])){ctx.fillStyle=ACCENTS[this.index%3];ctx.beginPath();ctx.arc(x(f),y(vals[f]),3,0,Math.PI*2);ctx.fill();}
    setText(this.q('.chart-label'),`${chart.label}${chart.unit?' · '+chart.unit:''} · 全片段变化`);
  }
  dispose(){this.disposed=true;this.loadId++;this.resizeObserver.disconnect();if(this.videoCallback)this.video.cancelVideoFrameCallback(this.videoCallback);this.video.pause();this.archiveVideo.pause();this.archiveVideo.removeAttribute('src');this.archiveVideo.load();this.video.removeAttribute('src');this.video.load();this.viewer?.dispose();this.el.remove();}
}

function frameTime(f){const times=state.sequence?.timestamps||state.sequence?.timestamps_s;if(times?.length)return times[f]-times[0];return f/(state.sequence?.fps||30);}
function updateTimeline(f){
  if(!state.sequence)return;state.frame=Math.max(0,Math.min(state.sequence.frames-1,Math.round(f)));state.clock=frameTime(state.frame);
  $('scrubber').value=state.frame;$('frame-number').value=state.frame;setText($('time-label'),`${fmt(state.clock)} s`);
  setText($('source-frame-label'),state.sequence.frame_ids?.length?`源帧 ${state.sequence.frame_ids[state.frame]}`:`第 ${state.frame+1} / ${state.sequence.frames} 帧`);
}
function setFrame(f){updateTimeline(f);for(const p of state.panels)p.update(state.frame);}
function setPlaying(value){
  const token=state.playToken=(state.playToken||0)+1,wasPlaying=state.playing;
  state.playing=value;setText($('play-icon'),value?'Ⅱ':'▶');setText($('play-text'),value?'暂停':'播放');$('play').setAttribute('aria-label',value?'暂停':'播放');
  if(!value){for(const p of state.panels){p.video.pause();p.archiveVideo.pause();}if(wasPlaying&&state.panels[0]){updateTimeline(state.panels[0].mediaFrame(state.panels[0].presentedMediaTime??state.panels[0].video.currentTime));for(const p of state.panels)p.update(state.frame);}return;}
  if(!state.panels.length)return;
  if(state.frame>=state.sequence.frames-1)setFrame(0);
  const videos=state.panels.filter(p=>p.video.src);
  for(const p of videos){p.seek(state.frame);p.video.playbackRate=Number($('speed').value);p.archiveVideo.playbackRate=p.video.playbackRate*(state.sequence.fps||30)/(p.archiveFPS||state.sequence.fps);}
  Promise.all(videos.flatMap(p=>[p.video.play(),...(p.activeVideoTab!=='input'?[p.archiveVideo.play()]:[])])).catch(e=>{if(state.playToken===token){setPlaying(false);setText($('load-status'),'视频未能开始播放，请等待加载后重试。');}});
}
function tick(now){
  if(state.playing&&state.panels.length){
    const master=state.panels[0],time=master.video.currentTime;
    // Older browsers without rVFC still use the media clock, never repeated seeking.
    for(const p of state.panels)if(!p.video.requestVideoFrameCallback&&!p.video.seeking){const f=p.mediaFrame();if(p===master)updateTimeline(f);if(f!==p.displayFrame)p.present(f);}
    if(now-(state.lastSync||0)>300){
      state.lastSync=now;
      for(const p of state.panels)if(p.activeVideoTab!=='input'&&p.archiveVideo.readyState>=2&&!p.archiveVideo.seeking){const target=p.video.currentTime*(state.sequence.fps||30)/(p.archiveFPS||30);if(Math.abs(p.archiveVideo.currentTime-target)>.12)p.archiveVideo.currentTime=target;}
      for(const p of state.panels.slice(1)){
        const delta=(p.video.currentTime-(p.method.video_offset_s||0))-(time-(master.method.video_offset_s||0));
        if(!p.video.seeking&&p.video.readyState>=2&&Math.abs(delta)>.12){p.video.currentTime=Math.max(0,time+(p.method.video_offset_s||0)-(master.method.video_offset_s||0));}
      }
    }
  }
  requestAnimationFrame(tick);
}

function renderMetrics(){
  const seq=state.sequence,target=$('metrics');target.replaceChildren();const methods=(seq.methods||[]).filter(m=>m.metrics?.length);
  const cols=[];for(const m of methods)for(const metric of m.metrics||[]){const key=metric.id||metric.label;if(!cols.find(c=>c.key===key))cols.push({...metric,key});}
  setText($('metric-scope'),seq.metric_scope||'同一片段 · 以注明的有效帧与单位为准');
  if(!cols.length){target.append(node('p','empty-inline',seq.metrics_missing_reason||'尚未提供该片段的定量评估。'));return;}
  const table=document.createElement('table'),head=document.createElement('thead'),hr=document.createElement('tr');hr.append(node('th','','方法'));for(const c of cols)hr.append(node('th','',`${c.label}${c.unit?' ('+c.unit+')':''}${c.direction==='lower'?' ↓':c.direction==='higher'?' ↑':''}`));head.append(hr);table.append(head);
  const body=document.createElement('tbody');for(const m of methods){const tr=document.createElement('tr');tr.append(node('td','method-name-cell',m.label||m.id));for(const c of cols){const met=(m.metrics||[]).find(v=>(v.id||v.label)===c.key),value=met?.value;tr.append(node('td',value==null?'metric-missing':'',value==null?'—':typeof value==='number'?fmt(value,met.digits??2):String(value)));}body.append(tr);}table.append(body);target.append(table);
}
function renderDescription(){
  const seq=state.sequence,man=state.manifest;$('protocol-items').replaceChildren();
  for(const p of [...(man.protocol||[]),...(seq?.protocol||[])]){const el=node('div','protocol-item');el.append(node('span','key',p.label),node('span','value',p.value));$('protocol-items').append(el);}
  setText($('sequence-description'),seq?.description||'');$('sequence-description').hidden=!seq?.description;
  $('result-notes').replaceChildren();for(const note of [...(seq?.notes||[]),...(man.notes||[])])$('result-notes').append(node('p','',typeof note==='string'?note:note.text));
  if(!$('result-notes').childElementCount)$('result-notes').append(node('p','','左图与右图随同一时间轴切换。三维窗口可以联动旋转，便于观察手部位置、姿态与场景的关系。'));
  $('downloads').replaceChildren();for(const d of [...(seq?.downloads||[]),...(man.downloads||[])]){const a=node('a','',d.label||'下载');a.href=urlFor(d.url);a.target='_blank';a.rel='noopener';$('downloads').append(a);}
}
async function selectSequence(id){
  state.loadId++;state.controller?.abort();state.controller=new AbortController();setPlaying(false);for(const p of state.panels)p.dispose();state.panels=[];assetCache.clear();legacyCache.clear();
  state.sequence=state.manifest.sequences.find(s=>s.id===id);if(!state.sequence)return;
  const seq=state.sequence;$('sequence').value=id;const frames=Math.max(1,seq.frames||1);seq.frames=frames;$('scrubber').max=frames-1;$('frame-number').max=frames-1;setText($('duration-label'),`${fmt(frameTime(frames-1))} s`);
  $('sequence-meta').replaceChildren();for(const t of [seq.dataset,`${frames} 帧`,`${seq.fps||30} fps`,seq.input_label||''])if(t)$('sequence-meta').append(node('span','chip',t));
  renderDescription();renderMetrics();renderLegacyMaterials();notice(seq.notice||state.manifest.notice||'');$('load-status').classList.remove('error');setText($('load-status'),'正在读取片段资源…');
  renderPanels();setFrame(0);saveQuery();
}
function renderPanels(){
  setPlaying(false);
  const selected=state.panels.map(p=>p.method?.id);for(const p of state.panels)p.dispose();state.panels=[];
  const methods=state.sequence?.methods||[],n=Math.min(Number($('columns').value),Math.max(1,methods.length));$('comparison').classList.toggle('three-columns',n===3);
  if(!methods.length){$('comparison').append(node('p','empty-inline','这个片段尚无可显示的方法结果。'));return;}
  const params=new URLSearchParams(location.search),ids=params.get('methods')?.split(',')||[];
  for(let i=0;i<n;i++){const candidate=selected[i]||ids[i];const id=methods.some(m=>m.id===candidate)?candidate:methods[Math.min(i,methods.length-1)].id;const p=new MethodPanel(i,id);state.panels.push(p);}
  deduplicateGTProjections();
}
function saveQuery(){if(!state.sequence)return;const q=new URLSearchParams();q.set('seq',state.sequence.id);if(state.panels.length)q.set('methods',state.panels.map(p=>p.method?.id).filter(Boolean).join(','));history.replaceState(null,'',`${location.pathname}?${q}`);}

function wireUI(){
  setText($('color-mode').querySelector('option[value="hand"]'),'左右手');
  $('speed').addEventListener('change',()=>{for(const p of state.panels){p.video.playbackRate=Number($('speed').value);p.archiveVideo.playbackRate=p.video.playbackRate*state.sequence.fps/(p.archiveFPS||state.sequence.fps);}});
  $('sequence').addEventListener('change',()=>selectSequence($('sequence').value));$('columns').addEventListener('change',()=>{renderPanels();saveQuery();});
  $('play').addEventListener('click',()=>setPlaying(!state.playing));$('previous').addEventListener('click',()=>{setPlaying(false);setFrame(state.frame-1);});$('next').addEventListener('click',()=>{setPlaying(false);setFrame(state.frame+1);});
  $('scrubber').addEventListener('input',()=>{setPlaying(false);setFrame(Number($('scrubber').value));});$('frame-number').addEventListener('change',()=>{setPlaying(false);setFrame(Number($('frame-number').value));});
  $('frame-number').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();setPlaying(false);setFrame(Number(e.currentTarget.value));}});
  for(const id of ['overlay','scene','hands','joints','trajectory','wireframe','color-mode','filled'])$(id).addEventListener('change',()=>state.panels.forEach(p=>p.update(state.frame)));
  $('linked').addEventListener('change',()=>{if($('linked').checked){const v=state.panels.find(p=>p.viewer?.data)?.viewer;if(v)syncCameras(v);}});
  $('reset-camera').addEventListener('click',()=>{for(const p of state.panels)p.viewer?.fit();if($('linked').checked){const v=state.panels.find(p=>p.viewer?.data)?.viewer;if(v)syncCameras(v);}});
  $('details-button').addEventListener('click',()=>{const isHidden=$('protocol').hidden;$('protocol').hidden=!isHidden;$('details-button').setAttribute('aria-expanded',String(isHidden));});
  window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement?.tagName))return;if(e.code==='Space'){e.preventDefault();setPlaying(!state.playing);}else if(e.code==='ArrowLeft'){e.preventDefault();setPlaying(false);setFrame(state.frame-1);}else if(e.code==='ArrowRight'){e.preventDefault();setPlaying(false);setFrame(state.frame+1);}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false);});
}
async function main(){
  wireUI();requestAnimationFrame(tick);
  try{
    const manifestURL=new URL('manifest.json',document.baseURI);manifestURL.search=new URL(import.meta.url).search;
    state.legacyManifest=await fetchJSON(urlFor('legacy-hggt.json'));
    state.manifest=await fetchJSON(manifestURL.href);const man=state.manifest;
    if(man.suite_summary){
      const table=node('table'),head=node('tr');
      for(const title of ['数据 / 输入','RGB 帧','有效手帧 / GT 手帧','覆盖率','绝对 MPJPE mm','去腕 MPJPE mm','PA MPJPE mm','PCK50 · 全 GT'])head.append(node('th','',title));
      const thead=node('thead');thead.append(head);table.append(thead);const body=node('tbody');
      for(const [dataset,result] of Object.entries(man.suite_summary)){
        const c=result.coverage,m=result.metrics,tr=node('tr');
        const values=[dataset+(dataset==='DexYCB'?' · 双 RGB 独立':' · 单 RGB 补充'),result.rgb_frames,`${c.evaluated_hand_frames} / ${c.gt_hand_frames}`,fmt(100*c.fraction,1)+'%',fmt(m.mpjpe_raw_mm),fmt(m.mpjpe_root_relative_mm),fmt(m.mpjpe_pa_mm),fmt(100*result.pck50_all_gt,1)+'%'];
        for(const v of values)tr.append(node('td','',String(v)));body.append(tr);
      }table.append(body);$('suite-metrics').append(table);
    }
    document.title=man.title||'EgoFound3R · 复现对比';setText($('page-title'),man.title||'EgoFound3R · 复现对比');setText($('page-subtitle'),man.subtitle||'');
    if(man.status&&man.status!=='complete'){$('run-state').hidden=false;setText($('run-state'),man.status_label||({blocked:'待权重授权 · 未完成推理',awaiting_results:'待导入推理结果',partial:'部分推理结果'})[man.status]||man.status);}
    for(const s of man.sources||[]){const a=node('a','',s.label);a.href=s.url;a.target='_blank';a.rel='noopener';$('source-links').append(a);}
    setText($('footer-provenance'),man.provenance||`仅展示已导出的真实推理结果${man.generated_at?' · '+man.generated_at:''}`);
    $('sequence').replaceChildren();for(const seq of man.sequences||[]){const o=node('option','',seq.label||seq.id);o.value=seq.id;$('sequence').append(o);}
    if(!man.sequences?.length){$('sequence').append(node('option','','尚未导入序列'));notice(man.notice||'尚未导入真实推理结果。');renderDescription();for(const id of ['play','previous','next','scrubber','frame-number','columns'])$(id).disabled=true;setText($('load-status'),'等待真实结果');return;}
    const params=new URLSearchParams(location.search);if(params.has('methods'))$('columns').value=params.get('methods').split(',').length>=3?'3':'2';else if(Number(man.default_columns)===3)$('columns').value='3';
    const requested=params.get('seq'),id=man.sequences.some(s=>s.id===requested)?requested:man.sequences[0].id;await selectSequence(id);
  }catch(e){notice(`结果索引读取失败：${e.message}`,true);setText($('load-status'),'请检查 manifest.json 与资源路径');$('load-status').classList.add('error');}
}
main();

