// Pure calibrated projection. Never fit predictions to the video or to GT.
export function frameMatrix(values, frame) {
  if (!values) return null;
  const k = Array.isArray(values) ? values.flat(Infinity) : values;
  const start = k.length === 9 ? 0 : frame * 9;
  if (start + 9 > k.length) return null;
  return k.slice(start, start + 9);
}
export function projectPoint(x, y, z, k) {
  if (!k || ![x,y,z].every(Number.isFinite) || z <= 1e-6) return null;
  const d=k[6]*x+k[7]*y+k[8]*z;
  if (!Number.isFinite(d) || d<=1e-6) return null;
  const u=(k[0]*x+k[1]*y+k[2]*z)/d,v=(k[3]*x+k[4]*y+k[5]*z)/d;
  return Number.isFinite(u)&&Number.isFinite(v)?[u,v]:null;
}
export function geometryFrame(g, frame) {
  if(g.frame_indices?.length){let best=0;while(best+1<g.frame_indices.length&&Math.abs(g.frame_indices[best+1]-frame)<Math.abs(g.frame_indices[best]-frame))best++;return best;}
  return Math.max(0,Math.min(g.frame_count-1,frame));
}
export function jointPixels(g, frame, hand, width, height) {
  const k=frameMatrix(g.projection_K,frame),result=[];
  for(let j=0;j<g.joint_count;j++){
    const index=(frame*g.hand_count+hand)*g.joint_count+j;
    if(g.joint_valid&&!g.joint_valid[index]){result.push(null);continue;}
    if(g.joints2d){const x=g.joints2d[index*2],y=g.joints2d[index*2+1];result.push(Number.isFinite(x)&&Number.isFinite(y)?[x*(g.joints2d_normalized?width:1),y*(g.joints2d_normalized?height:1)]:null);}
    else result.push(g.joints&&g.coordinate_space!=='world'?projectPoint(...g.joints.subarray(index*3,index*3+3),k):null);
  }
  return result;
}
export function frameFromMediaTime(time,fps,count,offset=0){return Math.max(0,Math.min(count-1,Math.floor((time-offset)*fps+1e-4)));}
