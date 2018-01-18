flux = (function () {

function blobAsArrayBuffer(blob) {
  return new Promise((resolve, reject) => {
    let reader = new FileReader();
    reader.addEventListener("loadend", function() {
      resolve(reader.result);
    });
    reader.readAsArrayBuffer(blob);
  });
}

async function fetchBytes(url) {
  let resp = await fetch(url);
  if (!resp.ok) throw(resp);
  let blob = await resp.blob();
  let buf  = await blobAsArrayBuffer(blob);
  return new DataView(buf);
}

let TAGS = [
  'any', 'vector', 'array',
  'float16', 'float32', 'float64',
  'int16', 'int32', 'int64'
]

function readTag(data, idx) {
  let tag = TAGS[data.getUint8(idx, true)-1];
  idx += 1;
  if (tag == 'vector' || tag == 'array') {
    let eltype;
    [eltype, idx] = readTag(data, idx);
    return [[tag, eltype], idx];
  } else {
    return [tag, idx];
  }
}

function _readVector(data, eltype, len, idx) {
  if (eltype == 'any') {
    let vec = [];
    for (var i = 0; i < len; i++) {
      let x;
      [x, idx] = readData(data, idx);
      vec.push(x);
    }
    return [vec, idx];
  } else if (eltype == 'float32') {
    let vec = new Float32Array(len);
    for (var i = 0; i < vec.length; i++) {
      vec[i] = data.getFloat32(idx+i*4, true);
    }
    return [vec, idx + len*4];
  } else if (eltype == 'float64') {
    let vec = new Float64Array(len);
    for (var i = 0; i < vec.length; i++) {
      vec[i] = data.getFloat64(idx+i*8, true);
    }
    return [vec, idx + len*8];
  } else {
    throw(`Unsupported element type ${eltype}.`)
  }
}

function readVector(data, eltype, idx) {
  let len = data.getUint32(idx, true);
  return _readVector(data, eltype, len, idx+4);
}

function readData(data, idx) {
  let tag;
  [tag, idx] = readTag(data, idx);
  if (tag == 'float32') {
    return [data.getFloat32(idx, true), idx+4];
  } else if (tag[0] == 'vector') {
    return readVector(data, tag[1], idx);
  } else {
    error(`Unsupport type ${tag}`);
  }
}

function readFile(data) {
  let version = data.getUint8(data, 0, true);
  if (version != 1) throw `Blob version ${version} not supported.`;
  return readData(data, 1)[0];
}

async function fetchBlob(url) {
  data = await fetchBytes(url);
  return readFile(data);
}

return {fetchBlob};

})();
