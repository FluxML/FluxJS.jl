flux = (function () {

let Buffer = new BSON().serialize({}).constructor

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
  return new Buffer(buf);
}

async function fetchData(url) {
  let buf = await fetchBytes(url);
  return new BSON().deserialize(buf);
}

// `buf` is a Uint8Array from BSON
function readFloat32(buf) {
  let view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let data = new Float32Array(view.byteLength/4);
  for (i = 0; i < data.length; i++) {
    data[i] = view.getFloat32(i*4, true);
  }
  return data;
}

function toTensor_(spec) {
  let type = spec.type.name;
  type = type[type.length-1];
  if (type == 'Float32') spec.data = readFloat32(spec.data.buffer);
  else throw `Array type ${spec.type.name} not supported.`;
  let array = dl.tensor(spec.data, spec.size.reverse());
  if (spec.size.length > 1) array = array.transpose();
  return array
}

function convertArrays_(data) {
  if (!(typeof data == "object")) return data
  if (data.tag == "array") {
    return toTensor_(data);
  } else {
    for (k of Object.keys(data)) {
      data[k] = convertArrays_(data[k])
    }
  }
  return data;
}

async function fetchBlob(url) {
  data = await fetchData(url);
  return convertArrays_(data)
}

return {fetchData, fetchBlob};

})();
