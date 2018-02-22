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

return {fetchBytes};

})();
