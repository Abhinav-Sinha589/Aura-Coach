// Browser stub for Node's `fs`.
// face-api.js contains `try { require("fs") } catch {}` for its Node.js build.
// That branch never runs in the browser, but webpack 5 still tries to resolve
// it and CRA turns the resulting warning into a CI build failure.
// jsconfig.json (baseUrl: "src") makes webpack resolve the bare "fs" here.
module.exports = null;
