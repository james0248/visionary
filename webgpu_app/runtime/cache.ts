export function cacheOutputNames(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  const kLayers = outputs.filter((name) => /^k_cache_\d+$/.test(name)).sort();
  const vLayers = outputs.filter((name) => /^v_cache_\d+$/.test(name)).sort();
  const candidateKLayers = outputs.filter((name) => /^candidate_k_cache_\d+$/.test(name)).sort();
  const candidateVLayers = outputs.filter((name) => /^candidate_v_cache_\d+$/.test(name)).sort();
  return {
    k: candidateKLayers.length
      ? candidateKLayers
      : kLayers.length
        ? kLayers
        : outputs.find((name) => name === 'k_cache' || name.endsWith('_k_cache')),
    v: candidateVLayers.length
      ? candidateVLayers
      : vLayers.length
        ? vLayers
        : outputs.find((name) => name === 'v_cache' || name.endsWith('_v_cache')),
    entryK: outputs.find((name) => name === 'candidate_k_entry' || name.endsWith('_k_entry')),
    entryV: outputs.find((name) => name === 'candidate_v_entry' || name.endsWith('_v_entry')),
    length: outputs.find((name) => name === 'cache_length' || name.endsWith('_cache_length')),
  };
}

export function inputCacheNames(spec) {
  const inputs = Object.keys(spec.inputs ?? {});
  const kLayers = inputs.filter((name) => /^k_cache_\d+$/.test(name)).sort();
  const vLayers = inputs.filter((name) => /^v_cache_\d+$/.test(name)).sort();
  return {
    k: kLayers.length ? kLayers : inputs.find((name) => name === 'k_cache' || name.endsWith('_k_cache')),
    v: vLayers.length ? vLayers : inputs.find((name) => name === 'v_cache' || name.endsWith('_v_cache')),
    length: inputs.find((name) => name === 'cache_length' || name.endsWith('_cache_length')),
  };
}

export function applyCacheFeeds(feeds, cache) {
  const next = { ...feeds };
  for (const name of Object.keys(next)) {
    const kLayer = /^k_cache_(\d+)$/.exec(name);
    const vLayer = /^v_cache_(\d+)$/.exec(name);
    if (kLayer && Array.isArray(cache.k)) next[name] = cache.k[Number(kLayer[1])];
    else if (name === 'k_cache' || name.endsWith('_k_cache')) next[name] = cache.k;
    if (vLayer && Array.isArray(cache.v)) next[name] = cache.v[Number(vLayer[1])];
    else if (name === 'v_cache' || name.endsWith('_v_cache')) next[name] = cache.v;
    if (name === 'cache_length' || name.endsWith('_cache_length')) next[name] = cache.length;
  }
  return next;
}
