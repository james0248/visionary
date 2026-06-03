export function findExport(manifest, name) {
  const entry = manifest.exports.find((item) => item.name === name);
  if (!entry) throw new Error(`Missing export ${name}`);
  return entry;
}

export function findFirstExport(manifest, names) {
  for (const name of names.filter(Boolean)) {
    const entry = manifest.exports.find((item) => item.name === name);
    if (entry) return entry;
  }
  throw new Error(`Missing exports: ${names.filter(Boolean).join(', ')}`);
}

export function formatShape(shape) {
  return `[${shape.join(',')}]`;
}

export function sameShape(actual, expected) {
  return (
    Array.isArray(actual) &&
    Array.isArray(expected) &&
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}
