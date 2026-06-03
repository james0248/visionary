import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { buildBrowserEntrypoints } from './build_browser_entrypoints';

await buildBrowserEntrypoints();

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i], process.argv[i + 1]);
}

const host = args.get('--host') ?? '127.0.0.1';
const port = Number(args.get('--port') ?? 4173);
const root = resolve('.');

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.data': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

function resolveRequestPath(url) {
  const pathname = decodeURIComponent(new URL(url, `http://${host}:${port}`).pathname);
  let relative = normalize(pathname).replace(/^[/\\]+/, '');
  if (relative === 'webgpu_app' || relative.startsWith(`webgpu_app${sep}`)) {
    relative = relative.replace(/^webgpu_app[/\\]?/, '');
  }
  const absolute = resolve(join(root, relative));
  if (absolute !== root && !absolute.startsWith(`${root}${sep}`)) {
    return null;
  }
  return absolute;
}

const server = createServer((request, response) => {
  if (request.url === '/health') {
    response.writeHead(200, {
      'content-type': 'text/plain; charset=utf-8',
      'cache-control': 'no-store, max-age=0',
    });
    response.end('ok');
    return;
  }

  const path = resolveRequestPath(request.url ?? '/');
  if (path === null) {
    response.writeHead(403);
    response.end('Forbidden');
    return;
  }

  try {
    const stat = statSync(path);
    if (!stat.isFile()) {
      response.writeHead(404);
      response.end('Not found');
      return;
    }

    response.writeHead(200, {
      'content-type': contentTypes[extname(path)] ?? 'application/octet-stream',
      'content-length': stat.size,
      'cache-control': 'no-store, max-age=0',
      'cross-origin-opener-policy': 'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
    });
    if (request.method === 'HEAD') {
      response.end();
      return;
    }
    createReadStream(path).pipe(response);
  } catch {
    response.writeHead(404);
    response.end('Not found');
  }
});

server.listen(port, host, () => {
  console.log(`Serving ${root} at http://${host}:${port}`);
});
