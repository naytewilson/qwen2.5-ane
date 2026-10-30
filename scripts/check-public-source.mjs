// Portable, read-only checks for public research source packaging.
// This is NOT a credential scanner, macOS build, ANE audit, or inference test.
import { execFileSync } from 'node:child_process';
import { lstatSync, readFileSync } from 'node:fs';

const fail = (msg) => { throw new Error(msg); };
const tracked = execFileSync('git', ['ls-files', '-z'], { encoding: 'utf8' })
  .split('\0').filter(Boolean);
if (tracked.length === 0) fail('No tracked source files');
const restricted = [
  /(^|\/)models(\/|$)/i,
  /(^|\/)\.env(?:$|\.)/i,
  /(^|\/)(?:credentials|secrets)(?:\.[^/]*)?$/i,
  /\.(?:mlpackage|mlmodelc)(?:\/|$)/i,
  /\.(?:safetensors|gguf|onnx|pt|pth|ckpt|pkl|pickle|key|pem)$/i,
];
for (const file of tracked) {
  if (restricted.some((re) => re.test(file))) {
    fail('Restricted model, credential, or private-artifact path tracked: ' + file);
  }
  if (lstatSync(file).isSymbolicLink()) {
    fail('Tracked symbolic link requires intentional provenance review: ' + file);
  }
}
const inputs = [
  'tokenizer/config.json',
  'tokenizer/generation_config.json',
  'tokenizer/tokenizer_config.json',
  'tokenizer/tokenizer.json',
  'tokenizer/vocab.json',
];
const parsed = new Map();
for (const file of inputs) {
  if (!tracked.includes(file)) fail('Required tokenizer metadata not tracked: ' + file);
  const json = JSON.parse(readFileSync(file, 'utf8'));
  if (!json || typeof json !== 'object' || Array.isArray(json)) {
    fail('Expected a JSON object: ' + file);
  }
  parsed.set(file, json);
}
if (parsed.get('tokenizer/config.json').model_type !== 'qwen2') {
  fail('Unexpected model_type (expected qwen2)');
}
if (parsed.get('tokenizer/tokenizer.json').model?.type !== 'BPE') {
  fail('Unexpected tokenizer model type (expected BPE)');
}
if (Object.keys(parsed.get('tokenizer/vocab.json')).length < 1000) {
  fail('Unexpectedly small vocabulary');
}
console.log('PASS: ' + inputs.length + ' tokenizer JSON files; ' + tracked.length +
  ' tracked paths checked for known restricted names and symlinks.');
console.log('Scope: source packaging hygiene, not a full secret scan or ANE validation.');
