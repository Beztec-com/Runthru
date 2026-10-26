// The provider layer: which key belongs to which provider, which provider wins when more than
// one is set, and which model a connected key resolves to.
//
// Nothing here talks to a provider. The one case that could — resolving a model with nothing
// pinned — lands on the provider's default model whether the model list succeeds, is rejected or
// can't be reached at all, so it is asserted on that outcome rather than on the network.
//
// The keys below are FAKE fixtures: correctly shaped, obviously not real, used only to exercise
// the shape checks. Real keys live in .env and are read in exactly one place, server/llm.js.

import { test, describe, before, beforeEach, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  PROVIDERS,
  detectProvider,
  activeProvider,
  hasKey,
  aiStatus,
  resetModelCache,
  resolveModel,
  callModel,
} from '../server/llm.js';

const VARS = [
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'DEEPSEEK_API_KEY',
  'AI_PROVIDER',
  'OPENAI_MODEL',
  'ANTHROPIC_MODEL',
  'DEEPSEEK_MODEL',
];
const saved = new Map(VARS.map((k) => [k, process.env[k]]));

// Shaped like the real thing — including the DeepSeek key, which is `sk-` plus 32 characters, the
// one shape that has to be told apart from OpenAI's — but a run of one character, so nobody
// mistakes a fixture for a credential.
const FAKE_OPENAI_KEY = `sk-proj-${'a'.repeat(120)}`;
const FAKE_ANTHROPIC_KEY = `sk-ant-api03-${'b'.repeat(40)}`;
const FAKE_DEEPSEEK_KEY = `sk-${'d'.repeat(32)}`;

function clearVars() {
  for (const k of VARS) delete process.env[k];
  resetModelCache();
}

before(clearVars);
beforeEach(clearVars);
after(() => {
  for (const [k, v] of saved) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
});

describe('detectProvider', () => {
  test('recognises a DeepSeek key by its shape', () => {
    assert.equal(detectProvider(FAKE_DEEPSEEK_KEY), 'deepseek');
  });

  test('still recognises OpenAI and Anthropic keys', () => {
    assert.equal(detectProvider(FAKE_OPENAI_KEY), 'openai');
    assert.equal(detectProvider(`sk-${'c'.repeat(48)}`), 'openai');
    assert.equal(detectProvider(FAKE_ANTHROPIC_KEY), 'anthropic');
  });

  test('does not mistake a 31-character sk- key for a DeepSeek key', () => {
    assert.equal(detectProvider(`sk-${'c'.repeat(31)}`), 'openai');
  });

  test('ignores whitespace and rejects anything else', () => {
    assert.equal(detectProvider(`  ${FAKE_DEEPSEEK_KEY}\n`), 'deepseek');
    assert.equal(detectProvider(''), null);
    assert.equal(detectProvider('not-a-key'), null);
  });
});

describe('PROVIDERS', () => {
  test('DeepSeek is registered with its own env vars and key page', () => {
    assert.equal(PROVIDERS.deepseek.label, 'DeepSeek');
    assert.equal(PROVIDERS.deepseek.keyEnv, 'DEEPSEEK_API_KEY');
    assert.equal(PROVIDERS.deepseek.modelEnv, 'DEEPSEEK_MODEL');
    assert.match(PROVIDERS.deepseek.keysUrl, /^https:\/\//);
  });
});

describe('activeProvider', () => {
  test('nothing connected', () => {
    assert.equal(activeProvider(), null);
    assert.equal(hasKey(), false);
  });

  test('a DeepSeek key alone is enough', () => {
    process.env.DEEPSEEK_API_KEY = FAKE_DEEPSEEK_KEY;
    assert.equal(activeProvider(), 'deepseek');
    assert.equal(hasKey(), true);
  });

  test('the last connected key wins, via AI_PROVIDER', () => {
    process.env.OPENAI_API_KEY = FAKE_OPENAI_KEY;
    process.env.DEEPSEEK_API_KEY = FAKE_DEEPSEEK_KEY;

    assert.equal(activeProvider(), 'openai'); // nothing pinned: the tie-break order decides
    process.env.AI_PROVIDER = 'deepseek';
    assert.equal(activeProvider(), 'deepseek');
    process.env.AI_PROVIDER = 'anthropic'; // pinned to a provider with no key: falls back
    assert.equal(activeProvider(), 'openai');
  });

  test('a key that is not DeepSeek-shaped in DEEPSEEK_API_KEY is not used', () => {
    process.env.DEEPSEEK_API_KEY = FAKE_OPENAI_KEY;
    assert.equal(activeProvider(), null);
  });
});

describe('model choice', () => {
  test('a pinned DEEPSEEK_MODEL is used as-is', async () => {
    process.env.DEEPSEEK_API_KEY = FAKE_DEEPSEEK_KEY;
    process.env.DEEPSEEK_MODEL = 'deepseek-v4-pro';
    assert.equal(await resolveModel(), 'deepseek-v4-pro');
  });

  test('with nothing pinned it lands on a DeepSeek model either way', async () => {
    process.env.DEEPSEEK_API_KEY = FAKE_DEEPSEEK_KEY;
    assert.equal(await resolveModel(true), 'deepseek-flash');

    const status = await aiStatus();
    assert.equal(status.configured, true);
    assert.equal(status.provider, 'deepseek');
    assert.equal(status.providerLabel, 'DeepSeek');
    assert.equal(status.model, 'deepseek-flash');
  });

  test('without a key there is no model and no call', async () => {
    assert.equal(await resolveModel(), null);
    assert.deepEqual(await aiStatus(), {
      configured: false,
      provider: null,
      providerLabel: null,
      model: null,
      error: 'No API key connected',
    });
    await assert.rejects(() => callModel({ messages: [{ role: 'user', content: 'hi' }] }), /No API key connected/);
  });
});
