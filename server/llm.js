// Model providers.
//
// The studio works with an OpenAI, an Anthropic or a DeepSeek key. Everything above this file
// is written in the OpenAI chat-completions shape — a messages array, `tools` as function
// schemas, `response_format` for JSON, and `choices[0].message` coming back — because that is
// what the drafting, review, chat and autopilot passes were built against.
//
// So this module owns the difference: it picks the provider from whichever key is present,
// asks that key which models it can actually see, sends the request through the client that
// provider needs, and for Anthropic translates the request on the way out and the reply on the
// way back. DeepSeek already speaks the OpenAI shape, so it needs nothing but its own base URL
// and key. Nothing above here knows which API answered.
//
// Keys live in .env and are read only here, server-side. They never reach the browser.

const OPENAI_API = 'https://api.openai.com/v1';
const ANTHROPIC_API = 'https://api.anthropic.com/v1';
const ANTHROPIC_VERSION = '2023-06-01';
// DeepSeek serves the OpenAI chat-completions shape from its own host: this is the base URL its
// own examples use, with /v1 as the alias the OpenAI SDKs get pointed at.
const DEEPSEEK_API = 'https://api.deepseek.com';

export const PROVIDERS = {
  openai: {
    label: 'OpenAI',
    keyEnv: 'OPENAI_API_KEY',
    modelEnv: 'OPENAI_MODEL',
    keysUrl: 'https://platform.openai.com/api-keys',
    keyHint: 'sk-…',
  },
  anthropic: {
    label: 'Anthropic',
    keyEnv: 'ANTHROPIC_API_KEY',
    modelEnv: 'ANTHROPIC_MODEL',
    keysUrl: 'https://console.anthropic.com/settings/keys',
    keyHint: 'sk-ant-…',
  },
  deepseek: {
    label: 'DeepSeek',
    keyEnv: 'DEEPSEEK_API_KEY',
    modelEnv: 'DEEPSEEK_MODEL',
    keysUrl: 'https://platform.deepseek.com/api_keys',
    keyHint: 'sk-…',
  },
};

// The clients we send through. OpenAI and DeepSeek take the same request and return the same
// reply, so they share one call path and differ only in the host they are reached at — and in
// whether the endpoint can be held to a JSON schema. Anthropic has its own client further down,
// because it needs its request translated.
const CLIENTS = {
  openai: { label: 'OpenAI', api: OPENAI_API },
  deepseek: {
    label: 'DeepSeek',
    api: DEEPSEEK_API,
    // DeepSeek's JSON mode takes {type:'json_object'} and nothing else, so a pass that asked for
    // a schema is asked for JSON in the prompt instead and unwrapped on the way back.
    jsonObject: true,
  },
};

// Ordered best-first. We ask the key what it can actually see rather than hardcoding one id,
// so a retired model never breaks the studio. Newer ids than these may exist — add them here,
// or pin OPENAI_MODEL / ANTHROPIC_MODEL / DEEPSEEK_MODEL in .env.
const PREFERRED = {
  openai: [
    'gpt-5.6-sol',
    'gpt-5.6',
    'gpt-5.5',
    'gpt-5.4',
    'gpt-5.3-chat-latest',
    'gpt-5.2',
    'gpt-5.1',
    'gpt-5',
    'gpt-5.4-mini',
    'gpt-5-mini',
    'gpt-4.1',
    'gpt-4o',
  ],
  // Ordered for this workload: every pass reads screenshots, and the drafting and review
  // passes ask for a JSON schema back, so models with native structured outputs come first.
  anthropic: [
    'claude-opus-5',
    'claude-sonnet-5',
    'claude-opus-4-8',
    'claude-haiku-4-5',
    'claude-opus-4-7',
    'claude-sonnet-4-6',
    'claude-opus-4-6',
  ],
  // Same reasoning as Anthropic, and it matters more here: only DeepSeek's Flash model takes
  // images, so it has to come first — every pass reads screenshots. The pro model is text-only,
  // and the legacy names still resolve to Flash for keys that only advertise those.
  deepseek: ['deepseek-flash', 'deepseek-v4-pro', 'deepseek-v4-flash', 'deepseek-v4-flash-vision-exp'],
};

const FALLBACK_MODEL = { openai: 'gpt-4o', anthropic: 'claude-sonnet-5', deepseek: 'deepseek-flash' };

// Anthropic requires an explicit output ceiling on every request. Generous enough for a long
// drafting pass, small enough to stay well inside an ordinary HTTP timeout.
const ANTHROPIC_MAX_TOKENS = 16000;

// ---------------------------------------------------------------- key + provider

// DeepSeek keys are `sk-` followed by 32 characters — short enough to tell apart from an OpenAI
// key (sk-proj-…, or the older sk-… at 48 or more), which is what lets the setup dialog take any
// of the three keys without asking which one you pasted.
const DEEPSEEK_KEY = /^sk-[A-Za-z0-9]{32}$/;

function keyOf(provider) {
  return (process.env[PROVIDERS[provider]?.keyEnv] || '').trim();
}

// A real key, not the .env.example placeholder or a blank line.
function hasKeyFor(provider) {
  const k = keyOf(provider);
  if (provider === 'anthropic') return k.startsWith('sk-ant-') && k.length > 20;
  if (provider === 'deepseek') return DEEPSEEK_KEY.test(k);
  return k.startsWith('sk-') && !k.startsWith('sk-ant-') && k.length > 20;
}

/** Which provider a pasted key belongs to, or null if it looks like none of them. */
export function detectProvider(key) {
  const k = String(key || '').trim();
  if (k.startsWith('sk-ant-')) return 'anthropic';
  if (DEEPSEEK_KEY.test(k)) return 'deepseek';
  if (k.startsWith('sk-')) return 'openai';
  return null;
}

// With more than one key present AI_PROVIDER decides; the studio writes it whenever a key is
// connected from the UI, so the last key you connected is the one that gets used. The order
// below is only the tie-break when nothing has pinned one.
export function activeProvider() {
  const want = String(process.env.AI_PROVIDER || '').trim().toLowerCase();
  if (PROVIDERS[want] && hasKeyFor(want)) return want;
  if (hasKeyFor('anthropic')) return 'anthropic';
  if (hasKeyFor('openai')) return 'openai';
  if (hasKeyFor('deepseek')) return 'deepseek';
  return null;
}

export function hasKey() {
  return activeProvider() !== null;
}

function key(provider = activeProvider()) {
  if (!provider) throw new Error('No API key connected. Add one from the studio home page.');
  return keyOf(provider);
}

/** Verify a pasted key against its own API. Returns the provider it belongs to. */
export async function verifyKey(k) {
  const provider = detectProvider(k);
  if (!provider) {
    throw new Error(
      'That doesn’t look like an API key — OpenAI keys start with sk-, Anthropic keys with sk-ant-, and DeepSeek keys are sk- followed by 32 characters.',
    );
  }
  const { host, res } =
    provider === 'anthropic'
      ? {
          host: 'api.anthropic.com',
          res: await fetch(`${ANTHROPIC_API}/models`, {
            headers: { 'x-api-key': k, 'anthropic-version': ANTHROPIC_VERSION },
          }).catch(() => null),
        }
      : {
          // OpenAI and DeepSeek both verify by listing models with a bearer token; only the host
          // differs, and the client table is the one place that knows it.
          host: new URL(CLIENTS[provider].api).host,
          res: await fetch(`${CLIENTS[provider].api}/models`, { headers: { Authorization: `Bearer ${k}` } }).catch(() => null),
        };

  const label = PROVIDERS[provider].label;
  if (!res) throw new Error(`Couldn’t reach ${host} — check your connection and try again.`);
  if (res.status === 401 || res.status === 403) throw new Error(`${label} rejected that key. Make sure it was copied in full.`);
  if (!res.ok) throw new Error(`Couldn’t verify the key (HTTP ${res.status}). Try again in a moment.`);
  return provider;
}

// ---------------------------------------------------------------- model choice

const cachedModel = new Map();
const modelError = new Map();

// Called after the key changes at runtime so the next call re-asks which models it can see.
export function resetModelCache() {
  cachedModel.clear();
  modelError.clear();
}

async function listModels(provider) {
  const res =
    provider === 'anthropic'
      ? await fetch(`${ANTHROPIC_API}/models?limit=100`, {
          headers: { 'x-api-key': key(provider), 'anthropic-version': ANTHROPIC_VERSION },
        })
      : await fetch(`${CLIENTS[provider].api}/models`, { headers: { Authorization: `Bearer ${key(provider)}` } });
  if (!res.ok) throw new Error(`models list failed: HTTP ${res.status}`);
  const data = await res.json();
  return new Set((data.data || []).map((m) => m.id));
}

export async function resolveModel(force = false) {
  const provider = activeProvider();
  if (!provider) return null;

  const pinned = (process.env[PROVIDERS[provider].modelEnv] || '').trim();
  if (pinned) return pinned;
  if (cachedModel.has(provider) && !force) return cachedModel.get(provider);

  const prefix = provider === 'anthropic' ? 'claude-' : provider === 'deepseek' ? 'deepseek-' : 'gpt-';
  try {
    const available = await listModels(provider);
    const pick =
      PREFERRED[provider].find((m) => available.has(m)) ||
      [...available].find((m) => m.startsWith(prefix)) ||
      FALLBACK_MODEL[provider];
    cachedModel.set(provider, pick);
    modelError.delete(provider);
  } catch (e) {
    modelError.set(provider, e.message);
    cachedModel.set(provider, FALLBACK_MODEL[provider]);
  }
  return cachedModel.get(provider);
}

export async function aiStatus() {
  const provider = activeProvider();
  if (!provider) return { configured: false, provider: null, providerLabel: null, model: null, error: 'No API key connected' };
  const model = await resolveModel();
  return {
    configured: true,
    provider,
    providerLabel: PROVIDERS[provider].label,
    model,
    error: modelError.get(provider) || null,
  };
}

// ---------------------------------------------------------------- dispatch

/**
 * Send an OpenAI-shaped chat request to whichever provider is connected, and get an
 * OpenAI-shaped reply back.
 */
export async function callModel(body) {
  const provider = activeProvider();
  if (!provider) throw new Error('No API key connected. Add one from the studio home page.');
  const model = body.model || (await resolveModel());
  // Anthropic gets its own translated client; OpenAI and DeepSeek share one, told apart by the
  // provider key it is sent through.
  return provider === 'anthropic' ? callAnthropic({ ...body, model }) : callOpenAI(provider, { ...body, model });
}

// ---------------------------------------------------------------- openai-shaped

// Params a given model has already rejected, so we stop sending them. Model generations
// differ on what they accept — the gpt-5 family, for instance, only allows the default
// temperature — and hardcoding a per-family table goes stale the moment a new one ships.
const unsupported = new Map();

// Models that refuse tool calls unless reasoning is explicitly switched off on this endpoint.
// Newer reasoning models reject "function tools with reasoning_effort" on /v1/chat/completions
// and name the remedy in the error, so learn it from them rather than hardcoding a list. How a
// model is told to stop reasoning differs by endpoint: OpenAI takes reasoning_effort, DeepSeek a
// thinking toggle.
const needsNoReasoning = new Set();
const NO_REASONING = {
  openai: { reasoning_effort: 'none' },
  deepseek: { thinking: { type: 'disabled' } },
};

function dropUnsupported(provider, model, body) {
  const dropped = unsupported.get(model);
  let out = body;
  if (dropped) {
    out = { ...out };
    for (const p of dropped) delete out[p];
  }
  if (needsNoReasoning.has(model) && out.tools?.length) out = { ...out, ...NO_REASONING[provider] };
  return out;
}

// A schema-constrained reply, asked for in the prompt. Used by DeepSeek, whose JSON mode takes
// {type:'json_object'} and nothing else, and by an Anthropic model that won't hold a schema.
function schemaPreamble(schema) {
  return (
    'Reply with a single JSON object and nothing else — no prose, no code fences. It must ' +
    `match this JSON Schema:\n${JSON.stringify(schema)}`
  );
}

// The OpenAI-shaped request, adjusted for an endpoint that can't be held to a JSON schema.
function toOpenAIRequest(provider, body) {
  if (!CLIENTS[provider].jsonObject) return body;
  const schema = body.response_format?.json_schema?.schema;
  if (!schema) return body;
  return {
    ...body,
    response_format: { type: 'json_object' },
    messages: [{ role: 'system', content: schemaPreamble(schema) }, ...(body.messages || [])],
  };
}

async function callOpenAI(provider, body, attempt = 0) {
  const client = CLIENTS[provider];
  const model = body.model;
  const wantsJson = !!body.response_format?.json_schema;
  const res = await fetch(`${client.api}/chat/completions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${key(provider)}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(toOpenAIRequest(provider, dropUnsupported(provider, model, body))),
  });

  if (res.ok) {
    const data = await res.json();
    if (client.jsonObject && wantsJson) {
      const msg = data.choices?.[0]?.message;
      // JSON mode is a nudge, not a guarantee: the schema went in through the prompt, so a code
      // fence around the reply has to come off before the caller parses it.
      if (typeof msg?.content === 'string' && msg.content.trim()) msg.content = unwrapJson(msg.content);
      if (!msg?.content?.trim()) throw new Error(`${client.label} came back with an empty reply. Try that pass again.`);
    }
    return data;
  }

  const text = await res.text();
  let msg = text;
  try {
    msg = JSON.parse(text).error?.message || text;
  } catch {}

  // "Function tools with reasoning_effort are not supported for <model> in /v1/chat/completions.
  // …or set reasoning_effort to 'none'." DeepSeek says the same about its thinking mode. Take
  // the remedy the API offers and remember it for that model.
  if (
    attempt < 4 &&
    !needsNoReasoning.has(model) &&
    /reasoning_effort|reasoning_content|thinking/i.test(msg) &&
    /not supported|unsupported|invalid|not allowed/i.test(msg)
  ) {
    needsNoReasoning.add(model);
    return callOpenAI(provider, body, attempt + 1);
  }

  // "Unsupported value: 'temperature' does not support 0.4…" / "Unsupported parameter: 'x'" on
  // OpenAI, "Invalid parameter: 'tool_choice'" on DeepSeek. Whichever wording arrives, if it
  // names something we sent, drop that parameter and ask again.
  const bad = /(?:unsupported|invalid|unknown|unrecognized)[^'"`]{0,40}?['"`]([A-Za-z_][A-Za-z0-9_]*)['"`]/i.exec(msg)?.[1];
  if (bad && attempt < 4 && bad in body) {
    const set = unsupported.get(model) || new Set();
    set.add(bad);
    unsupported.set(model, set);
    return callOpenAI(provider, body, attempt + 1);
  }

  throw new Error(`${client.label}: ${msg}`);
}

// ---------------------------------------------------------------- anthropic

// Learned per model, the same way the OpenAI path learns rejected params: a model that turns
// out not to accept adaptive thinking or a schema-constrained reply is simply asked without it
// from then on, rather than failing the pass.
const noThinking = new Set();
const noSchema = new Set();

// data: URLs are how every image reaches us (the shrink pass returns them, and the drafting
// pass builds them from the PNG on disk).
function imageBlock(url) {
  const m = /^data:([^;,]+);base64,(.+)$/s.exec(url || '');
  if (m) return { type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } };
  return url ? { type: 'image', source: { type: 'url', url } } : null;
}

function toBlocks(content) {
  if (typeof content === 'string') return content.trim() ? [{ type: 'text', text: content }] : [];
  const out = [];
  for (const part of content || []) {
    if (part.type === 'text' && part.text?.trim()) out.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url') {
      const block = imageBlock(part.image_url?.url);
      if (block) out.push(block);
    }
  }
  return out;
}

// OpenAI keeps system prompts, tool calls and tool results in one flat messages array;
// Anthropic takes the system prompt separately, carries tool calls as content blocks on the
// assistant turn, and expects tool results as a user turn.
function toAnthropicMessages(messages, model) {
  const system = [];
  const out = [];

  const addUser = (blocks) => {
    if (!blocks.length) return;
    const last = out[out.length - 1];
    if (last?.role === 'user') last.content.push(...blocks);
    else out.push({ role: 'user', content: blocks });
  };

  for (const m of messages || []) {
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content : toBlocks(m.content).map((b) => b.text || '').join('\n');
      if (text.trim()) system.push(text);
      continue;
    }

    if (m.role === 'tool') {
      addUser([{ type: 'tool_result', tool_use_id: m.tool_call_id, content: String(m.content ?? '') }]);
      continue;
    }

    if (m.role === 'assistant') {
      // An assistant turn we produced ourselves carries its original blocks, thinking included.
      // Replaying them verbatim is what keeps a multi-round tool conversation valid.
      let blocks = Array.isArray(m._blocks)
        ? m._blocks.filter((b) => !(noThinking.has(model) && b.type === 'thinking'))
        : toBlocks(m.content);
      if (!Array.isArray(m._blocks)) {
        for (const call of m.tool_calls || []) {
          let input = {};
          try {
            input = JSON.parse(call.function?.arguments || '{}');
          } catch {}
          blocks.push({ type: 'tool_use', id: call.id, name: call.function?.name, input });
        }
      }
      if (!blocks.length) continue;
      const last = out[out.length - 1];
      if (last?.role === 'assistant') last.content.push(...blocks);
      else out.push({ role: 'assistant', content: blocks });
      continue;
    }

    addUser(toBlocks(m.content));
  }

  if (!out.length || out[0].role !== 'user') out.unshift({ role: 'user', content: [{ type: 'text', text: 'Continue.' }] });
  return { system: system.join('\n\n'), messages: out };
}

// JSON Schema, minus the parts Anthropic's structured outputs don't take. The nullable fields
// our schemas use are written as `type: ['string', 'null']`, which becomes an anyOf.
function normalizeSchema(node) {
  if (Array.isArray(node)) return node.map(normalizeSchema);
  if (!node || typeof node !== 'object') return node;

  const out = {};
  for (const [k, v] of Object.entries(node)) {
    if (k === 'type' && Array.isArray(v)) continue;
    // Property names are data, not schema keywords — never recurse into them as keywords.
    out[k] = k === 'properties' || k === '$defs' || k === 'definitions'
      ? Object.fromEntries(Object.entries(v || {}).map(([name, sub]) => [name, normalizeSchema(sub)]))
      : normalizeSchema(v);
  }
  if (Array.isArray(node.type)) out.anyOf = node.type.map((t) => ({ type: t }));
  return out;
}

function toAnthropicRequest(body) {
  const model = body.model;
  const { system, messages } = toAnthropicMessages(body.messages, model);
  const req = { model, max_tokens: body.max_tokens || ANTHROPIC_MAX_TOKENS, messages };
  const preamble = system ? [system] : [];

  // Sampling parameters are rejected outright by the current Claude models, and thinking depth
  // is the knob that replaces them.
  if (!noThinking.has(model)) req.thinking = { type: 'adaptive' };

  if (body.tools?.length) {
    req.tools = body.tools.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: normalizeSchema(t.function.parameters),
    }));
    if (body.tool_choice === 'auto' || !body.tool_choice) req.tool_choice = { type: 'auto' };
  }

  const schema = body.response_format?.json_schema?.schema;
  if (schema) {
    if (noSchema.has(model)) {
      // Asked for by hand when the model can't be constrained to a schema. The reply is
      // unwrapped below before it reaches the caller's JSON.parse.
      preamble.push(schemaPreamble(schema));
    } else {
      req.output_config = { format: { type: 'json_schema', schema: normalizeSchema(schema) } };
    }
  }

  if (preamble.length) req.system = preamble.join('\n\n');
  return req;
}

// Models talked into returning JSON by prompt alone still reach for a code fence now and then.
// Shared by the Anthropic path (for a model that won't hold a schema) and by DeepSeek's JSON
// mode, which only takes {type:'json_object'}.
function unwrapJson(text) {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(text);
  const body = (fenced ? fenced[1] : text).trim();
  const start = body.search(/[{[]/);
  if (start < 0) return body;
  const end = Math.max(body.lastIndexOf('}'), body.lastIndexOf(']'));
  return end > start ? body.slice(start, end + 1) : body.slice(start);
}

function fromAnthropicResponse(res, wantsJson) {
  const blocks = res.content || [];
  let text = blocks.filter((b) => b.type === 'text').map((b) => b.text).join('').trim();
  if (wantsJson && text) text = unwrapJson(text);

  const toolCalls = blocks
    .filter((b) => b.type === 'tool_use')
    .map((b) => ({ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) } }));

  // `_blocks` keeps the original turn — thinking blocks and all — for the next round of a tool
  // conversation. Everything else here is the shape the callers already read.
  const message = { role: 'assistant', content: text || null, _blocks: blocks };
  if (toolCalls.length) message.tool_calls = toolCalls;

  return {
    id: res.id,
    model: res.model,
    usage: res.usage,
    choices: [{ index: 0, message, finish_reason: toolCalls.length ? 'tool_calls' : 'stop' }],
  };
}

async function callAnthropic(body, attempt = 0) {
  const model = body.model;
  const wantsJson = !!body.response_format?.json_schema;
  const res = await fetch(`${ANTHROPIC_API}/messages`, {
    method: 'POST',
    headers: {
      'x-api-key': key(),
      'anthropic-version': ANTHROPIC_VERSION,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(toAnthropicRequest(body)),
  });

  if (res.ok) {
    const data = await res.json();
    if (data.stop_reason === 'refusal') throw new Error('Anthropic declined that request.');
    if (data.stop_reason === 'max_tokens' && wantsJson) {
      throw new Error('Anthropic: the reply was cut off before it was complete. Try a shorter demo or fewer steps at a time.');
    }
    return fromAnthropicResponse(data, wantsJson);
  }

  const text = await res.text();
  let msg = text;
  try {
    msg = JSON.parse(text).error?.message || text;
  } catch {}

  // Learn what this model won't take, then ask again without it.
  if (attempt < 4 && !noThinking.has(model) && /thinking/i.test(msg)) {
    noThinking.add(model);
    return callAnthropic(body, attempt + 1);
  }
  if (attempt < 4 && wantsJson && !noSchema.has(model) && /output_config|json_schema|schema|format/i.test(msg)) {
    noSchema.add(model);
    return callAnthropic(body, attempt + 1);
  }

  throw new Error(`Anthropic: ${msg}`);
}
