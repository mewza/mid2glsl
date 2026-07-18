#!/usr/bin/env python3
"""
mid2glsl.py -- convert a Standard MIDI File into a Shadertoy sound shader.
No dependencies. SMF format 0/1, running status, tempo, multi-track.

Two emit modes:
  grains (default) : each note -> summed additive grain. Naturally POLYPHONIC,
                     no voice-splitting. Drawbar-organ instrument (after the
                     'good organ' additive shader) + compact AD envelope.
  macro            : tekf-style N()/L() delta-tick track fns (monophonic,
                     voice-split). Kept for the scan-based approach.

Usage:  python3 mid2glsl.py in.mid out.glsl [--bpm N] [--mode grains|macro]
        --viz [1|2|led|cd] : also emit a visualizer as ShaderToy Buffer A +
                Image passes (4 files: Common=data, Sound=synth, Buffer A=viz
                state, Image=render). 1/led (default) = "LED Band Spectro 3D"
                tunnel; 2/cd = "Soundform CD" raymarched disc (spectrum ring,
                diffraction rainbows, mouse-drag rotation, keys 1/2 toggle
                rings). Both are driven by the embedded MIDI notes — ShaderToy
                has no Sound-tab FFT input, so the FFT is midiSpec() and the CD
                waveform is a mono re-synthesis of mainSound.
        --inst brass|piano : synth voice for --mode player. piano = the
                "piano man" ShaderToy model (penguinPiano low register cross-
                faded into a heavily modified iq piano, minus an antiResLayer,
                plus 10 soundboard body modes excited per note; stretch tuning,
                inharmonicity, 2-3 detuned strings, hammer thwack/click).
        Every playable build also writes <base>_shadertoy.json — a one-click
        ShaderToy > Import descriptor (same schema as MOD2GLSL's emit_json;
        after import, DELETE the sacrificial "Buffer B" tab).
"""
import sys, struct, bisect, re, os

# ----------------------------------------------------------------- SMF parse
def _varlen(d, i):
    v = 0
    while True:
        b = d[i]; i += 1
        v = (v << 7) | (b & 0x7f)
        if not (b & 0x80): break
    return v, i

def _ctrl_at(evs, keys, tick, default):
    """last controller value at-or-before tick (evs sorted, keys = [t for t,_ in evs])"""
    i = bisect.bisect_right(keys, tick) - 1
    return evs[i][1] if i >= 0 else default

def _note_fx(n, bend, cc11, cc7, brange, progev):
    """(s,dur,note,vel,gate,ch) -> (s,dur,note,vel,gate, b0..b3, e0..e3).
    Pitch bend (signed 1/32-semitone, byte-biased +128) and expression gain
    (CC11 x CC7, 0..255) sampled at 4 points across the KEY-HELD span — the
    GLSL player lerps between them, so slides / swells / tracker fades survive
    the packing (fast vibrato won't; 4 points is the budget)."""
    s, dur, note, vel, gate, ch = n
    bev = bend.get(ch, []); bk = [t for t, _ in bev]
    e11 = cc11.get(ch, []); k11 = [t for t, _ in e11]
    e7  = cc7.get(ch, []);  k7  = [t for t, _ in e7]
    rng = brange.get(ch, 2.0)                       # GM default bend range: ±2 semitones
    bs, es = [], []
    for k in range(4):
        tk = s + (gate * k) // 3
        semis = _ctrl_at(bev, bk, tk, 0.0) * rng
        bs.append(max(0, min(255, round(semis * 32.0) + 128)))
        # CC7 initial = 100/127 per GM — but only when the channel actually uses
        # CC7; otherwise 1.0 so plain files keep their current loudness exactly
        g = (_ctrl_at(e11, k11, tk, 127) / 127.0) * (_ctrl_at(e7, k7, tk, 100 if e7 else 127) / 127.0)
        es.append(max(0, min(255, round(g * 255.0))))
    if ch == 9:
        prog = 128                                    # drum-kit sentinel (dispatched by note number)
    else:
        pev = progev.get(ch, []); pk = [t for t, _ in pev]
        prog = _ctrl_at(pev, pk, s, 0)                # GM program at note-on (0 = Ac. Grand)
    return (s, dur, note, vel, gate, *bs, *es, prog)

def parse_midi(path):
    d = open(path, "rb").read()
    assert d[:4] == b"MThd", "not a MIDI file"
    fmt, ntrk, div = struct.unpack(">HHH", d[8:14])
    tpq = div if div < 0x8000 else 480
    pos, tempo, tracks = 14, 500000, []
    tmap = []                        # FULL tempo map: every 0x51 meta as (abs_tick, us_per_quarter)
    for _ in range(ntrk):
        assert d[pos:pos+4] == b"MTrk"
        length = struct.unpack(">I", d[pos+4:pos+8])[0]
        pos += 8; end = pos + length
        t = 0; status = 0; on = {}; notes = []
        ped = {}; held = {}          # sustain pedal (CC64) per channel: down?, notes ringing past note-off
        bend = {}; cc11 = {}; cc7 = {}   # per-channel controller timelines [(tick, value)]
        rpn = {}; brange = {}            # RPN address / pitch-bend range (semitones) per channel
        progev = {}                      # per-channel GM program-change timeline [(tick, program)]
        while pos < end:
            dt, pos = _varlen(d, pos); t += dt
            b = d[pos]
            if b & 0x80: status = b; pos += 1
            ev = status & 0xf0; ch = status & 0x0f
            if status == 0xff:
                mtype = d[pos]; pos += 1
                mlen, pos = _varlen(d, pos)
                if mtype == 0x51:
                    tempo = struct.unpack(">I", b"\x00"+d[pos:pos+3])[0]
                    tmap.append((t, tempo))
                pos += mlen
            elif status in (0xf0, 0xf7):
                slen, pos = _varlen(d, pos); pos += slen
            elif ev in (0x90, 0x80):
                note, vel = d[pos], d[pos+1]; pos += 2
                if ch == 9:                           # GM drums: one-shots (kept for --gm, dropped in
                    if ev == 0x90 and vel > 0:        # mono/piano builds); ignore drum note-offs
                        _dd = max(1, tpq//8)
                        notes.append((t, _dd, note, vel, _dd, ch))
                    continue
                if ev == 0x90 and vel > 0:
                    if (ch, note) in held:   # re-strike of a pedal-sustained note ends the old one
                        s, v, off = held.pop((ch, note)); notes.append((s, t-s, note, v, off-s, ch))
                    on.setdefault(note, []).append((t, vel))
                else:
                    if on.get(note):
                        s, v = on[note].pop(0)
                        if ped.get(ch):      # pedal down: ring until pedal release
                            held[(ch, note)] = (s, v, t)      # remember the RAW key-off
                        else:
                            notes.append((s, t-s, note, v, t-s, ch))
            elif ev == 0xb0:
                cc, val = d[pos], d[pos+1]; pos += 2
                if cc == 64:
                    if val >= 64: ped[ch] = True
                    elif ped.get(ch):
                        ped[ch] = False
                        for (c, note), (s, v, off) in [it for it in held.items() if it[0][0] == ch]:
                            notes.append((s, t-s, note, v, off-s, c)); held.pop((c, note))
                elif cc == 11: cc11.setdefault(ch, []).append((t, val))
                elif cc == 7:  cc7.setdefault(ch, []).append((t, val))
                elif cc == 101: rpn[ch] = (val, rpn.get(ch, (127, 127))[1])
                elif cc == 100: rpn[ch] = (rpn.get(ch, (127, 127))[0], val)
                elif cc == 6 and rpn.get(ch) == (0, 0): brange[ch] = float(val)  # RPN 0,0 = bend range
            elif ev == 0xe0:
                lo, hi = d[pos], d[pos+1]; pos += 2
                bend.setdefault(ch, []).append((t, ((lo | (hi << 7)) - 8192) / 8192.0))
            elif ev == 0xc0:
                progev.setdefault(ch, []).append((t, d[pos])); pos += 1   # GM program change
            elif ev == 0xd0: pos += 1
            else:            pos += 2
        for (c, note), (s, v, off) in held.items():   # pedal never released
            notes.append((s, t-s, note, v, off-s, c))
        if notes: tracks.append(sorted(_note_fx(n, bend, cc11, cc7, brange, progev) for n in notes))
    parse_midi.tempo_map = sorted(set(tmap)) or [(0, tempo)]   # piecewise tick→sec for build_buffer
    return tpq, tempo, tracks

def ntof(n): return 440.0 * 2.0 ** ((n - 69) / 12.0)

# ============================================================ GRAINS (default)
GRAIN_TEMPLATE = """// auto-generated by mid2glsl.py (grains) -- {n} notes, polyphonic
#define T 6.28318530718

float w(float f, float x){{ return sin(x*f*T); }}

// compact AD envelope: linear attack vs sqrt(duration)-scaled exp decay
float env(float x, float start, float dur, float atk){{
    float dt = max(0.0, x-start);
    return min(dt/atk, exp(2.0*(atk-dt)/sqrt(dur)));
}}

// drawbar organ grain (harmonics 1,2,4,6 -- hollow Hammond color)
float g(float f, float start, float dur, float x){{
    float s = w(f,x)*0.15 + w(f*2.0,x)*0.20 + w(f*4.0,x)*0.075 + w(f*6.0,x)*0.015;
    return s * env(x, start, dur, 0.004);
}}

float music(float x){{
    float s = 0.0;
{body}
    return s;
}}

// ── master soft-knee limiter (ported 1:1 from MOD2GLSL's mod_player.py) ──
// over=max(|x|-T,0); reduced=HEAD*over/(over+HEAD); y=sign*(min(|x|,T)+reduced)
// T=0.85 knee: bit-perfect below it; the summed voices then APPROACH 1.0
// smoothly instead of hard-clipping (the many-voices distortion).
float softLimit1(float x){{
    const float T=0.85, HEAD=1.0-T;
    float ax=abs(x), over=max(ax-T,0.0);
    return sign(x)*(min(ax,T) + HEAD*over/(over+HEAD));
}}

vec2 mainSound(int samp, float time){{
    float x = time;                                   // tempo baked into grains
    float m = softLimit1(music(x)*{gain:.3f});
    return vec2(m * smoothstep(0.0, 0.3, time));
}}
"""

def build_grains(tpq, tempo, tracks, bpm=None):
    us_per_q = 60_000_000 / bpm if bpm else tempo
    tps = tpq * (1_000_000 / us_per_q)
    notes = sorted(n for trk in tracks for n in trk)          # flatten all voices
    calls = [f"g({ntof(n):.3f},{s/tps:.4f},{max(dur/tps,0.02):.4f},x)"
             for (s, dur, n, v, *_g) in notes]
    lines, chunk = [], 10
    for i in range(0, len(calls), chunk):
        lines.append("    s += " + " + ".join(calls[i:i+chunk]) + ";")
    body = "\n".join(lines)
    gain = 0.45
    return GRAIN_TEMPLATE.format(n=len(notes), body=body, gain=gain)

# ============================================================ MACRO (scan)
def split_voices(notes):
    voices = []
    for s, dur, n, v, *_g in notes:
        for vc in voices:
            if vc[-1][0] + vc[-1][1] <= s: vc.append((s, dur, n, v)); break
        else: voices.append([(s, dur, n, v)])
    return voices

def emit_track(name, voice):
    out, t = [], 0
    for i, (s, dur, n, _) in enumerate(voice):
        delta = s - t
        nxt = voice[i+1][0] if i+1 < len(voice) else None
        out.append(f"N({delta},{n})" if (nxt is not None and s+dur >= nxt)
                   else f"L({delta},{n},{dur})")
        t = s
    return (f"vec2 {name}(float x){{ vec2 r=vec2(-1.0); float t=0.0;\n"
            f"    {''.join(out)}\n    return r; }}")

MACRO_TEMPLATE = """// auto-generated by mid2glsl.py (macro) -- {nv} voices, {tps:.1f} ticks/s
#define TAU 6.28318530718
#define N(D,V) t+=float(D); if(x>t) r=vec2(V,t);
#define L(D,V,X) t+=float(D); if((x>t)&&(x<(t+float(X)))) r=vec2(V,t);

float NoteToHz(float n){{ return 440.0*exp2((n-69.0)/12.0); }}
float fbsin(float p,float fb){{ float y=sin(p); y=sin(p+fb*y); y=sin(p+fb*y); return y; }}
float env(float t,float a,float dec){{ return (1.0-exp(-t/max(a,1e-4)))*exp2(-t*(5.0/dec)); }}

float voice(vec2 ft){{
    float f=ft.x, t=ft.y; if(f<=0.0) return 0.0;
    float ph=TAU*f*t; float I=2.4*exp2(-t*4.0)+0.8; float m=I*fbsin(ph,0.5);
    return (sin(ph+m)+sin(ph*1.004+m))*0.5*env(t,0.012,0.55);
}}

{tracks}

const float kTPS = {tps:.6f};
vec2 note(vec2 midi, float now){{
    if(midi.x < 0.0) return vec2(0.0, 1e3);
    return vec2(NoteToHz(midi.x), max(now-midi.y,0.0)/kTPS); }}

// ── master soft-knee limiter (ported 1:1 from MOD2GLSL's mod_player.py) ──
// T=0.85 knee: bit-perfect below it; summed voices approach 1.0 smoothly
// instead of hard-clipping (the many-voices distortion).
float softLimit1(float x){{
    const float T=0.85, HEAD=1.0-T;
    float ax=abs(x), over=max(ax-T,0.0);
    return sign(x)*(min(ax,T) + HEAD*over/(over+HEAD));
}}

float PlayMidi(float time){{
    if(time<0.0) return 0.0; float x=time*kTPS; float r=0.0;
{plays}
    return softLimit1(r*{gain:.3f}); }}

vec2 mainSound(int samp, float time){{
    return vec2(PlayMidi(time)*smoothstep(0.0,0.4,time)); }}
"""

def build_macro(tpq, tempo, tracks, bpm=None):
    us_per_q = 60_000_000 / bpm if bpm else tempo
    tps = tpq * (1_000_000 / us_per_q)
    voices = [vc for trk in tracks for vc in split_voices(trk)]
    fns   = [emit_track(f"Trk{i}", vc) for i, vc in enumerate(voices)]
    plays = "\n".join(f"    r += voice(note(Trk{i}(x), x));" for i in range(len(voices)))
    gain = 1.0 / max(len(voices), 1) ** 0.5
    return MACRO_TEMPLATE.format(nv=len(voices), tps=tps,
                                 tracks="\n\n".join(fns), plays=plays, gain=gain)

# ============================================================ BUFFER (block-indexed)
# Emits the packed data block a block-indexed player (e.g. the modal-piano engine)
# consumes. Layout inferred strictly from that decoder's reads:
#   midiOM[i] = onTick | (meta<<16);  meta = noteOffset(7b) | (vel7<<7)
#   midiDur[i>>1] packs two 16b durations;  midiSC[b] = start(16b) | count(16b)
SHARD = 2048   # max elements per const array (lower if a driver still rejects it)

def emit_sharded(name, vals, shard=SHARD):
    """Split a uint list into <=shard-sized const uvec4 arrays + a runtime accessor.
    4 uints are packed per uvec4 element — which is what the GPU / ANGLE const-
    storage budget actually counts — so this quarters the declared slot count
    (e.g. uint[2048] -> uvec4[512]) and helps weak cards fit. uvec4 (not ivec4)
    because the data is unsigned: each lane reads straight back as a uint, with no
    reliance on uint(negativeInt) wrap for values >= 2^31 (long note durations)."""
    if not vals:
        vals = [0]
    chunks = [vals[i:i+shard] for i in range(0, len(vals), shard)]
    lines = []
    for c, ch in enumerate(chunks):
        packed = []
        for i in range(0, len(ch), 4):
            g = (ch[i:i+4] + [0, 0, 0, 0])[:4]         # zero-pad the final group to 4
            packed.append("uvec4(%du,%du,%du,%du)" % (g[0]&0xFFFFFFFF, g[1]&0xFFFFFFFF, g[2]&0xFFFFFFFF, g[3]&0xFFFFFFFF))
        lines.append(f"const uvec4 {name}{c}[{len(packed)}] = uvec4[{len(packed)}]({','.join(packed)});")
    acc = [f"uint {name}(uint i){{"]
    for c, ch in enumerate(chunks):
        rd = f"{name}{c}[i>>2u][int(i&3u)]"            # uvec4 element + lane -> uint
        if c == 0:
            acc.append(f"  if(i<{len(ch)}u) return {rd};")
        else:
            acc.append(f"  i-={len(chunks[c-1])}u; if(i<{len(ch)}u) return {rd};")
    acc.append("  return 0u;\n}")
    return "\n".join(lines) + "\n" + "\n".join(acc)

# ---- RTX-2080-class const-storage model (user-verified) ----------------------
# One SM has 64KB of 4-byte constant registers = 4096 vec4 slots per pass. The
# synth/player code's own tables and literals eat ~512 of them, leaving ~3584
# vec4s for packed data (the "Toccata and Fugue in Brute Force" ceiling).
# Separately, the compiler caps ANY single const array at 1024 ITEMS regardless
# of element type (int[1024] and vec4[1024] are both max-size), so every blob
# must ship as <=1024-element shards. emit_sharded() already satisfies this
# (2048 uints pack into uvec4[512]); the checks below make it a hard guarantee
# for EVERYTHING the tool emits, PCM sample blobs included.
MAX_ARRAY_ITEMS  = 1024
VEC4_SLOTS_TOTAL = 4096
VEC4_SLOTS_MISC  = 512
VEC4_DATA_BUDGET = VEC4_SLOTS_TOTAL - VEC4_SLOTS_MISC          # 3584

_LANES = {"float":1,"int":1,"uint":1,"bool":1,"vec2":2,"ivec2":2,"uvec2":2,
          "vec3":3,"ivec3":3,"uvec3":3,"vec4":4,"ivec4":4,"uvec4":4}
_DATA_PREFIXES = ("midi", "mv", "p5", "p6", "p7")   # melody lanes + PCM blobs

def _const_arrays(text):
    """(name, type, items) for every const array declared in a GLSL string;
    unsized declarations get their items counted from the initializer."""
    out = []
    for m in re.finditer(r'const\s+(\w+)\s+(\w+)\s*\[(\d*)\]', text):
        ty, name, n = m.group(1), m.group(2), m.group(3)
        if n:
            cnt = int(n)
        else:
            i = text.find('(', m.end())
            if i < 0: continue
            depth, cnt, j = 1, 1, i+1
            while j < len(text) and depth > 0:
                c = text[j]
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                elif c == ',' and depth == 1: cnt += 1
                j += 1
        out.append((name, ty, cnt))
    return out

def _v4slots(ty, items):
    return (items*_LANES.get(ty, 4) + 3)//4

def audit_glsl_pass(text, label, strict=True):
    """Verify one FINAL GLSL pass against the 2080 model: no const array over
    MAX_ARRAY_ITEMS items; data arrays (midi*/mv*/PCM) within VEC4_DATA_BUDGET;
    everything within the 4096-slot bank. Raises (or warns, strict=False)."""
    arrs  = _const_arrays(text)
    data  = sum(_v4slots(t, c) for n, t, c in arrs if n.startswith(_DATA_PREFIXES))
    total = sum(_v4slots(t, c) for n, t, c in arrs)
    bad   = [(n, t, c) for n, t, c in arrs if c > MAX_ARRAY_ITEMS]
    msgs  = []
    if bad:
        msgs.append(f"const array over {MAX_ARRAY_ITEMS} items (compiler limit): "
                    + ", ".join(f"{t} {n}[{c}]" for n, t, c in bad))
    if data > VEC4_DATA_BUDGET:
        msgs.append(f"data arrays use {data} vec4 slots > the {VEC4_DATA_BUDGET} budget "
                    f"(64KB SM constants minus ~{VEC4_SLOTS_MISC} for code) — lower --time")
    if total > VEC4_SLOTS_TOTAL:
        msgs.append(f"all const arrays use {total} vec4 slots > the {VEC4_SLOTS_TOTAL}-slot bank")
    for msg in msgs:
        if strict: raise SystemExit(f"!! {label}: {msg}")
        print(f"  !! {label}: {msg}")
    # unroll guard: constant-trip loops > 16 iterations must carry an opaque
    # sentinel (?:-ternary / ZERO param / DEUN|NU offset) or ANGLE-class
    # compilers unroll them — compile hangs & private-var blowups. Warn-only.
    defs = dict(re.findall(r'#define\s+(\w+)\s+(\d+)u?\b', text))
    for m in re.finditer(r'for\s*\(\s*u?int\s+\w+\s*=\s*\d+u?\s*;([^;{]*);', text):
        cond = m.group(1)
        if '?' in cond or re.search(r'\b(DEUN|ZERO|NU)\b', cond): continue
        b = re.search(r'<=?\s*(?:(\d+)u?|([A-Za-z_]\w*))\s*$', cond.strip())
        if not b: continue
        v = b.group(1) or b.group(2)
        n = int(v) if v.isdigit() else (int(defs[v]) if v in defs and defs[v].isdigit() else None)
        if n and n > 16:
            print(f"  !! {label}: constant loop bound {v}={n} lacks a de-unroll "
                  f"sentinel: for(...;{cond.strip()};...)")
    return data, total

def reshard_const_uvec4(glsl, max_items=MAX_ARRAY_ITEMS):
    """Split oversized const uvec4 arrays (e.g. a baked PCM blob's p6w0[2730])
    into <=max_items shards + a <name>at(uint) accessor, rewriting the reads."""
    while True:
        m = next((m for m in re.finditer(
                  r'const\s+uvec4\s+(\w+)\[(\d+)\]\s*=\s*uvec4\[\2\]\(', glsl)
                  if int(m.group(2)) > max_items), None)
        if m is None:
            return glsl
        name = m.group(1)
        i, depth = m.end(), 1
        while depth > 0:                                   # matching ')' of the init
            c = glsl[i]
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            i += 1
        body = glsl[m.end():i-1]
        elems, buf, depth = [], [], 0
        for c in body:                                     # top-level comma split
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if c == ',' and depth == 0:
                elems.append("".join(buf).strip()); buf = []
            else:
                buf.append(c)
        if "".join(buf).strip(): elems.append("".join(buf).strip())
        chunks = [elems[k:k+max_items] for k in range(0, len(elems), max_items)]
        decls = [f"const uvec4 {name}_{c}[{len(ch)}] = uvec4[{len(ch)}]({','.join(ch)});"
                 for c, ch in enumerate(chunks)]
        acc = [f"uvec4 {name}at(uint i){{"]
        for c, ch in enumerate(chunks):
            if c: acc.append(f"  i-={len(chunks[c-1])}u;")
            acc.append(f"  if(i<{len(ch)}u) return {name}_{c}[i];")
        acc.append(f"  return {name}_0[0];\n}}")
        rest = glsl[i:]
        if rest.startswith(';'): rest = rest[1:]
        glsl = glsl[:m.start()] + "\n".join(decls) + "\n" + "\n".join(acc) + rest
        outp, k, pat = [], 0, name + "["                   # rewrite reads -> nameat(
        while True:
            j = glsl.find(pat, k)
            if j < 0:
                outp.append(glsl[k:]); break
            prev = glsl[j-1] if j else " "
            if prev.isalnum() or prev == '_':              # p5w0_0[ etc: not a read
                outp.append(glsl[k:j+len(pat)]); k = j+len(pat); continue
            e, d2 = j+len(pat), 1
            while d2 > 0:
                ch2 = glsl[e]
                if ch2 == '[': d2 += 1
                elif ch2 == ']': d2 -= 1
                e += 1
            outp.append(glsl[k:j] + name + "at(" + glsl[j+len(pat):e-1] + ")")
            k = e
        glsl = "".join(outp)

def _pcm_blob(inst):
    """The baked PCM sample blob for --inst piano5/6/7, resharded to the
    MAX_ARRAY_ITEMS compiler limit. '' for synth-only instruments."""
    if inst == "piano5":   from piano5_pcm import PIANO5_PCM_GLSL as G
    elif inst == "piano6": from piano6_pcm import PIANO6_PCM_GLSL as G
    elif inst == "piano7": from piano7_pcm import PIANO7_PCM_GLSL as G
    else: return ""
    return reshard_const_uvec4(G)

def pcm_reserve_vec4(inst):
    """vec4 slots the instrument's PCM blob occupies — pre-subtracted from the
    melody budget so blob + melody always fit the same 3584-slot pool."""
    b = _pcm_blob(inst)
    return sum(_v4slots(t, c) for _, t, c in _const_arrays(b)) if b else 0

def build_buffer(tpq, tempo, tracks, bpm=None, Q=256, shift=6, pad_sec=1.2, sr=44100.0,
                 tempo_map=None, trim_s=None, max_notes=5600, gm=False, budget_vec4=None):
    us_per_q = 60_000_000/bpm if bpm else tempo
    sec_per_tick = us_per_q/1e6/tpq                      # seconds per source MIDI tick
    if bpm is None and tempo_map and len(tempo_map) > 1:
        # piecewise tick→seconds from the FULL tempo map (rubato/accel files were
        # previously mistimed: the parser's scalar tempo is just the last 0x51)
        pts, cum, lt, lu = [], 0.0, 0, tempo_map[0][1]
        for (tk, us) in tempo_map:
            cum += (tk - lt) * lu / 1e6 / tpq; pts.append((tk, cum, us)); lt, lu = tk, us
        _ks = [p[0] for p in pts]
        def t2s(tick):
            i = bisect.bisect_right(_ks, tick) - 1
            if i < 0: return tick * pts[0][2] / 1e6 / tpq
            tk, cum, us = pts[i]; return cum + (tick - tk) * us / 1e6 / tpq
    else:
        t2s = lambda tick: tick * sec_per_tick
    qSec = Q/sr                                          # seconds per output tick
    c16 = lambda x: min(int(x), 0xFFFF)
    notes = []
    _pw = bpm is None and tempo_map and len(tempo_map) > 1   # piecewise?
    for trk in tracks:
        for (s, dur, n, v, *g) in trk:
            gate = g[0] if g else dur
            if _pw:
                onS = t2s(s)
                onT = round(onS/qSec)
                dT  = max(1, round((t2s(s+dur)  - onS)/qSec))
                gT  = max(1, round((t2s(s+gate) - onS)/qSec))
            else:   # single tempo: keep the exact legacy expressions (bit-stable)
                onT = round(s*sec_per_tick/qSec)
                dT  = max(1, round(dur*sec_per_tick/qSec))
                gT  = max(1, round(gate*sec_per_tick/qSec))
            bd  = tuple(g[1:5]) if len(g) >= 5 else (128, 128, 128, 128)  # bend lane
            ex  = tuple(g[5:9]) if len(g) >= 9 else (255, 255, 255, 255)  # expression lane
            pg  = g[9] if len(g) >= 10 else 0                             # GM program
            if pg == 128 and not gm: continue                            # drums: only in --gm builds
            notes.append((onT, dT, n, v, gT, bd, ex, pg))
    notes.sort(key=lambda e: e[0])
    _cap = None
    if trim_s:
        trim_ticks = round(trim_s/qSec)
        kept = [e for e in notes if e[0] < trim_ticks]
        if len(kept) < len(notes):
            print(f"  trimmed to the {trim_s:.0f}s playback limit: {len(kept)} of "
                  f"{len(notes)} notes kept (--time N to change)")
        notes = kept
        _cap = trim_ticks
    # RTX-2080 const model: fit the melody into its vec4-slot budget by EXACT
    # accounting of every array emitted below (OM/DURW/GTW/SC + optional
    # BD/EXP/PGW), trimming trailing notes if needed. budget_vec4 arrives
    # pre-shrunk by pcm_reserve_vec4() when a --inst piano5/6/7 blob rides
    # along in the same pass.
    if not notes:
        raise SystemExit("!! no notes to emit (empty MIDI, or --time too small)")
    budget = VEC4_DATA_BUDGET if budget_vec4 is None else budget_vec4
    _pmax, _pbd, _pex = [0], [False], [False]      # prefix: max end-tick / lane use
    for e in notes:
        _pmax.append(max(_pmax[-1], e[0]+e[1]))
        _pbd.append(_pbd[-1] or e[5] != (128, 128, 128, 128))
        _pex.append(_pex[-1] or e[6] != (255, 255, 255, 255))
    c4 = lambda n: (n+3)//4
    def _cost(k):                                  # exact vec4 slots for k notes
        if k == 0: return 0
        et = min(_pmax[k]+1, 0xFFFF)
        if _cap is not None: et = min(et, _cap)
        nb = (et + (1 << shift) - 1) >> shift
        v = c4(k) + 2*c4((k+1)//2) + c4(nb)        # OM + DURW/GTW + SC
        if _pbd[k]: v += c4(k)                     # bend lane
        if _pex[k]: v += c4(k)                     # expression lane
        if gm:      v += c4((k+3)//4)              # program lane
        return v
    if _cost(len(notes)) > budget:
        lo, hi = 0, len(notes)
        while lo < hi:                             # largest prefix that fits
            mid = (lo+hi+1)//2
            if _cost(mid) <= budget: lo = mid
            else: hi = mid-1
        if lo == 0:
            raise SystemExit(f"!! melody cannot fit the {budget}-vec4 GPU const "
                             f"budget (PCM blob too large for this instrument?)")
        print(f"  !! GPU const budget: full melody needs {_cost(len(notes))} vec4 "
              f"slots, {budget} available — kept {lo} of {len(notes)} notes "
              f"(~{_pmax[lo]*qSec:.0f}s of music; lower --time to pick the cut)")
        notes = notes[:lo]
    if len(notes) > max_notes:
        print(f"  !! {len(notes)} notes exceeds the ~{max_notes} proven-safe count "
              f"on ANGLE-class drivers (fits the 2080 const model though)")
    print(f"  const budget: {_cost(len(notes))} of {budget} vec4 slots "
          f"({len(notes)} notes{', bend' if _pbd[len(notes)] else ''}"
          f"{', expr' if _pex[len(notes)] else ''}{', prog' if gm else ''})")
    N = len(notes)
    note_min = min(e[2] for e in notes)
    blk = 1 << shift
    end_tick = c16(max(e[0]+e[1] for e in notes) + 1)
    if _cap is not None: end_tick = min(end_tick, _cap)   # loop at the trim point
    nblocks  = (end_tick + blk - 1)//blk

    OM, DUR, GT, BD, EXP, PG = [], [], [], [], [], []
    for onT, dT, n, v, gT, bd, ex, pg in notes:
        meta = ((n-note_min) & 0x7F) | ((max(0, min(127, v))) << 7)
        OM.append(c16(onT) | (meta << 16))
        DUR.append(c16(dT))
        GT.append(c16(gT))
        BD.append(bd[0] | (bd[1] << 8) | (bd[2] << 16) | (bd[3] << 24))
        EXP.append(ex[0] | (ex[1] << 8) | (ex[2] << 16) | (ex[3] << 24))
        PG.append(pg & 0xFF)
    while len(PG) % 4: PG.append(0)
    PGW = [PG[i] | (PG[i+1] << 8) | (PG[i+2] << 16) | (PG[i+3] << 24) for i in range(0, len(PG), 4)]
    if len(DUR) % 2: DUR.append(0)
    DURW = [DUR[i] | (DUR[i+1] << 16) for i in range(0, len(DUR), 2)]
    if len(GT) % 2: GT.append(0)
    GTW = [GT[i] | (GT[i+1] << 16) for i in range(0, len(GT), 2)]
    # bend/expression arrays only ship when the file actually uses them — plain
    # MIDIs pay zero const-storage (ANGLE budget) and zero per-note ALU
    has_bend = any(w != 0x80808080 for w in BD)
    has_expr = any(w != 0xFFFFFFFF for w in EXP)

    SC, idx = [], 0
    for b in range(nblocks):
        start, cnt = idx, 0
        while idx < N and (notes[idx][0] >> shift) == b:
            idx += 1; cnt += 1
        SC.append((start & 0xFFFF) | (min(cnt, 0xFFFF) << 16))

    max_dur   = max(e[1] for e in notes)
    pad_ticks = round(pad_sec/qSec)
    lookback  = (max_dur + pad_ticks)//blk + 2

    defs = "\n".join([
        "// --- paste into the Common tab; drives the block-indexed player ---",
        "#define MIDI_SAMPLE_RATE 44100.0",
        f"#define MIDI_TIME_Q_SAMPLES {Q}",
        f"#define MIDI_END_TICK {end_tick}u",
        f"#define MIDI_BLOCK_SHIFT_TICKS {shift}u",
        f"#define MIDI_BLOCK_COUNT {nblocks}u",
        f"#define MIDI_SEG_COUNT {N}u",
        f"#define MIDI_LOOKBACK_BLOCKS {lookback}u",
        f"#define MIDI_NOTE_MIN {note_min}",
        "#define MIDI_VEL_BITS 7u",
        f"#define MIDI_HAS_BEND {1 if has_bend else 0}",
        f"#define MIDI_HAS_EXPR {1 if has_expr else 0}",
        f"#define MIDI_HAS_PROG {1 if gm else 0}", ""])
    blob = (defs + "\n" + emit_sharded("midiOM", OM) + "\n\n"
            + emit_sharded("midiDur", DURW) + "\n\n" + emit_sharded("midiSC", SC) + "\n\n"
            + emit_sharded("midiGT", GTW) + "\n")
    if has_bend: blob += "\n" + emit_sharded("midiBD", BD) + "\n"
    if has_expr: blob += "\n" + emit_sharded("midiEXP", EXP) + "\n"
    if gm:       blob += "\n" + emit_sharded("midiPG", PGW) + "\n"
    progs_used = sorted(set(p & 0xFF for p in PG))
    return blob, [e[:5] for e in notes], dict(
            Q=Q, shift=shift, blk=blk, nblocks=nblocks, lookback=lookback,
            note_min=note_min, end_tick=end_tick, N=N, progs=progs_used)

PIANO7_GLSL = r"""// ── Bösendorfer × waveguide ("piano7") — the merge: THE boesendorfer.mod
// piano samples (slots 1/4/5, native 8363Hz) open every note, then hand off
// to the Lorenzoncina WAVEGUIDE STRING rendered in closed modal form (mode
// freqs solve the loop phase condition with the ad/ap_num dispersion — real
// inharmonic stretch; 3 offtuned parallel loops = unison; per-mode decay from
// the gl loss table). Seam fixes for "detuned / ringing off":
//   * roots measured by autocorrelation IN THE CROSSFADE WINDOW → the tail is
//     in tune with the sample exactly where they overlap
//   * modal decay CALIBRATED to each sample's measured late decay (p7dec) —
//     partial 1 rings exactly as long as the recording does
#ifndef P7_SAMP
#define P7_SAMP 1.0
#endif
#ifndef P7_SYNTH
#define P7_SYNTH 0.9
#endif
#ifndef P7_DAMP
#define P7_DAMP 1.0           // ring multiplier (>1 = drier tail)
#endif
#define P7_AD (-0.30)
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
vec3 p7wgParams(float f0){
    if(f0 > 3000.0)   return vec3(-0.997,  0.0, 0.01);
    if(f0 > 1900.0)   return vec3(-0.997,  2.0, 0.005);
    if(f0 > 1800.0)   return vec3(-0.997,  3.0, 0.005);
    if(f0 > 1500.0)   return vec3(-0.995,  4.0, 0.01);
    if(f0 >  980.0)   return vec3(-0.995,  6.0, 0.02);
    if(f0 >  750.0)   return vec3(-0.993,  8.0, 0.03);
    if(f0 >  390.0)   return vec3(-0.99,  12.0, 0.04);
    if(f0 >  261.626) return vec3(-0.985, 14.0, 0.06);
    if(f0 >  200.0)   return vec3(-0.98,  16.0, 0.09);
    if(f0 >  150.0)   return vec3(-0.975, 18.0, 0.13);
    if(f0 >  120.0)   return vec3(-0.968, 20.0, 0.18);
    return               vec3(-0.96,  20.0, 0.25);
}
float p7atanX(float w){
    return atan( (P7_AD*P7_AD - 1.0)*sin(w), 2.0*P7_AD + (P7_AD*P7_AD + 1.0)*cos(w) );
}
float p7tuneDelay(float C){ return (1.0 - C)/(1.0 + C); }
// one modal waveguide string; decScale locks partial-1 ring to the sample
float p7wgString(float t, float Nlin, float apn, float gl, float v, float decScale, int nPart){
    float y = 0.0;
    for(int k = 1; k <= 16; k++){
        if(k > nPart) break;
        float fk = float(k);
        float w = 6.2831853*fk/Nlin;
        for(int it = 0; it < 4; it++)
            w = (6.2831853*fk + apn*p7atanX(w))/Nlin;
        float fHz = w*7018.0;                 // w*FS/2π, FS=44100
        if(fHz > 15000.0) break;
        float lossMag = min(abs(gl), 0.99995);
        float rate = -log(lossMag)*(44100.0/Nlin)*decScale;
        float comb = abs(sin(3.14159265*fk*0.12))*0.85 + 0.15;
        float amp = comb/pow(fk, 0.25);
        amp /= (1.0 + pow(fHz/(3000.0 + 6500.0*v), 2.0));
        y += amp*exp(-rate*t)*sin(w*44100.0*t + fract(fk*0.618)*6.2831853);
    }
    return y;
}
vec2 p7tail(float t, float f0, float v, int nPart, int nStr){
    vec3 P = p7wgParams(f0);
    float gl = P.x, apn = P.y, offt = P.z;
    float w0 = 6.2831853*f0/44100.0;
    float Nex = (6.2831853 + apn*p7atanX(w0))/w0;
    float M = floor(Nex/2.0);
    float Pd = Nex - 2.0*M;
    float C = (1.0 - Pd)/(1.0 + Pd);
    // ring calibration: waveguide partial-1 rate → the sample's measured decay
    float rate1 = -log(min(abs(gl),0.99995))*(44100.0/Nex);
    int a = 0; float best = 1e9;
    for(int i = 0; i < 3; i++){ float dd = abs(log(f0/p7root[i])); if(dd < best){ best = dd; a = i; } }
    float decScale = P7_DAMP*(p7dec[a]*(f0/p7root[a]))/max(rate1, 0.05);
    float s1 = p7wgString(t, 2.0*M + p7tuneDelay(C), apn, gl, v, decScale, nPart);
    float s2 = 0.0, s3 = 0.0;
    if(nStr > 1){
        float C2 = min(C*(1.0 + offt), 0.999);
        s2 = p7wgString(t, 2.0*M + p7tuneDelay(C2), apn, gl, v, decScale, nPart);
    }
    if(nStr > 2){
        float C3 = C*(1.0 - offt);
        s3 = p7wgString(t, 2.0*M + p7tuneDelay(C3), apn, gl, v, decScale, nPart);
    }
    return vec2(0.40*s1 + 0.36*s2 + 0.24*s3,
                0.40*s1 + 0.24*s2 + 0.36*s3) / (nStr > 2 ? 1.0 : (nStr > 1 ? 0.76 : 0.40));
}
vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 6.0) return vec2(0.0);   // t<0: reverb taps pre-onset
    float f = note2freq(k);
    int a = 0; float best = 1e9;
    for(int i = 0; i < 3; i++){ float dd = abs(log(f/p7root[i])); if(dd < best){ best = dd; a = i; } }
    float rate = f/p7root[a];                  // seam-window root → in-tune handoff
    float sp = t*8363.0*rate;
    float u  = sp/float(p7len[a]);
    float smp = 0.0;
    if(u < 1.0){
        uint idx = p7start[a] + uint(sp);
        smp = mix(p7pcm(idx), p7pcm(idx+1u), fract(sp));
        smp *= 1.0 - smoothstep(0.55, 0.95, u);
    }
    float xf = smoothstep(0.30, 0.80, min(u, 1.0));
    float regW = mix(1.0, 0.35, smoothstep(70.0, 96.0, k));
    vec2 tail = p7tail(t, f, v, 16, 3)*regW*xf*v*P7_SYNTH;
    vec2 o = vec2(P7_SAMP*smp*(0.35 + 0.65*v)) + tail;
    float pan = clamp((k - 60.0)/40.0, -1.0, 1.0)*0.35*smoothstep(80.0, 220.0, f);
    o *= vec2(1.0 - pan, 1.0 + pan);
    o *= exp(-max(0.0, t - s)*(6.0 + f*0.004));
    return o * 1.1;
}
// lite wet: sample + 8-partial 2-string tail (reverb taps stay affordable)
vec2 pianoWet(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 6.0) return vec2(0.0);
    float f = note2freq(k);
    int a = 0; float best = 1e9;
    for(int i = 0; i < 3; i++){ float dd = abs(log(f/p7root[i])); if(dd < best){ best = dd; a = i; } }
    float rate = f/p7root[a];
    float sp = t*8363.0*rate;
    float u  = sp/float(p7len[a]);
    float smp = 0.0;
    if(u < 1.0){
        uint idx = p7start[a] + uint(sp);
        smp = mix(p7pcm(idx), p7pcm(idx+1u), fract(sp));
        smp *= 1.0 - smoothstep(0.55, 0.95, u);
    }
    float xf = smoothstep(0.30, 0.80, min(u, 1.0));
    float regW = mix(1.0, 0.35, smoothstep(70.0, 96.0, k));
    vec2 o = vec2(P7_SAMP*smp*(0.35 + 0.65*v)) + p7tail(t, f, v, 8, 2)*regW*xf*v*P7_SYNTH;
    o *= exp(-max(0.0, t - s)*(6.0 + f*0.004));
    return o * 1.1;
}"""

PIANO6_GLSL = r"""// ── Bösendorfer hybrid ("piano6") — THE boesendorfer.mod piano samples
// (slots 1/3/4/5/6 = C2/C3/C4/C5/C6, native 8363Hz 8-bit — bit-faithful to
// what the tracker plays) opening every note, crossfaded into the same
// 2-op FM tail as piano5. Full circle: boesendorfer.mid on its own piano.
// Requires the p6pcm data block (mid2glsl adds it to Common / file head).
#ifndef P6_SAMP
#define P6_SAMP 1.0           // sample-attack level
#endif
#ifndef P6_SYNTH
#define P6_SYNTH 0.9          // synth-tail level
#endif
#ifndef P6_RING
#define P6_RING 1.0           // tail ring multiplier (higher = drier/shorter)
#endif
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
// ring rate fitted to the MEASURED sample decays (C2 0.70/s, C4 1.82, C5 4.30)
// — the old flat 0.55+f·0.0011 rang 2-3× longer than the piano it extends
float p6dec(float f0){ return clamp(1.8*pow(f0/264.0, 0.95), 0.55, 8.0)*P6_RING; }
float p6fm(float time, float f0, float v)
{
    if(time < 0.0) return 0.0;
    float ph = 6.2831*f0*time;
    float I = (0.35 + 0.9*v)*exp(-time*(1.6 + f0*0.0012)) + 0.18;
    float y = sin(ph + I*sin(ph));
    y += 0.22*sin(2.0*ph + 0.7*I*sin(ph));
    return y*exp(-time*p6dec(f0));
}
vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 6.0) return vec2(0.0);   // t<0: reverb taps pre-onset
    float f = note2freq(k);
    // the COMPOSER's register map (traversed from boesendorfer.mod pattern
    // data): each anchor covers one octave UPWARD of its root, so the playback
    // rate stays in 1.0..2.81 — the tracker never down-pitches. That is the
    // MOD's brightness trick: every note is an up-sped sample (faster decay,
    // wider bandwidth), never a dulled slowed-down one.
    int a = (k < 55.0) ? 0 : (k < 67.0) ? 1 : (k < 79.0) ? 2 : (k < 91.0) ? 3 : 4;
    float rate = f/p6root[a];
    // the REAL Bösendorfer sample at native 8363Hz, tracker-style rate playback:
    // attack+body play through, then the sample LOOPS exactly as the tracker
    // sustains it. The bank stores [pre | loop cycle]; the bake's skipped
    // mid-decay is reapplied here (p6dcy → p6flr, measured from the sample),
    // p6lg rescales the full-scale-baked loop back to seam level.
    float sp = t*8363.0*rate;
    float pre = float(p6pre[a]);
    float u  = min(sp/pre, 1.0);                 // body progress (synth-tail pacing)
    float smp = 0.0;
    if(p6ll[a] == 0u){                           // one-shot anchor (C6, no loop in the MOD)
        if(sp < pre){
            uint idx = p6start[a] + uint(sp);
            smp = mix(p6pcm(idx), p6pcm(idx+1u), fract(sp));
            smp *= 1.0 - smoothstep(0.55, 0.95, sp/pre);
        }
    }else{
        float ll = float(p6ll[a]);
        // past the baked 256-sample seam crossfade the play head cycles the loop
        float spl = (sp < pre) ? sp : pre + mod(sp - pre + 256.0, ll);
        float sp1 = spl + 1.0;                   // 2nd interp tap, wrap-aware
        if(sp < pre)           sp1 = min(sp1, pre - 1.0);
        else if(sp1 >= pre+ll) sp1 -= ll;
        smp = mix(p6pcm(p6start[a] + uint(spl)), p6pcm(p6start[a] + uint(sp1)), fract(spl));
        if(sp >= pre)
            smp *= p6lg[a]*max(exp(-(sp - pre)/8363.0*p6dcy[a]), p6flr[a]);
    }
    float xf = smoothstep(0.30, 0.80, u);
    float tail = ( p6fm(t, f, v) + 0.6*p6fm(t + 0.0013, f*1.00032, v) )/1.6;
    tail += 0.06*sin(6.2831*f*t + 1.1)*exp(-t*p6dec(f)*0.55);   // aftersound: 2nd stage, now brief
    float regW = mix(1.0, 0.35, smoothstep(70.0, 96.0, k));
    vec2 o = vec2( P6_SAMP*smp*(0.35 + 0.65*v) + P6_SYNTH*regW*tail*xf*v );
    float pan = clamp((k - 60.0)/40.0, -1.0, 1.0)*0.35*smoothstep(80.0, 220.0, f);
    o *= vec2(1.0 - pan, 1.0 + pan);
    o *= exp(-max(0.0, t - s)*(6.0 + f*0.004));
    return o * 1.1;
}
// wet voice (reverb taps): LITE — replays only the PCM sample. The FM gloss +
// aftersound are quiet and vanish into the 2×REVERB_TAPS tap sum, but they cost
// ~6 transcendentals per eval; dropping them ≈ halves the preload's tap bill.
// Hammer attack still suppressed (taps replay the TONE, not the strike).
vec2 pianoWet(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 6.0) return vec2(0.0);
    float f = note2freq(k);
    int a = (k < 55.0) ? 0 : (k < 67.0) ? 1 : (k < 79.0) ? 2 : (k < 91.0) ? 3 : 4;
    float rate = f/p6root[a];
    float sp = t*8363.0*rate;
    float pre = float(p6pre[a]);
    float smp = 0.0;
    if(p6ll[a] == 0u){
        if(sp < pre){
            uint idx = p6start[a] + uint(sp);
            smp = mix(p6pcm(idx), p6pcm(idx+1u), fract(sp));
            smp *= 1.0 - smoothstep(0.55, 0.95, sp/pre);
        }
    }else{
        float ll = float(p6ll[a]);
        float spl = (sp < pre) ? sp : pre + mod(sp - pre + 256.0, ll);
        float sp1 = spl + 1.0;
        if(sp < pre)             sp1 = min(sp1, pre - 1.0);
        else if(sp1 >= pre + ll) sp1 -= ll;
        smp = mix(p6pcm(p6start[a] + uint(spl)), p6pcm(p6start[a] + uint(sp1)), fract(spl));
        if(sp >= pre)
            smp *= p6lg[a]*max(exp(-(sp - pre)/8363.0*p6dcy[a]), p6flr[a]);
    }
    vec2 o = vec2( P6_SAMP*smp*(0.35 + 0.65*v) );
    float pan = clamp((k - 60.0)/40.0, -1.0, 1.0)*0.35*smoothstep(80.0, 220.0, f);
    o *= vec2(1.0 - pan, 1.0 + pan);
    o *= exp(-max(0.0, t - s)*(6.0 + f*0.004));
    return o * 1.1 * smoothstep(0.0, 0.06, t);
}"""

PIANO5_GLSL = r"""// ── Salamander hybrid ("piano5") — REAL Yamaha C5 attack (Salamander Grand
// Piano V3 by Alexander Holm, CC-BY 3.0): 7 octave anchors A0-A6 @ 22.05kHz
// 8-bit with measured roots, crossfaded into a piano4-DNA cascade tail.
// Commuted synthesis, ShaderToy edition: the sample carries hammer/felt/body
// (the un-synthesizable part), the synth carries sustain + velocity.
// Requires the p5pcm data block (mid2glsl adds it to Common / file head).
#ifndef P5_SAMP
#define P5_SAMP 1.0           // sample-attack level
#endif
#ifndef P5_SYNTH
#define P5_SYNTH 0.9          // synth-tail level
#endif
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
// 2-op FM tail (the SY77/RCM half): 1:1 pair whose index decays bright -> warm
// (velocity-scaled), + a small octave op for body; register-true decay
// (bass rings, treble dies) so the tail matches the real snippet it follows.
float p5fm(float time, float f0, float v)
{
    if(time < 0.0) return 0.0;
    float ph = 6.2831*f0*time;
    float I = (0.35 + 0.9*v)*exp(-time*(1.6 + f0*0.0012)) + 0.18;
    float y = sin(ph + I*sin(ph));
    y += 0.22*sin(2.0*ph + 0.7*I*sin(ph));
    return y*exp(-time*(0.55 + f0*0.0011));
}
vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 6.0) return vec2(0.0);   // t<0: reverb taps pre-onset
    float f = note2freq(k);
    // nearest recorded anchor (max resample +-6 semitones)
    int a = 0; float best = 1e9;
    for(int i=0; i<7; i++){ float dd = abs(log(f/p5root[i])); if(dd < best){ best = dd; a = i; } }
    float rate = f/p5root[a];
    // the REAL attack: Salamander PCM, tracker-style rate playback
    float sp = t*22050.0*rate;
    float u  = sp/float(p5len[a]);
    float smp = 0.0;
    if(u < 1.0){
        uint idx = p5start[a] + uint(sp);
        smp = mix(p5pcm(idx), p5pcm(idx+1u), fract(sp));
        smp *= 1.0 - smoothstep(0.55, 0.95, u);          // hand over to the synth
    }
    // FM tail (detuned unison pair + aftersound), fading in as the sample
    // bows out. Register blend: treble notes live almost entirely in the REAL
    // snippet (they die that fast), bass leans on the FM ring.
    float xf = smoothstep(0.30, 0.80, min(u, 1.0));
    float tail = ( p5fm(t, f, v) + 0.6*p5fm(t + 0.0013, f*1.00032, v) )/1.6;
    tail += 0.10*sin(6.2831*f*t + 1.1)*exp(-t*(0.9 + f*0.0005));
    float regW = mix(1.0, 0.35, smoothstep(70.0, 96.0, k));
    vec2 o = vec2( P5_SAMP*smp*(0.35 + 0.65*v) + P5_SYNTH*regW*tail*xf*v );
    float pan = clamp((k - 60.0)/40.0, -1.0, 1.0)*0.35*smoothstep(80.0, 220.0, f);
    o *= vec2(1.0 - pan, 1.0 + pan);
    o *= exp(-max(0.0, t - s)*(6.0 + f*0.004));          // damper
    return o * 1.1;
}
vec2 pianoWet(float k, float t, float s, float v, int ZERO){ return pianoStrings(k, t, s, v, ZERO); }"""

PIANO4_GLSL = r"""// ── enhanced cascade piano ("piano4") — piano2 (the user's ear-test champion)
// upgraded without touching the core: the 25-partial descending cascade is
// byte-equivalent at neutral settings. Added, all tunable:
//   P4_B          inharmonic ladder f_k = k·f0·√(1+B·k²) (0 = piano2's exact
//                 harmonic ladder); default = 60% of piano3's register curve
//   Railsback stretch tuning (piano3's noteToHz)
//   P4_BRIGHT     velocity → spectral tilt (piano2 had loudness-only velocity)
//   register-dependent damper (bass rings longer after note-off)
//   hammer click + thump; 1.5ms raised-cosine attack replaces the digital
//   onset click; octave double detuned ×1.0008 for shimmer
#ifndef P4_B_SCALE
#define P4_B_SCALE 0.6        // 0.0 = piano2's exact harmonic ladder
#endif
#ifndef P4_BRIGHT
#define P4_BRIGHT 0.10        // velocity tilt depth (0 = piano2 behavior)
#endif
#ifndef P4_RELEASE
#define P4_RELEASE 1.0        // note-off damper speed multiplier (higher = shorter release)
#endif
#ifndef P4_UNISON
#define P4_UNISON 0.6         // detuned unison-string mix (0 = piano2's single string);
#endif                        // gives the beating + two-stage decay of real 2-3 string courses
#ifndef P4_WET_RUNGS
#define P4_WET_RUNGS 9        // cascade rungs for the REVERB taps (exact ladder truncation;
#endif                        // high rungs die in ms and are inaudible in the wet — big load win)
#ifndef P4_STRIKE
#define P4_STRIKE 0.35        // strike-position comb depth (0 = piano2's flat rolloff);
#endif                        // notches ~the 8th harmonic region -> "wooden", less buzzy
#ifndef P4_AFTER
#define P4_AFTER 0.10         // aftersound: quiet slow fundamental = real two-stage decay
#endif
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
float p4Hz(float k){                   // Railsback stretch (from piano3)
    float d = k - 60.0;
    float stretchCents = d * 0.28 + d * d * 0.0018;
    return 440.0 * exp2((k - 69.0 + stretchCents / 1200.0) / 12.0);
}
float p4B(float k){                    // piano3's inharmonicity curve, scaled
    float Braw = 0.000065 * exp((k - 28.0) * 0.028);
    float Bcap = mix(0.0052, 0.0022, smoothstep(72.0, 108.0, k));
    return clamp(Braw, 0.000025, Bcap) * P4_B_SCALE;
}
vec2 p4Single(float time, float f0, float kMidi, float v, int rungs)
{
    if (time<0.0)
        return vec2(0.0);
    float sampleX = 0.;
    float B  = p4B(kMidi);
    float vb = 1.0 + P4_BRIGHT*(v - 0.55);   // >1 brighter strike, <1 mellower
    float bg = pow(vb, float(rungs-1));      // tilt pivots on the FUNDAMENTAL (=1.0):
    for (int i=25-rungs; i<25; i++)          // top partial gets vb^(rungs-1), fundamental
    {                                        // vb^0 — a pure spectral tilt, never a gain.
        float kp = 25.0 - float(i);          // rungs<25 = exact ladder truncation (the
        float fk = f0*kp*sqrt(1.0 + B*kp*kp);// weights of kept partials are unchanged)
        float sf = sin(3.14159265*kp*0.118); // strike-position comb (softened, floor 0.1)
        float sg = mix(1.0, pow(sf*sf, 0.8)*0.9 + 0.1, P4_STRIKE);
        sampleX = (sampleX*1.5+sin(float(i)*2.+6.2831*fk*time)*exp((-1.5-sqrt(fk)/25.)*time)*bg*sg)/2.5;
        bg /= vb;
    }
    return vec2(sampleX);
}
vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 4.5) return vec2(0.0);   // t<0: reverb taps pre-onset
    float f = p4Hz(k);                          // Railsback-stretched fundamental
    // main string + detuned unison partner (~0.5 cents, level-normalized so
    // P4_UNISON 0 == piano2's exact core) + the detuned octave double
    vec2 uni = ( p4Single(t, f, k, v, 25)
               + P4_UNISON*p4Single(t + 0.0013, f*1.00032, k, v, 25) ) / (1.0 + P4_UNISON);
    vec2 o = 0.5*( uni + p4Single(t + 0.005, f*2.0*1.0008, k, v, 25) );
    // aftersound: quiet, slowly-decaying fundamental — the "sings on" second
    // decay stage of a real string (the cascade is the prompt sound)
    o += P4_AFTER*sin(6.2831*f*t + 1.1)*exp(-t*(0.9 + f*0.0005));
    float pan = clamp((k - 60.0)/40.0, -1.0, 1.0)*0.4;
    pan *= smoothstep(80.0, 220.0, f);                 // bass strings sit centered
    o *= vec2(1.0 - pan, 1.0 + pan);
    o *= .5-.5*cos(3.14159265*clamp(t/.0015,0.,1.));   // kill the digital onset click
    o *= exp(-max(0.0, t - s)*P4_RELEASE*(6.0 + f*0.004));  // damper: >= piano2's 6.0, faster in treble
    o *= 1.0 - 0.18*smoothstep(64.0, 88.0, k);         // gentle upper-mid shelf trim
    o *= v * 1.3;
    // hammer: click + thump + felt-noise chiff supply the attack (velocity ONCE)
    float clickF = mix(1500., 2800., smoothstep(24., 108., k));
    o += .010*v*exp(-t*900.)*vec2(sin(6.2831*clickF*t), sin(6.2831*clickF*1.05*t+.3));
    o += .018*v*sin(6.2831*f*t+.1)*exp(-t/.02);
    o += .016*v*(fract(sin(floor(t*44100.)*45.233)*43758.5453)*2.-1.)*exp(-t*650.);
    // damper landing: tiny half-fundamental thud at note-off (dampers end ~key 90)
    float toff = max(0.0, t - s);
    o += .011*v*sin(6.2831*max(f*0.5, 55.0)*toff)*exp(-toff*80.0)*step(s, t)*(1.0-smoothstep(78.0, 90.0, k));
    return o;
}
// lite voice for the reverb taps: truncated ladder, no click/thump — ~3× cheaper wet
vec2 pianoWet(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 4.5) return vec2(0.0);
    float f = p4Hz(k);
    vec2 uni = ( p4Single(t, f, k, v, P4_WET_RUNGS)
               + P4_UNISON*p4Single(t + 0.0013, f*1.00032, k, v, P4_WET_RUNGS) ) / (1.0 + P4_UNISON);
    vec2 o = 0.5*( uni + p4Single(t + 0.005, f*2.0*1.0008, k, v, P4_WET_RUNGS) );
    o += P4_AFTER*sin(6.2831*f*t + 1.1)*exp(-t*(0.9 + f*0.0005));
    float pan = clamp((k - 60.0)/40.0, -1.0, 1.0)*0.4;
    pan *= smoothstep(80.0, 220.0, f);
    o *= vec2(1.0 - pan, 1.0 + pan);
    o *= .5-.5*cos(3.14159265*clamp(t/.0015,0.,1.));
    o *= exp(-max(0.0, t - s)*P4_RELEASE*(6.0 + f*0.004));
    o *= 1.0 - 0.18*smoothstep(64.0, 88.0, k);
    return o * v * 1.3;
}"""

PIANO3_GLSL = r"""// ── sophisticated modal piano ("piano3") — user-supplied, written natively
// against mid2glsl's data format. Railsback stretch tuning, per-string detune
// + per-string inharmonicity, mode splitting (string↔bridge), attack pitch
// glide, bloom triple-decay, HF aging, phantom bass partials, hammer
// transients/felt noise, damper thud. Treble specifically de-cheesed: strike
// position fixed (was "tinny/octave illusion"), B capped in high treble,
// strike combing weakened, duplex shimmer hard-gated.
// Port adaptations (flagged): array reads → sharded accessors; per-note
// pianoStrings wrapper (global band-split compressor + soundboard drone left
// to the master chain); ZERO threaded through the 64-mode loop (de-unroll);
// their TAU/PI/MASTER_GAIN defines dropped (collide with the templates').

#define VEL_CURVE               0.88
#define N_MODES_MAX             64
#define N_MODES_MID             32
#define B_SCALE                 0.000065
#define B_RATE                  0.028
#define B_REF_NOTE              28.0
#define B_MIN                   0.000025
#define B_MAX                   0.0052
#define HAMMER_K_SOFT           0.28
#define HAMMER_K_HARD           0.38
#define HAMMER_VEL_CURVE        0.90
#define HAMMER_NOTCH_DEPTH      0.28
#define HAMMER_NOTCH_WIDTH      4.0
#define HAMMER_NOTCH_MODE_SOFT  3.8
#define HAMMER_NOTCH_MODE_HARD  5.5
#define STRIKE_POS_BASS         0.085
#define STRIKE_POS_TREBLE       0.315
#define STRIKE_NULL_FLOOR_BASS  0.10
#define STRIKE_NULL_FLOOR_TREBLE 0.04
#define STRIKE_GAMMA_BASS       0.75
#define STRIKE_GAMMA_TREBLE     0.95
#define BASS_FUND_BOOST         1.1
#define BASS_BOOST_ROLLOFF      0.32
#define BASS_DECAY_BLOOM        0.22
#define BLOOM_FAST_WEIGHT       0.62
#define BLOOM_FAST_TAU_MULT     0.28
#define BLOOM_MAIN_WEIGHT       0.38
#define BLOOM_HUMP_RISE         1.1
#define BLOOM_HUMP_TAU_MULT     4.2
#define DOUBLE_DECAY_MIX        0.14
#define DOUBLE_DECAY_TAU_MULT   7.0
#define DECAY_TAU_BASS          16.0
#define DECAY_TAU_TREBLE        0.48
#define DECAY_MODE_POWER_BASS   1.45
#define DECAY_MODE_POWER_TREBLE 1.15
#define STR2_START              34.0
#define STR2_END                46.0
#define STR3_START              49.0
#define STR3_END                61.0
#define STR2_AMP                0.80
#define STR3_AMP                0.70
#define DETUNE_CENTS_BASS       0.19
#define DETUNE_CENTS_TREBLE     0.085
#define UNISON_B_VAR            0.024
#define COUPLING_MOD_FREQ       0.38
#define COUPLING_MOD_DEPTH      0.012
#define COUPLING_RISE_RATE      0.8
#define MICRO_DETUNE_CENTS      0.16
#define WOW_RATE_BASE           0.35
#define WOW_RATE_SPREAD         0.02
#define WOW_DEPTH               0.000018
#define SPLIT_LEVEL_BASS        0.10
#define SPLIT_LEVEL_TREBLE      0.26
#define SPLIT_CENTS_MIN         0.25
#define SPLIT_CENTS_MAX         2.10
#define SPLIT_FAST_TAU_MULT     0.28
#define GLIDE_TAU_BASS          0.050
#define GLIDE_TAU_TREBLE        0.020
#define GLIDE_CENTS_BASS        1.8
#define GLIDE_CENTS_TREBLE      0.6
#define HF_LOSS_RATE            0.000020
#define DAMPER_TAU_BASS         0.14
#define DAMPER_TAU_TREBLE       0.065
#define DAMPER_THUD_DECAY       0.007
#define DAMPER_THUD_LEVEL_BASS  0.055
#define DAMPER_THUD_LEVEL_TREBLE 0.013
#define TRANSIENT_DECAY         0.008
#define TRANSIENT_LEVEL_BASS    0.18
#define TRANSIENT_LEVEL_TREBLE  0.010
#define FELT_NOISE_DECAY        65.0
#define FELT_NOISE_LEVEL        0.013
#define FELT_WINDOW_BASS        0.028
#define FELT_WINDOW_TREBLE      0.018
#define PHANTOM_SPREAD          0.007
#define PHANTOM_LEVEL           0.140
#define PHANTOM_TAU_MULT        2.85
#define STEREO_WIDTH            0.44
#define STEREO_BASS_CUTOFF_LO   80.0
#define STEREO_BASS_CUTOFF_HI   180.0

// RAILSBACK STRETCH TUNING
float noteToHz(int n) {
    float note = float(n);
    float d    = note - 60.0;
    float stretchCents = d * 0.28 + d * d * 0.0018;
    return 440.0 * exp2((note - 69.0 + stretchCents / 1200.0) / 12.0);
}

float envAttack(float noteN) {
    // bass ~11ms, treble ~3ms
    return mix(0.011, 0.0022, smoothstep(44.0, 96.0, noteN));
}

float stringHash(float noteN, float n, float idx) {
    return fract(sin(noteN * 127.1 + n * 311.7 + idx * 74.3) * 43758.5453);
}

float partialHz(float f0, float n, float B) {
    return f0 * n * sqrt(1.0 + B * n * n);
}

float inharmonicB(float noteN) {
    float Braw = B_SCALE * exp((noteN - B_REF_NOTE) * B_RATE);
    float Bcap = mix(B_MAX, 0.0022, smoothstep(72.0, 108.0, noteN));
    return clamp(Braw, B_MIN, Bcap);
}

float decayModePower(float noteN) {
    return mix(DECAY_MODE_POWER_BASS, DECAY_MODE_POWER_TREBLE,
               smoothstep(45.0, 92.0, noteN));
}

float modeTau(float n, float noteN) {
    float tauBase = exp(mix(log(DECAY_TAU_BASS), log(DECAY_TAU_TREBLE),
                            smoothstep(21.0, 108.0, noteN)));
    float p    = decayModePower(noteN);
    float tauN = tauBase / pow(n, p);
    return max(tauN, 0.01);
}

float strikePos(float noteN) {
    return mix(STRIKE_POS_BASS, STRIKE_POS_TREBLE, smoothstep(24.0, 96.0, noteN));
}

float strikeFilter(float n, float noteN) {
    float x = strikePos(noteN);
    float s = sin(3.14159265 * n * x);
    float a = s * s;
    float t = smoothstep(30.0, 90.0, noteN);
    float floorv = mix(STRIKE_NULL_FLOOR_BASS, STRIKE_NULL_FLOOR_TREBLE, t);
    float gamma  = mix(STRIKE_GAMMA_BASS, STRIKE_GAMMA_TREBLE, t);
    a = pow(a, gamma);
    return a * (1.0 - floorv) + floorv;
}

float airDamp(float fHz, float noteN) {
    float kk = mix(0.00010, 0.00018, smoothstep(80.0, 108.0, noteN));
    return exp(-max(fHz - 2500.0, 0.0) * kk);
}

float hfLoss(float ageOn, float fHz) {
    return exp(-ageOn * HF_LOSS_RATE * fHz);
}

float strCount2(float noteN) { return smoothstep(STR2_START, STR2_END, noteN); }
float strCount3(float noteN) { return smoothstep(STR3_START, STR3_END, noteN); }

vec2 partialPan(float fn, float noteN) {
    float notePos = clamp((noteN - 64.5) / 43.5, -1.0, 1.0) * STEREO_WIDTH;
    float panFade = smoothstep(STEREO_BASS_CUTOFF_LO, STEREO_BASS_CUTOFF_HI, fn);
    float panPos  = notePos * panFade;
    return vec2(sqrt(clamp(0.5 - panPos * 0.5, 0.0, 1.0)),
                sqrt(clamp(0.5 + panPos * 0.5, 0.0, 1.0)));
}

float bloomDecay(float ageOn, float tau, float bassness) {
    float fast  = exp(-ageOn / max(tau * BLOOM_FAST_TAU_MULT, 0.001));
    float main  = exp(-ageOn / tau);
    float bloom = 1.0 + bassness * BASS_DECAY_BLOOM
                      * (1.0 - exp(-ageOn * BLOOM_HUMP_RISE))
                      * exp(-ageOn / max(tau * BLOOM_HUMP_TAU_MULT, 0.001));
    float slowTail = bassness * DOUBLE_DECAY_MIX
                   * exp(-ageOn / max(tau * DOUBLE_DECAY_TAU_MULT, 0.001));
    float d = (fast * BLOOM_FAST_WEIGHT + main * BLOOM_MAIN_WEIGHT + slowTail) * bloom;
    float dAt0 = BLOOM_FAST_WEIGHT + BLOOM_MAIN_WEIGHT + bassness * DOUBLE_DECAY_MIX;
    return d / max(dAt0, 0.001);
}

float modeSplitCents(float noteN, float n) {
    float reg = smoothstep(40.0, 92.0, noteN);
    float nn  = smoothstep(2.0, 14.0, n);
    return mix(SPLIT_CENTS_MIN, SPLIT_CENTS_MAX, reg) * nn;
}

float splitLevel(float noteN) {
    return mix(SPLIT_LEVEL_BASS, SPLIT_LEVEL_TREBLE, smoothstep(36.0, 92.0, noteN));
}

float hammerSpectrum(float n, float vel, float noteN) {
    float kk       = mix(HAMMER_K_SOFT, HAMMER_K_HARD, pow(vel, HAMMER_VEL_CURVE));
    float rolloff  = exp(-kk * pow(n - 1.0, 1.20));
    float notchMode = mix(HAMMER_NOTCH_MODE_SOFT, HAMMER_NOTCH_MODE_HARD, vel);
    float notch     = 1.0 - HAMMER_NOTCH_DEPTH
                          * exp(-pow(n - notchMode, 2.0) / HAMMER_NOTCH_WIDTH);
    float bassBoost = mix(BASS_FUND_BOOST, 1.0, smoothstep(28.0, 65.0, noteN));
    float regWeight = mix(bassBoost * exp(-BASS_BOOST_ROLLOFF * (n - 1.0)), 1.0,
                          smoothstep(50.0, 80.0, noteN));
    float A = rolloff * notch * regWeight;
    if (n > 1.5) {
        float strikeAmt = mix(1.0, 0.30, smoothstep(64.0, 104.0, noteN));
        A *= mix(1.0, strikeFilter(n, noteN), strikeAmt);
    } else {
        A *= 1.0 + 0.35 * smoothstep(86.0, 108.0, noteN);
    }
    return A;
}

float phantomPartials(float ageOn, float f0, float vel, float noteN, float tauFund) {
    if (noteN > 48.0) return 0.0;
    float bassness = 1.0 - smoothstep(28.0, 48.0, noteN);
    float env      = vel * exp(-ageOn / max(tauFund * PHANTOM_TAU_MULT, 0.001));
    float sig      = 0.0;
    for (int k = 1; k <= 4; k++) {
        float fp = f0 * 2.0 * (1.0 + PHANTOM_SPREAD * (float(k) - 2.5));
        float ph = stringHash(noteN, float(k), 55.0);
        sig += sin(TAU * fract(fp * ageOn + ph)) / float(k + 1);
    }
    return sig * env * PHANTOM_LEVEL * bassness;
}

vec2 pianoModal(float ageOn, float f0, float vel, float noteN, int ZERO)
{
    float B0      = inharmonicB(noteN);
    float nyquist = MIDI_SAMPLE_RATE * 0.49;
    float bassness = 1.0 - smoothstep(40.0, 76.0, noteN);

    float microCents = (stringHash(noteN, 0.0, 99.0) - 0.5) * 2.0 * MICRO_DETUNE_CENTS;
    float microMult  = pow(2.0, microCents / 1200.0);
    float wowRate   = WOW_RATE_BASE + WOW_RATE_SPREAD * stringHash(noteN, 0.0, 123.0);
    float wowDrift  = 1.0 + WOW_DEPTH * sin(TAU * wowRate * ageOn + noteN * 0.1);

    float gTau   = mix(GLIDE_TAU_BASS, GLIDE_TAU_TREBLE, smoothstep(36.0, 92.0, noteN));
    float gCent  = mix(GLIDE_CENTS_BASS, GLIDE_CENTS_TREBLE, smoothstep(36.0, 92.0, noteN));
    float gAmt   = gCent * clamp(vel, 0.0, 1.0);
    float gRatio = pow(2.0, (gAmt * exp(-ageOn / max(gTau, 0.001))) / 1200.0);

    float fBase = f0 * microMult * wowDrift * gRatio;

    float detC = mix(DETUNE_CENTS_BASS, DETUNE_CENTS_TREBLE, smoothstep(33.0, 84.0, noteN));
    float c1 = detC * (stringHash(noteN, 0.0, 901.0) - 0.5) * 2.0;
    float c2 = detC * (stringHash(noteN, 0.0, 902.0) - 0.5) * 2.0;
    float c3 = detC * (stringHash(noteN, 0.0, 903.0) - 0.5) * 2.0;

    float f1 = fBase * pow(2.0, c1 / 1200.0);
    float f2 = fBase * pow(2.0, c2 / 1200.0);
    float f3 = fBase * pow(2.0, c3 / 1200.0);

    float B1 = B0 * (1.0 + UNISON_B_VAR * (stringHash(noteN, 0.0, 910.0) - 0.5));
    float B2 = B0 * (1.0 + UNISON_B_VAR * (stringHash(noteN, 0.0, 911.0) - 0.5));
    float B3 = B0 * (1.0 + UNISON_B_VAR * (stringHash(noteN, 0.0, 912.0) - 0.5));

    float str2 = strCount2(noteN);
    float str3 = strCount3(noteN);

    float couplingEnv  = 1.0 - exp(-ageOn * COUPLING_RISE_RATE);
    float couplingBase = stringHash(noteN, 0.0, 66.0) * TAU;

    int nModes = (noteN < 50.0) ? N_MODES_MAX : N_MODES_MID;

    vec2 sig = vec2(0.0);

    for (int i = ZERO; i < N_MODES_MAX; i++) {
        if (i >= nModes) break;

        float n = float(i + 1);
        float tau = modeTau(n, noteN);
        float A = hammerSpectrum(n, vel, noteN);
        if (A < 0.0002 && i > 6) continue;

        float dMain = bloomDecay(ageOn, tau, bassness);

        float fn1 = partialHz(f1, n, B1);
        if (fn1 >= nyquist) break;

        float ampBase = A * vel;
        float amp1 = ampBase * dMain;
        amp1 *= airDamp(fn1, noteN) * hfLoss(ageOn, fn1);

        float ph1 = stringHash(noteN, n, 4.0);
        vec2  pan1 = partialPan(fn1, noteN);
        sig += amp1 * sin(TAU * (fract(fn1 * ageOn) + ph1)) * pan1;

        if (n >= 2.0) {
            float sc  = modeSplitCents(noteN, n);
            float rat = pow(2.0, sc / 1200.0);
            float fnS = fn1 * rat;
            if (fnS < nyquist) {
                float phS  = stringHash(noteN, n, 204.0);
                float tauS = max(tau * SPLIT_FAST_TAU_MULT, 0.01);
                float dS   = bloomDecay(ageOn, tauS, bassness);
                float aS   = ampBase * splitLevel(noteN) * dS;
                aS *= airDamp(fnS, noteN) * hfLoss(ageOn, fnS);
                sig += aS * sin(TAU * (fract(fnS * ageOn) + phS)) * pan1;
            }
        }

        if (str2 > 0.001) {
            float fn2 = partialHz(f2, n, B2);
            if (fn2 < nyquist) {
                float tau2 = tau * (1.0 + 0.045 * (stringHash(noteN, n, 12.0) - 0.5));
                float d2   = bloomDecay(ageOn, tau2, bassness);
                float modPhase2 = ageOn * COUPLING_MOD_FREQ + couplingBase;
                float coupling2 = 1.0 + COUPLING_MOD_DEPTH * sin(modPhase2) * couplingEnv;
                float amp2 = ampBase * d2 * coupling2;
                amp2 *= STR2_AMP * str2;
                amp2 *= airDamp(fn2, noteN) * hfLoss(ageOn, fn2);
                float ph2 = stringHash(noteN, n, 1.0);
                vec2  pan2 = partialPan(fn2, noteN);
                sig += amp2 * sin(TAU * (fract(fn2 * ageOn) + ph2)) * pan2;
                if (n >= 2.0) {
                    float sc  = modeSplitCents(noteN, n);
                    float rat = pow(2.0, sc / 1200.0);
                    float fnS = fn2 * rat;
                    if (fnS < nyquist) {
                        float phS  = stringHash(noteN, n, 205.0);
                        float tauS = max(tau2 * SPLIT_FAST_TAU_MULT, 0.01);
                        float dS   = bloomDecay(ageOn, tauS, bassness);
                        float aS   = ampBase * splitLevel(noteN) * dS * STR2_AMP * str2;
                        aS *= airDamp(fnS, noteN) * hfLoss(ageOn, fnS);
                        sig += aS * sin(TAU * (fract(fnS * ageOn) + phS)) * pan2;
                    }
                }
            }
        }

        if (str3 > 0.001) {
            float fn3 = partialHz(f3, n, B3);
            if (fn3 < nyquist) {
                float tau3 = tau * (1.0 + 0.045 * (stringHash(noteN, n, 13.0) - 0.5));
                float d3   = bloomDecay(ageOn, tau3, bassness);
                float modPhase3 = ageOn * COUPLING_MOD_FREQ + couplingBase + 1.3;
                float coupling3 = 1.0 + COUPLING_MOD_DEPTH * sin(modPhase3) * couplingEnv;
                float amp3 = ampBase * d3 * coupling3;
                amp3 *= STR3_AMP * str3;
                amp3 *= airDamp(fn3, noteN) * hfLoss(ageOn, fn3);
                float ph3 = stringHash(noteN, n, 3.0);
                vec2  pan3 = partialPan(fn3, noteN);
                sig += amp3 * sin(TAU * (fract(fn3 * ageOn) + ph3)) * pan3;
                if (n >= 2.0) {
                    float sc  = modeSplitCents(noteN, n);
                    float rat = pow(2.0, sc / 1200.0);
                    float fnS = fn3 * rat;
                    if (fnS < nyquist) {
                        float phS  = stringHash(noteN, n, 206.0);
                        float tauS = max(tau3 * SPLIT_FAST_TAU_MULT, 0.01);
                        float dS   = bloomDecay(ageOn, tauS, bassness);
                        float aS   = ampBase * splitLevel(noteN) * dS * STR3_AMP * str3;
                        aS *= airDamp(fnS, noteN) * hfLoss(ageOn, fnS);
                        sig += aS * sin(TAU * (fract(fnS * ageOn) + phS)) * pan3;
                    }
                }
            }
        }
    }

    if (bassness > 0.01) {
        float tauFund = modeTau(1.0, noteN);
        sig += vec2(phantomPartials(ageOn, f1, vel, noteN, tauFund));
    }

    return sig;
}

float hammerTransient(float ageOn, float vel, float noteN) {
    float env = exp(-ageOn / TRANSIENT_DECAY) * vel;
    if (env < 0.001) return 0.0;
    float hz      = noteToHz(int(noteN));
    float B       = inharmonicB(noteN);
    float nyquist = MIDI_SAMPLE_RATE * 0.49;
    float sig = 0.0;
    for (int k = 10; k <= 16; k++) {
        float fk = partialHz(hz, float(k), B);
        if (fk >= nyquist) break;
        sig += sin(TAU * fract(fk * ageOn)) / float(k);
    }
    float level = mix(TRANSIENT_LEVEL_BASS, TRANSIENT_LEVEL_TREBLE,
                      smoothstep(48.0, 92.0, noteN));
    return sig * env * level;
}

float hammerFeltNoise(float ageOn, float vel, float noteN) {
    float feltWindow = mix(FELT_WINDOW_BASS, FELT_WINDOW_TREBLE,
                           smoothstep(40.0, 72.0, noteN));
    if (ageOn > feltWindow) return 0.0;
    float env = exp(-ageOn * FELT_NOISE_DECAY) * vel * vel;
    if (env < 0.001) return 0.0;
    float baseFq = mix(320.0, 800.0, smoothstep(36.0, 76.0, noteN));
    float stepFq = mix(380.0, 620.0, smoothstep(36.0, 76.0, noteN));
    float sig = 0.0;
    for (int i = 0; i < 7; i++) {
        float h = stringHash(noteN, float(i), 88.0);
        sig += sin(TAU * (baseFq + float(i) * stepFq + h * 400.0) * ageOn);
    }
    float level = FELT_NOISE_LEVEL * mix(1.35, 0.75, smoothstep(30.0, 85.0, noteN));
    return sig * env * level;
}

float damperThud(float ageOff, float vel, float noteN) {
    float hasDamper = 1.0 - smoothstep(78.0, 84.0, noteN);
    if (hasDamper < 0.01) return 0.0;
    float env = exp(-ageOff / DAMPER_THUD_DECAY) * vel;
    if (env < 0.001) return 0.0;
    float hz      = noteToHz(int(noteN));
    float B       = inharmonicB(noteN);
    float nyquist = MIDI_SAMPLE_RATE * 0.49;
    float sig = 0.0;
    for (int k = 1; k <= 5; k++) {
        float fk = partialHz(hz, float(k), B) * (1.0 + float(k) * 0.012);
        if (fk >= nyquist) break;
        sig += sin(TAU * fract(fk * ageOff)) / float(k);
    }
    float level = mix(DAMPER_THUD_LEVEL_BASS, DAMPER_THUD_LEVEL_TREBLE,
                      smoothstep(36.0, 72.0, noteN));
    return sig * env * level * hasDamper;
}

float dampedRelease(float ageOff, float heldT, float noteN) {
    float tauDamped  = mix(DAMPER_TAU_BASS, DAMPER_TAU_TREBLE,
                           smoothstep(48.0, 88.0, noteN));
    float tauFund    = modeTau(1.0, noteN);
    float levelAtRel = exp(-heldT / tauFund);
    return levelAtRel * exp(-ageOff / tauDamped)
         * clamp(heldT / envAttack(noteN), 0.0, 1.0);
}

// per-note wrapper: full modal voice + transients + damper thud. The original's
// global split-band compressor / soundboard drone are handled by the template's
// poly gain + softLimit instead.
vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    if (t < 0.0 || t > 8.0) return vec2(0.0);
    float noteN = k;
    float vv = pow(clamp(v, 0.0, 1.0), VEL_CURVE);
    float f0 = noteToHz(int(k));
    float gateAmp = (t < s) ? clamp(t / envAttack(noteN), 0.0, 1.0)
                            : dampedRelease(t - s, s, noteN);
    vec2 osc = (gateAmp < 0.0001) ? vec2(0.0)
             : pianoModal(t, f0, vv, noteN, ZERO) * gateAmp;
    float transRamp = clamp(t / 0.002, 0.0, 1.0);
    float monoOn  = hammerTransient(t, vv, noteN) + hammerFeltNoise(t, vv, noteN);
    float monoOff = (t >= s) ? damperThud(t - s, vv, noteN) : 0.0;
    vec2 tPan = partialPan(500.0, noteN);
    vec2 y = osc + (monoOn * transRamp + monoOff) * tPan;
    return y * 0.12;
}"""

PIANO2_GLSL = r"""// ── cascade piano ("piano2") — user-ranked "best sounding piano synth so far":
// 25 DESCENDING partials (f = 25·f0 → f0) fed through a cascaded smoothing
// feedback (sampleX = (1.5·sampleX + partial)/2.5 — a one-pole blur down the
// partial ladder), intrinsic decay exp((−1.5−√f/25)·t), plus an octave-up
// double struck 5 ms late. Adaptations (flagged): the original balance formula
// (0.75 − note/12) diverges outside its 1.5-octave demo range → tame MIDI pan;
// added note-off damper + velocity (the original had neither).
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
vec2 p2Single(float time, float note)
{
    note += 15.;
    if (time<0.0)
        return vec2(0.0);
    float sampleX = 0.;
    float mult = pow(2.0,note/12.0);
    float f = 25.*55.*mult;
    for (int i=0; i<(iChannelResolution[0].x<0.0?99999:25); i++)   // de-unroll
    {
        sampleX = (sampleX*1.5+sin(float(i)*2.+6.2831*f*time)*exp((-1.5-sqrt(f)/25.)*time))/2.5;
        f -= 55.*mult;
    }
    return vec2(sampleX);
}
vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 6.0) return vec2(0.0);   // t<0: reverb taps pre-onset
    float note = k - 48.0;        // MIDI k → the shader's 55Hz-ladder domain
    vec2 o = 0.5*( p2Single(t, note) + p2Single(t + 0.005, note + 12.0) );
    float pan = clamp((k - 60.0)/40.0, -1.0, 1.0)*0.4;
    o *= vec2(1.0 - pan, 1.0 + pan);
    o *= exp(-max(0.0, t - s)*6.0);           // damper
    return o * v * 1.3;
}
vec2 pianoWet(float k, float t, float s, float v, int ZERO){ return pianoStrings(k, t, s, v, ZERO); }"""

EPIANO2_GLSL = r"""// ── Thibault FM e-piano ("epiano2") — user-ranked "best magical organ
// japanese roland style": glassy 14×-modulated slightly-sharp attack
// (sin-distorted at high nuance) + perfectly-in-tune FM body whose index is
// nuance- and low-note-boosted; note-dependent attack pan.
// By Alexis THIBAULT — CC BY-NC 4.0, attribution required.
// Adaptations (flagged): nuance driven by velocity; note-off damper added
// (original notes ring out freely).
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
#define ep2sin(x,m) sin(TAU*(x)+(m))
vec2 ep2Voice(float freq, float t, float nuance)
{
    vec2 f0 = vec2(freq*0.998, freq*1.002);
    vec2 glass = ep2sin((f0+3.)*t, ep2sin(14.*f0*t,0.) * exp(-30.*t) * nuance) * exp(-4.*t) * nuance;
    glass = sin(glass); // Distort at high nuances
    vec2 body = ep2sin(f0*t, ep2sin(f0*t,0.) * exp(-0.5*t) * nuance * pow(440./f0.x, 0.5)) * exp(-t) * nuance;
    float panDir = clamp(log2(freq/400.)/2., -1., 1.);
    vec2 pan = normalize(vec2(0.5-0.5*panDir, 0.5+0.5*panDir));
    return (glass*pan + body) * 0.05 * smoothstep(0.,0.001,t);
}
vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    if(t < 0.0 || t > 6.0) return vec2(0.0);   // t<0: reverb taps pre-onset
    vec2 o = ep2Voice(note2freq(k), t, 0.4 + 1.2*v);
    o *= exp(-max(0.0, t - s)*7.0);            // damper
    return o * 3.2;
}
vec2 pianoWet(float k, float t, float s, float v, int ZERO){ return pianoStrings(k, t, s, v, ZERO); }"""

EPIANO_GLSL = r"""// ── hybrid FM e-piano ("epiano") — scanned/hybridized from the pasted models:
// DX7-style FM core (1:1 body op with velocity-decaying index + 14:1 tine
// ping), carter feldman's piano (=penguinPiano) inharmonicity / dual-string
// detune / two-stage decay / note-off damper, newPiano's hammer click+thump,
// and an EP stereo tremolo shimmer.
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
vec2 pianoStrings(float k, float t, float s, float v, int ZERO){
    if(t < 0.0 || t > 6.0) return vec2(0.0);   // t<0: reverb taps pre-onset (exp terms would blow up)
    float f = note2freq(k);
    float B = clamp(.00007*pow(1.4, log2(max(f,27.5)/27.5)), .00003, .035);
    float det = .00025;
    float tau1 = .55 + .9/(1.+f*.0018), tau2 = 6. + 14./(1.+f*.0008);
    float env  = .82*exp(-t/tau1) + .18*exp(-t/tau2);
    float att  = .5 - .5*cos(3.14159265359*clamp(t/.012, 0., 1.));
    float rel  = exp(-max(0., t-s)*(5.5 + f*.003));
    float I1 = (.55 + 1.9*v)*exp(-t*4.2) + .32;      // body FM index (bark -> warm)
    float I2 = (2.2 + 2.4*v)*exp(-t*22.0);           // tine index (metallic strike)
    vec2 o = vec2(0.);
    for(int q = ZERO; q < 2; q++){                   // two detuned "tines"
        float fq = f*(1. + ((q==0)?-det:det))*sqrt(1.+B);
        float ph = TAU*fq*t + float(q)*1.7;
        float body = sin(ph + I1*fbsin(ph, .25));
        float tine = sin(ph + I2*sin(14.0*ph + float(q)));
        float sub  = sin(.5*ph)*smoothstep(300., 60., f)*.35*exp(-t*1.2);
        vec2 panq = (q==0) ? vec2(.56,.44) : vec2(.44,.56);
        o += panq*( body*.72 + tine*.5*exp(-t*9.0) + sub )*((q==0)?.58:.42);
    }
    float clickF = mix(1500., 2800., smoothstep(24., 108., k));
    o += .012*v*exp(-t*900.)*vec2(sin(TAU*clickF*t), sin(TAU*clickF*1.05*t+.3));
    o += .02*v*sin(TAU*f*t+.1)*exp(-t/.02);
    float tr = .5 + .5*sin(TAU*4.7*t);               // EP stereo tremolo
    o *= vec2(1.-.22*tr, 1.-.22*(1.-tr));
    o *= att*env*rel*v;
    return tanh(o*1.6)*.6;
}
vec2 pianoWet(float k, float t, float s, float v, int ZERO){ return pianoStrings(k, t, s, v, ZERO); }"""

ORGAN_GLSL = r"""// ── organ ("organ") — spalmer's Melody Maker `ins()` stacked-detuned-octave
// instrument, ported verbatim (the x2.01 layer stack IS the sound; the user\n// ranked the 3-layer i=-0..2 form 'best organ').
// Adaptations: o zero-init (the original read an uninitialized var) and a
// note-off release (the original had no note gate; player notes must stop).
const float tau = radians(360.);
vec2 ins(float t, float f)
{
    vec2 o = vec2(0);
    float g = 1., F = f;
    for (float i = -0.; i <= 2.; ++i) {   // "best organ" = 3 layers (-2. start = brighter 5-layer variant)
        float u = t - i;
        o += vec2(
        sin(tau * F * u) // sine doesn't need an attack envelope
         * g
         );
        F *= 2.01; // perfect 2.0 winds up cancelling the tone out somehow
        g *= .7;   // controls timbre
    }
    o *= exp2(-.004 * t * f);
    o *= 1. - exp2(-5e1 / sqrt(f)); // slower attack for high frequencies
    o *= exp2(-.001 * f);           // attenuate higher frequencies more
    return o;
}
float note2freq(float x){ return 440.0*exp2((x-69.0)/12.0); }
vec2 pianoStrings(float k, float t, float s, float v, int ZERO){
    if(t < 0.0 || t > 6.0) return vec2(0.0);   // t<0: reverb taps pre-onset (exp terms would blow up)
    vec2 o = ins(t, note2freq(k)) * v;
    return o * exp(-max(0., t - s)*10.0) * .5;
}
vec2 pianoWet(float k, float t, float s, float v, int ZERO){ return pianoStrings(k, t, s, v, ZERO); }"""

PIANOMAN_GLSL = r"""// ── "piano man" piano (user-supplied ShaderToy model) ──────────────────────
// penguinPiano (low register) crossfaded into newPiano (a heavily modified
// version of iq's piano, https://www.shadertoy.com/view/t3cfWf), minus an
// antiResLayer, plus 10 soundboard body modes excited per note (in mainSound).
// Notes are stereo with envelope, note-off damping and per-note tanh built in
// — the ADSR/reverb path is bypassed for VOICE==1.
const float kTau = 6.28318530718;

float note2freq( in float x )
{
    return 440.0*exp2((x-69.0)/12.0);
}

float hash12( vec2 p )
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.545);
}

float bandNoise( float t, float seed, float rate )
{
    float tt = t * rate;
    float i  = floor(tt);
    float f  = fract(tt);
    float a  = hash12(vec2(i,       seed)) - 0.5;
    float b  = hash12(vec2(i + 1.0, seed)) - 0.5;
    float sm = f*f*(3.0 - 2.0*f);
    return mix(a, b, sm);
}

vec4 bodyMode( int i )
{
    if( i==0 ) return vec4(  62.0, 1.00,  4.0, 0.0);
    if( i==1 ) return vec4(  88.0, 0.85,  4.5, 1.9);
    if( i==2 ) return vec4( 117.0, 0.75,  5.5, 0.7);
    if( i==3 ) return vec4( 155.0, 0.65,  7.0, 1.3);
    if( i==4 ) return vec4( 214.0, 0.50,  9.0, 2.4);
    if( i==5 ) return vec4( 296.0, 0.38, 11.0, 0.5);
    if( i==6 ) return vec4( 418.0, 0.28, 14.0, 1.9);
    if( i==7 ) return vec4( 612.0, 0.20, 19.0, 2.7);
    if( i==8 ) return vec4( 890.0, 0.13, 26.0, 0.9);
               return vec4(1340.0, 0.08, 35.0, 2.2);
}

vec2 newPiano( float k, float t, float s, float v, int ZERO )
{
    if( t > 5.0 ) return vec2(0.0);

    float nd = k - 66.0;
    float d  = 0.00035 * nd * nd * nd;
    float f  = note2freq(k + d / 100.0);

    f *= 1.0 + 0.00060 * v * exp(-t * 112.0);

    float B = 0.000060 * pow(1.345, log2(max(f, 27.5) / 27.5));
    B = clamp(B, 2.7e-5, 1.0e-3);

    float tauH = (0.00040 + 0.00445 * clamp((80.0 - k) / 60.0, 0.0, 1.0))
               * (1.0 - 0.35 * v);

    vec2 body = vec2(0.0);
    for( int n = 1 + ZERO; n <= 22; n++ )
    {
        float n1 = float(n);
        float n2 = n1 * n1;

        float fn = n1 * f * sqrt(1.0 + B * n2);
        if( fn > 15600.0 ) break;

        float am = 1.0;

        float x = fn * tauH;
        am *= 1.0 / (1.0 + x * x);

        float strike = 0.130;
        am *= abs(sin(3.14159265 * n1 * strike));

        float re = 0.945
                 + 0.375 * clamp((60.0 - k) / 24.0, 0.0, 1.0)
                 + 0.145 * (1.0 - v);
        am /= pow(n1, re);

        float brightEnv = exp(-t / 0.0275);
        am *= 1.0 + (0.26 + 0.43 * v) * brightEnv * (1.0 - 1.0 / sqrt(n1));

        float logF = log(fn);
        float formant = 0.85 + 0.46 * exp(-pow((logF - 6.16) / 0.40, 2.0));
        am *= formant;

        float hfCut = 1.0 / (1.0 + pow(fn / 5000.0, 2.5));
        am *= 0.27 + 0.73 * hfCut;

        if( n == 1 ) am *= 0.90;
        else if( (n & 1) == 1 ) am *= 0.83;

        float ka = 0.054 * sqrt(f);
        float kbP = (0.73 + 0.085 * f / 100.0) * 1e-6;
        float dampPrompt = (ka + kbP * fn * fn) * (1.0 + 0.086 * log(max(n1, 1.0)));
        float dampAfter  = 0.0215 * f + 0.00044 * fn;

        vec3 ts = vec3(2.0, 1.9, 2.1) * t + 3.35 * max(t - s, 0.0);
        vec3 atPrompt = exp(-dampPrompt * 0.75 * ts);
        vec3 atAfter  = exp(-dampAfter  * 0.103 * ts);
        vec3 at = 0.57 * atPrompt + 0.43 * atAfter;

        float pr = 1.0 - exp(-t / max(0.40 * tauH + 0.00002 * n1, 5e-5));
        at *= pr;

        float bassMix = 1.0 - smoothstep(45.0, 70.0, k);
        float phi = fract(n1 * 0.61803398875) * 0.46
                  + bassMix * hash12(vec2(k + 13.7, n1 + 2.3)) * 0.60;

        float detuneBase = mix(0.0025, 0.00105, smoothstep(36.0, 96.0, k));
        float de1 = 1.0 - detuneBase * (0.84 + 0.23 * v);
        float de3 = 1.0 + detuneBase * (0.84 + 0.23 * v);

        float s1 = sin(kTau * fn * de1 * t + phi       ) * at.x;
        float s2 = sin(kTau * fn        * t + phi + 0.2) * at.y;
        float s3 = (k > 53.0) ? sin(kTau * fn * de3 * t + phi + 0.4) * at.z : 0.0;

        float invN = (k > 53.0) ? (1.0 / 3.0) : 0.5;
        body.x += v * am * invN * (1.01 * s1 + 0.99 * s2 + 1.00 * s3);
        body.y += v * am * invN * (0.99 * s1 + 1.01 * s2 + 1.00 * s3);
    }

    body += 0.032 * v * clamp((60.0 - k) / 14.0, 0.0, 1.0)
          * sin(kTau * f * 15.9 * t) * exp(-18.5 * t);

    body *= 0.60;

    float velAtk = mix(0.014, 0.0029, v)
                 * mix(1.18, 0.58, smoothstep(36.0, 96.0, k));
    body *= 1.0 - exp(-t / velAtk);

    vec2 trans = vec2(0.0);

    float thwackEnv = v * exp(-t / 0.0048);
    float nL1 = bandNoise(t,           k * 0.31 + 1.7, 4500.0);
    float nR1 = bandNoise(t + 0.00009, k * 0.31 + 4.9, 4500.0);
    vec2 hammerNoise = vec2(nL1, nR1);
    float noiseGain = 0.115 + 0.065 * smoothstep(36.0, 96.0, k);
    trans += thwackEnv * noiseGain * hammerNoise;

    float clickEnv = v * exp(-t * 1300.0);
    float clickF   = mix(1350.0, 2600.0, smoothstep(24.0, 108.0, k));
    trans += 0.010 * clickEnv * vec2(
        sin(kTau * clickF * t),
        sin(kTau * clickF * 1.05 * t + 0.35)
    );

    trans += 0.012 * v * sin(kTau * f * t + 0.1) * exp(-t / 0.018);

    vec2 y = body + trans;
    return tanh(y * 2.0) / 2.0;
}

// penguinPiano = "piano by carter feldman (@cmpeq on twitter/x),
// creative commons share alike attribution" (user-identified source)
vec2 penguinPiano(float k, float t, float s, float v, int ZERO){
  if(t > 5.0) return vec2(0.0);
  v = clamp(v, 0.0, 1.0);
  float f = note2freq(k);

  float B=.00007*pow(1.4,log2(max(f,27.5)/27.5));
  B=clamp(B,.00003,.035);
  float att=.5-.5*cos(3.14159265359*clamp(t/.015,0.,1.));
  float tau1=.55+.9/(1.+f*.0018),tau2=6.+14./(1.+f*.0008);
  float env=.82*exp(-t/tau1)+.18*exp(-t/tau2);
  float ns=f<115.?1.:f<290.?2.:3.;
  float det=.00025;
  vec2 sig=vec2(0.);
  for(int i=1+ZERO;i<=14;i++){
    float n=float(i);
    float fn=f*n*sqrt(1.+B*n*n);
    if(fn>13000.)break;
    float ph=fract(n*.618033988749)*.5;
    float a;
    if(n<1.5)a=1.;else if(n<2.5)a=.75;else if(n<3.5)a=.55;else if(n<4.5)a=.4;else if(n<5.5)a=.3;
    else if(n<6.5)a=.22;else if(n<7.5)a=.16;else if(n<8.5)a=.12;else a=.1/pow(n-7.,.5);
    if(n>3.)a*=.88625;
    a*=exp(-t*(.4+n*.12));
    a*=pow(n, -(1.0-v)*0.8);
    if(ns<1.5)sig+=vec2(sin(kTau*fn*t+ph))*a;
    else if(ns<2.5){
      float f1=f*(1.-det)*n*sqrt(1.+B*n*n),f2=f*(1.+det)*n*sqrt(1.+B*n*n);
      sig.x+=(sin(kTau*f1*t+ph)*.55+sin(kTau*f2*t+ph+.3)*.45)*a;
      sig.y+=(sin(kTau*f1*t+ph)*.45+sin(kTau*f2*t+ph+.3)*.55)*a;
    }else{
      float f1=f*(1.-det)*n*sqrt(1.+B*n*n),f2=fn,f3=f*(1.+det)*n*sqrt(1.+B*n*n);
      sig.x+=(sin(kTau*f1*t+ph)*.36+sin(kTau*f2*t+ph+.2)*.36+sin(kTau*f3*t+ph+.4)*.28)*a;
      sig.y+=(sin(kTau*f1*t+ph)*.28+sin(kTau*f2*t+ph+.2)*.36+sin(kTau*f3*t+ph+.4)*.36)*a;
    }
  }
  sig*=.55/max(ns,1.);
  float be=exp(-t);
  float body=sin(kTau*f*t+.1)*.18*be+sin(kTau*f*.5*t+.15)*.1*be*smoothstep(300.,60.,f)+sin(kTau*f*2.*t+.05)*.05*be*exp(-t*2.);
  sig+=vec2(body)*.75*env;
  sig*=att*env;
  sig*=exp(-max(0.,t-s)*(4.5+f*.0025));
  return tanh(sig*1.02)*.62*v;
}

// makes it sound more "alive"
vec2 antiResLayer(float k, float t, float s, float v, int ZERO)
{
    if (t > 1.2) return vec2(0.0);

    float f = note2freq(k);

    float atk  = exp(-t / 0.030);
    float body = exp(-t / 0.22) * exp(-max(0.0, t - s) * 5.0);

    float bassMid = 1.0 - smoothstep(62.0, 96.0, k);

    float B = 0.000055 * pow(1.36, log2(max(f, 27.5) / 27.5));
    B = clamp(B, 2.5e-5, 8.0e-4);

    vec2 y = vec2(0.0);

    for (int n = 1+ZERO; n <= 5; n++)
    {
        float n1 = float(n);
        float fn = n1 * f * sqrt(1.0 + B * n1 * n1);

        float a = 1.0 / pow(n1, 1.15);
        a *= 0.7 + 0.3 * bassMid;

        float ph = 6.28318530718 * hash12(vec2(k + 0.71, n1 + 2.13));

        float d = mix(0.0018, 0.0008, smoothstep(36.0, 96.0, k));

        float s1 = sin(kTau * fn * (1.0 - d) * t + ph);
        float s2 = sin(kTau * fn * (1.0 + d) * t + ph + 0.35);

        float mono = 0.5 * (s1 + s2);
        float wid  = 0.18 * (s1 - s2);

        float e = exp(-t * (16.0 + 2.8 * n1));

        y += a * e * vec2(mono + wid, mono - wid);
    }

    float nL = 0.7 * bandNoise(t,           k * 0.37 + 1.1, 4200.0)
             + 0.3 * bandNoise(t,           k * 0.19 + 8.7, 7600.0);
    float nR = 0.7 * bandNoise(t + 0.00011, k * 0.37 + 4.4, 4200.0)
             + 0.3 * bandNoise(t + 0.00011, k * 0.19 + 9.9, 7600.0);

    vec2 noisePart = vec2(nL, nR);

    float ringF = min(900.0, 3.2 * f);
    vec2 ring = vec2(
        sin(kTau * ringF * t + 0.4),
        sin(kTau * ringF * 1.01 * t + 0.9)
    ) * exp(-t * 11.0);

    float gainAtk  = (0.010 + 0.020 * v) * (0.6 + 0.4 * bassMid);
    float gainBody = 0.012 * bassMid;

    return gainAtk * atk * y
         + gainAtk * 0.55 * atk * noisePart
         + gainBody * body * ring;
}

vec2 pianoStrings(float k, float t, float s, float v, int ZERO)
{
    // blend between penguin and new piano
    const float kLo = 51.0; // penguin
    const float kHi = 72.0; // new piano
    float w = smoothstep(kLo, kHi, k);

    float a  = w * 1.5707963;
    float gP = cos(a);
    float gN = sin(a);

    vec2 base = gP * penguinPiano(k, t, s, v, ZERO) + gN * newPiano(k, t, s, v, ZERO);

    return base - antiResLayer(k, t, s, v, ZERO);
}"""

PLAYER_CODE = r"""
// ---- self-contained block-indexed player (brass) ------------------
#define TAU 6.28318530718
float NoteToHz(int n){ return 440.0*exp2((float(n)-69.0)/12.0); }
uint  onTick(uint i){ return midiOM(i)&0xFFFFu; }
uint  meta16(uint i){ return midiOM(i)>>16; }
uint  durTick(uint i){ uint w=midiDur(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
int   metaNote(uint m){ return MIDI_NOTE_MIN+int(m&0x7Fu); }
float metaVel(uint m){ return float((m>>7)&127u)/127.0; }
uint  gateTick(uint i){ uint w=midiGT(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }

// ── MIDI pitch-bend + expression lanes: 4 bytes/note sampled across the
// key-held span, lerped at play time (slides + CC11/CC7 swells/fades).
// Arrays only exist when the file uses them (MIDI_HAS_* from the Common data).
#ifndef MIDI_HAS_BEND
#define MIDI_HAS_BEND 0
#endif
#ifndef MIDI_HAS_EXPR
#define MIDI_HAS_EXPR 0
#endif
#if MIDI_HAS_BEND || MIDI_HAS_EXPR
float midiLane4(uint w, float x){                 // 4 packed bytes -> piecewise-linear
    float p  = clamp(x, 0.0, 1.0)*3.0;
    float e0 = float(w&255u),        e1 = float((w>>8u)&255u);
    float e2 = float((w>>16u)&255u), e3 = float(w>>24u);
    return (p<1.0) ? mix(e0,e1,p) : (p<2.0) ? mix(e1,e2,p-1.0) : mix(e2,e3,p-2.0);
}
#endif
#if MIDI_HAS_BEND
float noteBend(uint i, float x){ return (midiLane4(midiBD(i), x)-128.0)/32.0; }  // semitones
#else
float noteBend(uint i, float x){ return 0.0; }
#endif
#if MIDI_HAS_EXPR
float noteExpr(uint i, float x){ return midiLane4(midiEXP(i), x)*(1.0/255.0); }  // 0..1 gain
#else
float noteExpr(uint i, float x){ return 1.0; }
#endif

float fbsin(float p,float fb){ float y=sin(p); y=sin(p+fb*y); y=sin(p+fb*y); return y; }
// TDM's hash (Bamboo bass, shadertoy XldcRr) — per-tap reverb jitter
vec2 rvHash(float p){ return fract(sin(p * vec2(110.082, 175.025)) * vec2(19871.8972, 11571.72)); }

// ── master soft-knee limiter (ported 1:1 from MOD2GLSL's mod_player.py) ──
// T=0.85 knee: bit-perfect below it; the summed voices approach 1.0 smoothly
// instead of saturating from the first sample like the old always-on tanh.
vec2 softLimit(vec2 x){
    const float T=0.85, HEAD=1.0-T;
    vec2 ax=abs(x), over=max(ax-T,vec2(0.0));
    return sign(x)*(min(ax,vec2(T)) + HEAD*over/(over+HEAD));
}

#ifndef VOICE
#define VOICE 0                // 0 = FM brass, 1 = additive piano (--inst piano)
#endif
#if VOICE==1
//__PIANOMAN__
#elif VOICE==2
//__EPIANO__
#elif VOICE==3
//__ORGAN__
#elif VOICE==4
//__PIANO2__
#elif VOICE==5
//__EPIANO2__
#elif VOICE==6
//__PIANO3__
#elif VOICE==7
//__PIANO4__
#elif VOICE==8
//__PIANO5__
#elif VOICE==9
//__PIANO6__
#elif VOICE==10
//__PIANO7__
#else
// FM brass voice: freq, seconds since onset, velocity
float voice(float f, float ageOn, float vel){
    float ph = TAU*f*ageOn;
    float I  = (1.0+2.4*vel)*exp2(-ageOn*3.5) + 0.6;
    float m  = I*fbsin(ph, 0.5);
    float dec= 0.55 + 0.45*exp2(-ageOn*0.7);            // gentle decay so holds don't drone
    return (sin(ph+m)+sin(ph*1.004+m))*0.5*dec;
}
#endif

// ── volume ADSR envelope + opt-in comb reverb ── tune via these #defines / -D ─
#ifndef ENV_ATTACK
#define ENV_ATTACK  0.02      // s: onset swell (raise for softer opening, e.g. 0.05)
#endif
#ifndef ENV_DECAY
#define ENV_DECAY   0.09      // s: peak -> sustain
#endif
#ifndef ENV_SUSTAIN
#define ENV_SUSTAIN 0.72      // 0..1: held level
#endif
#ifndef ENV_RELEASE
#define ENV_RELEASE 0.18      // s: note-off decay
#endif
#ifndef USE_REVERB
#define USE_REVERB 1          // reverb ON by default; pass --no-reverb (or set 0) to disable
#endif
#ifndef REVERB_MIX
#define REVERB_MIX 0.42       // wet amount
#endif
#ifndef REVERB_TAPS
#define REVERB_TAPS 16        // stochastic echo taps; raise (48-128) for a denser tail
#endif
#ifndef REVERB_DUR
#define REVERB_DUR 0.9        // reverb length, seconds
#endif
#ifndef REVERB_FALL
#define REVERB_FALL 3.0       // tap decay (exp2 exponent per second); lower = longer tail
#endif
#ifndef PAD_SEC
#define PAD_SEC 1.2           // how long a dead note stays in the scan (must cover the reverb tail)
#endif
#ifndef MASTER_GAIN
#define MASTER_GAIN 0.45      // overall level (raise if too quiet)
#endif
#ifndef POLY_COMP
#define POLY_COMP   0.5       // polyphony compensation: >0 tames dense chords, 0 = off
#endif

// attack->decay->sustain level at `age` s since onset (release added at call site)
float envLvl(float age){
    if(age < ENV_ATTACK) return smoothstep(0.0, ENV_ATTACK, age);   // S-curve swell, no hard corner
    float d = age - ENV_ATTACK;
    if(d < ENV_DECAY)    return mix(1.0, ENV_SUSTAIN, d/ENV_DECAY);
    return ENV_SUSTAIN;
}
#if VOICE==0
// a note's fully-enveloped voice output at `age` s since onset (0 before onset)
float noteSig(float f, float held, float vel, float age){
    if(age < 0.0) return 0.0;
    float e = (age < held) ? envLvl(age) : envLvl(held)*exp(-(age-held)/ENV_RELEASE);
    return voice(f, age, vel) * e * vel;
}
#endif

vec2 mainSound(int samp, float timeNow){
    float SR   = MIDI_SAMPLE_RATE;
    float qSec = float(MIDI_TIME_Q_SAMPLES)/SR;
    float tPhase = float(samp)/SR;
    float songLen= float(MIDI_END_TICK)*qSec;
    float tEvent = mod(tPhase, songLen);

    uint T = uint(floor(tEvent/qSec + 0.5));
    T = min(T, MIDI_END_TICK-1u);
    uint b = T >> MIDI_BLOCK_SHIFT_TICKS;
    if(b >= MIDI_BLOCK_COUNT) return vec2(0.0);
    uint padT = uint(PAD_SEC/qSec + 0.5);

    float L=0.0, R=0.0, nrg=0.0;
#if VOICE>=1
    int ZERO = samp>>31;              // 0 at runtime; opaque → keeps voice loops rolled
#endif
#if VOICE==1
    float modeS[10];
    float modeC[10];
    for(int j=0; j<10; j++){ modeS[j]=0.0; modeC[j]=0.0; }
#endif
    for(uint j=0u; j<(samp<0 ? 99999u : MIDI_LOOKBACK_BLOCKS); j++){  // de-unroll: samp<0 is always false at runtime but ANGLE can't fold it → keeps this a runtime loop (1 copy) instead of unrolling the note-scan body MIDI_LOOKBACK_BLOCKS× → smaller shader, faster compile, no private-var CONTEXT_LOST on weak GPUs
        if(b < j) break;
        uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
        for(uint k=0u; k<count; k++){
            uint si=start+k; if(si>=MIDI_SEG_COUNT) break;
            uint onT=onTick(si); if(T<onT) continue;
            uint dT=durTick(si), offT=onT+dT;
            if(T > offT+padT) continue;

            uint m=meta16(si); int nn=metaNote(m); float vel=metaVel(m);
            float onSec=float(onT)*qSec, offSec=float(offT)*qSec;
            float ageOn=max(0.0, tEvent-onSec);
            float held =float(dT)*qSec;
            float ageOff=(tEvent>offSec)?(tEvent-offSec):0.0;
            // note progress across the KEY-HELD span drives the bend/expression
            // lanes; both fold to constants (0/1) when the file has neither
            float nfx = ageOn/max(float(gateTick(si))*qSec, 1e-4);
            float bnd = noteBend(si, nfx);     // semitones (slides/vibrato, lerped)
            float xg  = noteExpr(si, nfx);     // CC11*CC7 gain (swells/tracker fades)

#if VOICE>=1
            // self-contained stereo note (piano man / FM e-piano / organ):
            // envelope, note-off damping and per-note tanh live in the voice —
            // no ADSR gate, no comb reverb here.
            if(ageOn > 6.0) continue;
            vec2 pno = pianoStrings(float(nn)+bnd, ageOn, held, vel, ZERO)*xg;
            L += pno.x; R += pno.y;
            nrg += 0.35*vel*xg*exp(-ageOn*2.0);
#if VOICE>=2 && VOICE!=6 && USE_REVERB
            {   // bamboo-style reverb for the cheap voices (epiano/organ);
                // VOICE 1 piano man supplies its own space (body modes+antiRes)
                float rdt = REVERB_DUR/float(REVERB_TAPS);
                vec2 rsum = vec2(0.0), rout = vec2(0.0);
                for(int ri=0; ri<(ZERO + REVERB_TAPS); ri++){   // ZERO keeps it rolled
                    vec2 rnd = rvHash(float(ri));
                    vec2 rt  = float(ri)*rdt + rdt*rnd*5.0;
                    vec2 ra  = exp2(-rt*REVERB_FALL);
                    ra *= vec2(rnd.x < 0.5 ? -1.0 : 1.0, rnd.y < 0.5 ? -1.0 : 1.0);  // polarity: no coherent LF pileup
                    rout.x += pianoWet(float(nn)+bnd, ageOn-rt.x, held, vel, ZERO).x*ra.x;
                    rout.y += pianoWet(float(nn)+bnd, ageOn-rt.y, held, vel, ZERO).y*ra.y;
                    rsum += abs(ra);
                }
                rout /= rsum;
                L += REVERB_MIX*rout.x*xg; R += REVERB_MIX*rout.y*xg;
            }
#endif
#if VOICE==1
            // soundboard body modes, excited per note (ported from play())
            float bassBoost = 0.4 + 0.8*clamp((60.0-float(nn))/40.0, 0.0, 1.0);
            float excite    = vel * bassBoost * xg;
            float rise      = 1.0 - exp(-100.0*ageOn);
            for(int q=0; q<10; q++){
                vec4 M = bodyMode(q);
                float env = excite * rise * exp(-M.z*ageOn);
                float wbt = kTau*M.x*onSec;
                modeS[q] += env*cos(wbt);
                modeC[q] += env*sin(wbt);
            }
#endif
#else
            // volume ADSR: attack->decay->sustain while held, exp release from note-off level
            float f = NoteToHz(nn)*exp2(bnd/12.0);
            float e = (tEvent<offSec) ? envLvl(ageOn) : envLvl(held)*exp(-ageOff/ENV_RELEASE);
            if(e < 0.0004) continue;
            float s = voice(f, ageOn, vel)*e*vel*xg;
            float pan = clamp((float(nn)-72.0)/36.0,-1.0,1.0)*0.30;
            nrg += e*vel*xg;   // sum of voice loudnesses, for polyphony normalization
            L += s*(0.5-pan*0.5); R += s*(0.5+pan*0.5);
#if USE_REVERB
            // "Bamboo"-style stochastic reverb (TDM, shadertoy XldcRr — from the
            // Bamboo bass): REVERB_TAPS hash-jittered echo taps with exp2 decay,
            // L/R-decorrelated delays, energy-normalized. Replaces the old
            // 5-tap comb. Wet is unpanned — the dry carries the imaging.
            {
                float rdt = REVERB_DUR/float(REVERB_TAPS);
                vec2 rsum = vec2(0.0), rout = vec2(0.0);
                for(int ri=0; ri<(samp<0 ? 99999 : REVERB_TAPS); ri++){  // de-unroll
                    vec2 rnd = rvHash(float(ri));
                    vec2 rt  = float(ri)*rdt + rdt*rnd*5.0;
                    vec2 ra  = exp2(-rt*REVERB_FALL);
                    ra *= vec2(rnd.x < 0.5 ? -1.0 : 1.0, rnd.y < 0.5 ? -1.0 : 1.0);  // polarity: no coherent LF pileup
                    rout.x += noteSig(f,held,vel,ageOn-rt.x)*ra.x;
                    rout.y += noteSig(f,held,vel,ageOn-rt.y)*ra.y;
                    rsum += abs(ra);
                }
                rout /= rsum;
                L += REVERB_MIX*rout.x*xg; R += REVERB_MIX*rout.y*xg;
            }
#endif
#endif
        }
    }
#if VOICE==1
    // soundboard output: 10 resonant body modes ringing from the note excitations
    for(int j=0; j<10; j++){
        vec4 M = bodyMode(j);
        float wt = kTau*M.x*tEvent;
        float cL = cos(wt),        sL = sin(wt);
        float cR = cos(wt + M.w),  sR = sin(wt + M.w);
        L += 0.08 * M.y * (sL*modeS[j] - cL*modeC[j]);
        R += 0.08 * M.y * (sR*modeS[j] - cR*modeC[j]);
    }
#endif
    // polyphony-aware master gain — DYNAMIC: nrg is the live summed voice energy,
    // recomputed every sample, so g rides up/down with the real-time polyphony
    // (chord hits -> g drops, chord releases -> g rises). sqrt() soft-knee: unity
    // when sparse, ~1/energy rolloff when dense, gliding smoothly across the knee
    // instead of stepping. POLY_COMP sets the knee (higher = tames sooner; 0 = off).
    float lvl = nrg*POLY_COMP;
    float g   = MASTER_GAIN * inversesqrt(1.0 + lvl*lvl);
    vec2 lr = softLimit(vec2(L,R)*g);
    return lr * smoothstep(0.0, 0.4, tPhase);
}
"""
# Voice code is injected at BUILD time by inject_voices(): only the SELECTED
# voice's GLSL ships — unselected synth/piano models are stripped from the
# emitted tab entirely (smaller tabs, kinder to weak drivers; user request).
_VOICE_TOKENS = {
    "piano":   "//__PIANOMAN__", "epiano": "//__EPIANO__",  "organ":  "//__ORGAN__",
    "piano2":  "//__PIANO2__",   "epiano2": "//__EPIANO2__", "piano3": "//__PIANO3__",
    "piano4":  "//__PIANO4__",   "piano5": "//__PIANO5__",   "piano6": "//__PIANO6__",
    "piano7":  "//__PIANO7__",
}
def _voice_glsl(inst):
    return {"piano": PIANOMAN_GLSL, "epiano": EPIANO_GLSL, "organ": ORGAN_GLSL,
            "piano2": PIANO2_GLSL, "epiano2": EPIANO2_GLSL, "piano3": PIANO3_GLSL,
            "piano4": PIANO4_GLSL, "piano5": PIANO5_GLSL, "piano6": PIANO6_GLSL,
            "piano7": PIANO7_GLSL}[inst]
def inject_voices(code, inst):
    for key, tok in _VOICE_TOKENS.items():
        code = code.replace(tok, _voice_glsl(inst) if key == inst
                            else "// (voice not selected -- stripped at build)")
    return code

GM_PLAYER_CODE = r"""
// ============================================================ GM MULTI-TIMBRAL
// Self-contained Sound tab: each note dispatches to a voice by its MIDI
// program (0xC0). PoC family = ORGANS (16-23), authored from the classic
// Hammond drawbar registrations; everything else uses a mellow FM fallback.
// Ships INSTEAD of PLAYER_CODE, so all accessors are defined here.
#define TAU 6.28318530718
float NoteToHz(int n){ return 440.0*exp2((float(n)-69.0)/12.0); }
uint  onTick(uint i){ return midiOM(i)&0xFFFFu; }
uint  meta16(uint i){ return midiOM(i)>>16; }
uint  durTick(uint i){ uint w=midiDur(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
uint  gateTick(uint i){ uint w=midiGT(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
int   metaNote(uint m){ return MIDI_NOTE_MIN+int(m&0x7Fu); }
float metaVel(uint m){ return float((m>>7)&127u)/127.0; }
uint  noteProg(uint i){ uint w=midiPG(i>>2u); return (w >> ((i&3u)*8u)) & 0xFFu; }

// bend + expression lanes (same 4-sample packing as PLAYER_CODE)
#ifndef MIDI_HAS_BEND
#define MIDI_HAS_BEND 0
#endif
#ifndef MIDI_HAS_EXPR
#define MIDI_HAS_EXPR 0
#endif
#if MIDI_HAS_BEND || MIDI_HAS_EXPR
float midiLane4(uint w, float x){
    float p  = clamp(x, 0.0, 1.0)*3.0;
    float e0 = float(w&255u),        e1 = float((w>>8u)&255u);
    float e2 = float((w>>16u)&255u), e3 = float(w>>24u);
    return (p<1.0) ? mix(e0,e1,p) : (p<2.0) ? mix(e1,e2,p-1.0) : mix(e2,e3,p-2.0);
}
#endif
#if MIDI_HAS_BEND
float noteBend(uint i, float x){ return (midiLane4(midiBD(i), x)-128.0)/32.0; }
#else
float noteBend(uint i, float x){ return 0.0; }
#endif
#if MIDI_HAS_EXPR
float noteExpr(uint i, float x){ return midiLane4(midiEXP(i), x)*(1.0/255.0); }
#else
float noteExpr(uint i, float x){ return 1.0; }
#endif

vec2 softLimit(vec2 x){
    const float T=0.85, HEAD=1.0-T;
    vec2 ax=abs(x), over=max(ax-T,vec2(0.0));
    return sign(x)*(min(ax,vec2(T)) + HEAD*over/(over+HEAD));
}

// ── user-provided synth library (ShaderToy analog voices, integrated) ──
float nse(float x){ return fract(sin(x*45.233)*43758.5453)*2.0-1.0; }   // (mine; reed/pipe breath)
float u_rand(float p){ p=fract(p*.1031); p*=p+33.33; p*=p+p; return fract(p); }
float u_noise(float s){ int si=int(floor(s)); float sf=fract(s); sf=sf*sf*(3.-2.*sf);
    return mix(u_rand(float(si)), u_rand(float(si+1)), sf)*2.-1.; }
float coloredNoise(float t, float fc, float df){ return sin(TAU*fract(fc*t))*u_noise(t*df); }
vec2 coloredNoise2(float t, float fc, float df){ return sin(TAU*fract(fc*t))*vec2(u_noise(t*df),u_noise(-1000.-t*df)); }
float u_sweep(float t, float dur){ float dt=dur/log(1e3); float intf=20000.*exp(-t/dt)*dt;
    float sig=sin(TAU*fract(intf)); float env=step(0.,t)*smoothstep(dur,0.7*dur,t); return sig*env*0.1; }
float triBipolar(float x){ return 1.-abs(2.-mod(x+1.,4.)); }
float ucurve(float x,float a,float b,float cur){ x=(x-a)/(b-a); x=clamp(x,0.,1.); return pow(x,exp(-cur)); }
float u_midicps(float n){ return 440.*exp2((n-69.)/12.); }
// analytic low-pass resonant sawtooth (lpfSaw3) — the subtractive-synth core
float lpfSaw3(float t, float f, float fc, float Q){
    float wc = 2.*3.14159*fc/f; t = f*t - floor(f*t);
    float al = wc/Q, be=exp(-al), c=cos(wc), s=sin(wc);
    float tanphi = (al*be*c + be*wc*s - al)/(wc + al*be*s - be*wc*c);
    float phi = atan(tanphi);
    float A = -2./(cos(phi) - be*cos(wc+phi));
    return 1.-2.*t + A*exp(-al*t)*cos(wc*t+phi);
}
// ---- drum kit (user's analog one-shots; age = time since hit) ----
float kick(float t, float te, float atk){        // SC-55 acoustic bass drum: tight thud + beater click
    float f0=u_midicps(34.), df=280., spd=55.;
    float phase = TAU*f0*te + TAU*df/spd*(1.-exp(-te*spd));
    float env = (exp(-19.*te)+2.5*exp(-120.*te))*smoothstep(-1e-6,0.,te);   // tight but weighty low-end
    float v = sin(phase)*env*1.15;
    // beater click: a short mid+high transient (the acoustic "knock" a synth kick lacks)
    float click = (u_noise(16000.*te)*0.5 + 0.5*sin(TAU*1900.*te))*exp(-90.*te)*smoothstep(0.,0.0006,te)*0.22;
    v /= 1.+0.25*abs(v);
    v += click;
    return v*smoothstep(-1e-6,atk,te);
}
float snare(float t, float te, float atk){
    float wn = u_noise(20000.*te)+coloredNoise(te,6500.,1000.)*0.3;
    float nenv=(exp(-5.*te)+exp(-30.*te))*smoothstep(0.2,0.0,te)*0.5;
    float phase=TAU*180.*te+TAU*80./50.*(1.-exp(-te*50.));
    float body=sin(phase)*1.5*smoothstep(0.,0.005,te)*smoothstep(0.05,0.,te);
    float v=0.5*(wn*nenv+body); v/=1.+abs(v);
    v*=smoothstep(-1e-6,atk,te)*(1.+0.3*smoothstep(0.01,0.0,te)+0.2*smoothstep(0.05,0.2,te));
    return v;
}
float hihat(float te, float dec){                 // dec = ring length (closed ~0.12, open ~0.7)
    float sig=coloredNoise(te,8200.,5200.)+coloredNoise(te,2000.,1800.)*0.08;   // a touch airier (SC-55)
    float env=smoothstep(0.0,0.0002,te)*(ucurve(te,dec,0.0,-2.)+0.5*smoothstep(0.01,0.,te));
    return sig*env;
}
float hihat(float te){ return hihat(te, 0.12); }   // default: closed
vec2 tomDrum(float te, float f0, float df){       // SC-55 tuned tom: dark low body + stick attack
    float phase=TAU*(f0*te + df/9.*(1.-exp(-te*13.)));   // pitch drops into the fundamental
    float env=smoothstep(0.,0.001,te)*exp(-6.5*te);      // tuned membrane decay (~0.4s)
    float tone=(sin(phase)+0.25*sin(2.0*phase))*env;     // + a mid membrane mode
    float click=(u_noise(9000.*te)+coloredNoise(te,2600.,3200.))*exp(-55.*te)*smoothstep(0.,0.0008,te)*0.28; // stick knock (brighter)
    vec2 sig=vec2(tone*0.9 + click);
    sig/=1.+abs(sig)*0.4;
    return sig;
}
vec2 ride(float te){                              // SC-55 ride: defined ping (shorter than crash)
    float bell=sin(TAU*520.*te)+0.7*sin(TAU*1240.*te)+0.5*sin(TAU*2100.*te);
    vec2 sig=coloredNoise2(te,7200.,9000.)*0.7 + 0.2*bell*smoothstep(0.,0.004,te);
    float env=smoothstep(0.,0.001,te)*(exp(-9.*te)+0.25*exp(-3.5*te));   // ping + short wash (~0.23s)
    return sig*env*0.18;
}
vec2 clap(float t){
    vec2 sig=0.8*coloredNoise2(t,1000.,800.)+0.5*coloredNoise2(t,3300.,3100.)+0.4*coloredNoise2(t,8240.,8000.);
    sig*=smoothstep(0.,0.01,t)*(ucurve(t,1.0,0.03,-2.3)+0.2*ucurve(t,5.0,0.1,-1.));
    sig*=mix(0.7,0.5+0.5*sin(TAU*80.*t),smoothstep(0.06,0.03,t)); sig/=1.+abs(sig);
    return sig;
}
vec2 crashCymbal(float t, float atk){
    float reson=sin(TAU*429.*t+5.*sin(TAU*1120.*t)+5.*sin(TAU*1812.*t));
    vec2 sig=coloredNoise2(t,7150.,10000.)+0.1*reson*smoothstep(0.,0.05,t);
    float env=ucurve(t,15.0,0.0,-3.)*ucurve(t,0.0,0.08,1.);
    env*=(1.+smoothstep(0.02,0.0,t)*2.)*(1.-smoothstep(0.0,0.05,t)*smoothstep(0.5,0.0,t)*0.5)*smoothstep(0.,atk,t);
    return sig*env*0.2;
}
// ---- "dirty" kit (user's distortion-driven kick/snare/hat, --kit dirty) ----
vec2 hash22(vec2 p){ vec3 p3=fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); p3+=dot(p3,p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float u_sine(float t,float f,float ph){ return sin(TAU*f*t+ph); }
float dist1(float x,float a){ return tanh(x*a); }
vec2  dist1(vec2  x,float a){ return tanh(x*a); }
vec2  dist2(vec2  x,float a){ return x*a/(1.+abs(x*a)); }
vec2 kick2(float t){        // punchy distorted kick: pitch sweep (punch) + beater click (definition)
    float ph = TAU*(72.*t + (100.0/46.0)*(1.0-exp(-t*46.0)));   // ~172Hz -> 72Hz sweep = the punch
    vec2 body = 1.15*dist2(exp(-t*22.)*vec2(sin(ph)), 2.2);      // distorted swept body (keeps the grit)
    vec2 sub  = 0.55*exp(-t*13.)*vec2(sin(TAU*68.*t));           // sustained sub tail
    float click = (u_noise(15000.*t)*0.6 + 0.55*sin(TAU*2100.*t))*exp(-95.*t)*smoothstep(0.0,0.0005,t)*0.40; // beater knock
    return body + sub + vec2(click);
}
vec2 termKick(float t){     // Dmitry Andreev "Terminator" FM kick (pitch-swept + FM transient)
    float f0=110.0;
    float f1=f0*(exp(-t*8.0)*1.5+0.5);                           // 220Hz -> 55Hz sweep
    float op2=sin(TAU*t*f0*2.0)*exp(-t*40.0);                    // FM transient modulator
    float op1=sin(TAU*(t*f1 + op2*0.20))*pow(clamp(1.2-t*2.0,0.0,1.0),0.5);
    return vec2(op1);
}
vec2 snare2(float t){
    vec2 s = exp(-t*23.)*(hash22(iSampleRate*vec2(t,t*1.423))*2.0-1.0);
    s = dist2(s,1.3)*.85;
    s += .5*dist1(vec2(exp(-t*28.)*u_sine(t,200.,0.)), 4.2);
    return s*0.5;
}
vec2 hat2(float t){
    vec2 s = sqrt(t*7.)*exp(-t*38.)*(hash22(iSampleRate*vec2(t,t*1.423))*2.0-1.0);
    return dist2(s,1.1)*.85*0.7;
}
#ifndef GM_KIT_DIRTY
#define GM_KIT_DIRTY 0
#endif
// GM drum map -> stereo voice (age = time since hit)
vec2 gmDrum(int nn, float a, float vel){
#if GM_KIT_DIRTY
    if(nn==35||nn==36)            return kick2(a)*0.85;
    if(nn==38||nn==40)            return snare2(a)*0.95;
    if(nn==37)                    return snare2(a)*0.5;
    if(nn==42||nn==44)            return hat2(a)*0.7;
    if(nn==46)                    return hat2(a)*1.0;
#else
    if(nn==35||nn==36)            return kick2(a)*0.95;             // punchy tuned kick (pitch sweep + beater click)
    if(nn==38||nn==40)            return vec2(snare(a,a,0.0))*0.72;
    if(nn==37)                    return vec2(snare(a,a,0.0))*0.42;
    if(nn==42||nn==44)            return vec2(hihat(a, 0.18))*0.40;  // closed hat (sits back)
    if(nn==46)                    return vec2(hihat(a, 1.90))*0.55;  // open hat rings
#endif
    if(nn==39)                    return clap(a)*0.8;
    if(nn==49||nn==57||nn==55)    return crashCymbal(a,0.0)*0.7;
    if(nn==51||nn==53||nn==59)    return ride(a)*0.9;                // defined ride ping (not a crash wash)
    if(nn==41||nn==43)            return tomDrum(a, 78.0, 55.0)*1.2; // floor toms (dark)
    if(nn==45||nn==47)            return tomDrum(a,105.0, 75.0)*1.2; // low toms
    if(nn==48||nn==50)            return tomDrum(a,150.0,100.0)*1.2; // high toms
    if(nn==54||nn==56||nn==69||nn==70) return vec2(hihat(a))*0.35;
    return vec2(hihat(a))*0.3;
}

// ADSR with exponential release from the held level
float gmADSR(float age, float held, float a, float d, float s, float r){
    if(age < 0.0) return 0.0;
    if(age < held){
        return (age < a)   ? age/max(a,1e-4)
             : (age < a+d) ? mix(1.0, s, (age-a)/max(d,1e-4))
             : s;
    }
    float hl = (held < a) ? held/max(a,1e-4)
             : (held < a+d) ? mix(1.0, s, (held-a)/max(d,1e-4)) : s;
    return hl * exp(-(age-held)/max(r,1e-4));
}

// Hammond drawbar: 9 bars at ratios [0.5,1.5,1,2,3,4,5,6,8], levels 0-8 each.
float drawbar(float ph, float b0,float b1,float b2,float b3,float b4,
                        float b5,float b6,float b7,float b8){
    float o = b0*sin(ph*0.5) + b1*sin(ph*1.5) + b2*sin(ph)      + b3*sin(ph*2.0)
            + b4*sin(ph*3.0) + b5*sin(ph*4.0) + b6*sin(ph*5.0)  + b7*sin(ph*6.0)
            + b8*sin(ph*8.0);
    return o * (1.0/24.0);   // normalize (~3 bars at 8 -> ~1.0)
}
float gmfb(float p,float fb){ float y=sin(p); y=sin(p+fb*y); y=sin(p+fb*y); return y; }
// additive saw (band-limited-ish): sum sin(h*ph)/h — strings/leads/pads
float sawAdd(float ph, int N){ float s=0.0;
    for(int h=1;h<=(iChannelResolution[0].x<0.0?99999:20);h++){ if(h>N) break; s += sin(ph*float(h))/float(h); } return s*0.55; }
// additive square (odd harmonics) — clarinet/hollow
float sqrAdd(float ph, int N){ float s=0.0;
    for(int h=1;h<=(iChannelResolution[0].x<0.0?99999:19);h+=2){ if(h>N) break; s += sin(ph*float(h))/float(h); } return s*0.8; }
// plucked/decaying harmonic sum — guitar / harp / pizz / mallet
float pluckAdd(float ph, float age, float bright){ float s=0.0;
    for(int h=1;h<=12;h++){ float hh=float(h); s += sin(ph*hh)/hh*exp2(-age*(1.2+hh*bright)); }
    return s; }
// struck decay envelope (piano/pluck/bell): decays even while held, faster on release
float gmStruck(float age, float held, float dec, float rel){
    return exp2(-age*dec) * (age < held ? 1.0 : exp(-(age-held)/max(rel,1e-4))); }

// ---- melodic voices from the user's library ----
// SC-55 bass additive partials (explicit harmonic amplitudes) + subtle stereo detune
vec2 bassPartials(float t, float f, int shape){
    float A[10];
    if(shape==1){        // fretless / slap (bright rolled series)
        A = float[10](1.0,0.870,0.661,0.474,0.319,0.158,0.086,0.0486,0.026,0.018);
    } else if(shape==2){ // synth bass (fast rolloff)
        A = float[10](1.0,0.696,0.298,0.109,0.054,0.027,0.0177,0.0108,0.008,0.006);
    } else if(shape==3){ // acoustic upright (dark; SC-55 profile w/ weak h5/h7)
        A = float[10](1.0,0.255,0.050,0.192,0.007,0.091,0.011,0.006,0.006,0.002);
    } else {             // finger / default
        A = float[10](1.0,0.693,0.329,0.193,0.108,0.074,0.021,0.0109,0.006,0.004);
    }
    float phL = TAU*f*t;
    float phR = TAU*(f*1.004)*t;
    float sl=0.0, sr=0.0;
    for(int h=1;h<=10;h++){ float a=A[h-1]; float hh=float(h);
        sl += a*sin(phL*hh); sr += a*sin(phR*hh); }
    return vec2(sl, sr);
}

vec2 bassStab(float t, float te, float f, uint prog){   // SC-55 bass: additive partials + pluck
    int shape = 0;                                       // finger / default
    if(prog==32u) shape = 3;                             // acoustic upright -> dark
    else if(prog==35u||prog==36u||prog==37u) shape = 1;  // fretless / slap -> bright
    else if(prog>=38u) shape = 2;                        // synth bass -> fast rolloff
    float sus = (shape==2) ? 0.223 : 0.45;
    float env = smoothstep(0.,0.006,te)*(sus + (1.-sus)*exp(-8.*te));
    vec2 body = bassPartials(t, f, shape);
    // bright plucked transient (decays <40ms; keeps attack bite, gone by sustain)
    float fcp = 400. + 9000.*exp(-60.*te);
    float plk = lpfSaw3(t, f, fcp, 1.)*exp(-40.*te)*smoothstep(0.,0.001,te);
    return (body*env + vec2(plk)*0.35)*0.18;
}
vec2 analogBrass(float t, float te, float f, uint prog){  // additive SC-55 brass, prog+freq aware
    // Bright brass with a broad harmonic plateau (matches SC-55 trumpet/tuba via freq).
    // Trombone(57) is brighter & h3-peaked; brass section(62) is markedly mellower.
    // Upper partials swell in over ~30ms.
    float wHi[14] = float[14](1.0,1.675,1.262,0.726,0.927,0.690,0.475,0.310,0.188,0.16,0.10,0.06,0.035,0.02);
    float wLo[14] = float[14](1.0,0.883,1.196,1.512,1.040,0.653,0.556,0.533,0.628,0.562,0.367,0.204,0.16,0.09);
    float wBn[14] = float[14](1.0,1.691,2.031,1.403,1.181,0.943,0.727,0.565,0.389,0.323,0.28,0.223,0.18,0.14); // trombone
    float wSc[14] = float[14](1.0,0.282,0.369,0.086,0.150,0.107,0.074,0.063,0.041,0.049,0.019,0.05,0.02,0.015); // section
    float wHn[14] = float[14](1.0,1.748,1.363,0.539,2.217,1.085,1.254,0.726,0.553,0.522,0.40,0.30,0.22,0.15); // french horn (h5/h7 formant bumps)
    float blend = smoothstep(120.,200.,f);          // 0 = tuba, 1 = trumpet
    float sw = smoothstep(0.0,0.03,te);             // bright attack swell
    float vib = 1.0 + 0.004*sin(TAU*5.2*te);        // gentle vibrato (right side only)
    vec2 sig = vec2(0.0);
    for(int n=1;n<=14;n++){
        float a  = mix(wLo[n-1], wHi[n-1], blend);
        if(prog==57u) a = wBn[n-1];                 // trombone: bright, h3-peaked
        else if(prog==62u) a = wSc[n-1];            // brass section: mellow
        else if(prog==61u) a = wHn[n-1];            // french horn: h5/h7 formant bumps
        float hs = (n>=5) ? mix(0.45,1.0,sw) : 1.0; // upper partials fade in
        float fn = f*float(n);
        float ph = TAU*fn*t;
        sig.x += a*hs*sin(ph);
        sig.y += a*hs*sin(ph*vib + 0.6);            // slight detune/vibrato for width
    }
    float env = smoothstep(0.0,0.05,te)*(0.62+0.38*exp(-te/0.4));
    return sig*0.045*env;
}
// bright rolled-saw lead partials, SC-55 synth-lead compromise (progs 80/81/84
// share one voice): per-harmonic L1-median of square-lead(80)+saw-lead(81)+
// charang(84) -> odd-emphasised bright saw.
float leadPartials(float ph){
    float s  = sin(ph);
    s += 0.453*sin( 2.*ph);
    s += 0.456*sin( 3.*ph);
    s += 0.258*sin( 4.*ph);
    s += 0.263*sin( 5.*ph);
    s += 0.159*sin( 6.*ph);
    s += 0.142*sin( 7.*ph);
    s += 0.095*sin( 8.*ph);
    s += 0.090*sin( 9.*ph);
    s += 0.073*sin(10.*ph);
    s += 0.055*sin(11.*ph);
    s += 0.042*sin(12.*ph);
    return s;
}
// hollow square-lead profile (SC-55 square lead 80: weak h2, strong odd h3/h5)
float leadSq(float ph){
    return sin(ph) + 0.094*sin(2.*ph) + 0.494*sin(3.*ph) + 0.187*sin(4.*ph)
         + 0.280*sin(5.*ph) + 0.165*sin(6.*ph) + 0.084*sin(7.*ph) + 0.040*sin(8.*ph)
         + 0.011*sin(9.*ph) + 0.006*sin(11.*ph);
}
vec2 sawLead(float t, float te, float f, uint prog){  // prog-aware lead (square vs rolled-saw)
    float env = smoothstep(0.,0.008,te);            // fast lead attack
    float phL = TAU*f*t, phR = TAU*f*1.003*t;       // gentle detune -> width (right only)
    float l, r;
    if(prog==80u || prog==87u){ l = leadSq(phL);      r = leadSq(phR); }      // square / bass-lead: hollow
    else                      { l = leadPartials(phL); r = leadPartials(phR); } // saw / charang / others
    return vec2(l, r)*0.16*env;
}
vec2 msin2(vec2 x, vec2 m){ return sin(TAU*fract(x)+m); }
vec2 epiano(float freq, float t, float nuance){     // mellow DX-tine e-piano (SC-55 EP1)
    vec2 f0 = vec2(freq*0.999, freq*1.001);
    vec2 ph = TAU*f0*t;
    float e = exp(-1.3*t);
    // SC-55 EP1 profile: mostly fundamental + a little h2/h3, near-silent above
    vec2 body = sin(ph)
              + 0.198*sin(2.0*ph)
              + 0.068*sin(3.0*ph)
              + 0.0063*sin(4.0*ph)
              + 0.0110*sin(5.0*ph)
              + 0.0029*sin(6.0*ph)
              + 0.0015*sin(7.0*ph)
              + 0.0016*sin(8.0*ph);
    // fast glassy tine ping (attack only, gone before the sustain window)
    vec2 tine = 0.16*sin(4.0*ph + 3.0*sin(ph))*exp(-24.0*t);
    float pd = clamp(log2(freq/400.)/2.,-1.,1.); vec2 pan = normalize(vec2(0.5-0.5*pd,0.5+0.5*pd));
    return (body*e + tine*pan)*(0.09*nuance*smoothstep(0.,0.002,t));
}
// Tambako morphable osc (tri->saw wf2 / PWM wf4) + 5-voice detuned unison —
// upgrades the pad + bowed-string families (the weakest, synthy ones)
float tambOsc(float t, float freq, float phase, int wf, float p1){
    float tt = fract(freq*t + phase);
    if(wf==2) return 2.*smoothstep(0.,p1,tt)*smoothstep(1.,p1,tt) - 1.;  // tri -> saw
    return tt > p1 ? 1.0 : -1.0;                                          // PWM square
}
vec2 tambPadV(float t, float f, float held, int wf, float p1, float atk){
    vec2 s = vec2(0.0);
    for(int n=0; n<5; n++){
        float fo = f*0.01*0.7*float(n-2);            // ~0.7% detune spread
        float b  = 0.45*float(n-2);
        vec2 bal = vec2(clamp(1.-b,0.,1.), clamp(b+1.,0.,1.));
        s += bal*vec2(tambOsc(t, f+fo, 0.0, wf, p1), tambOsc(t, f+fo, 0.25, wf, p1));
    }
    float env = smoothstep(0.,atk,t) * (t<held ? 1.0 : smoothstep(held+0.4,held,t));
    return s*env*0.045;
}

// GM voice dispatch — full 16-family map. Bass/brass/lead/EP use the user's
// analog voices (early-return, own envelope); the rest use primitive + gmADSR.
// Guitar family: explicit per-harmonic additive so mid-harmonic formant bumps
// (clean/steel) and distortion richness can be matched to the SC-55 profile.
// Coefficients are relative to h1 (=1.0); overall level set by caller via g.
float gtrHarm(float ph, float a2,float a3,float a4,float a5,float a6,
              float a7,float a8,float a9,float a10,float a11,float a12){
    return sin(ph)
         + a2*sin(ph*2.0)  + a3*sin(ph*3.0)  + a4*sin(ph*4.0)  + a5*sin(ph*5.0)
         + a6*sin(ph*6.0)  + a7*sin(ph*7.0)  + a8*sin(ph*8.0)  + a9*sin(ph*9.0)
         + a10*sin(ph*10.0)+ a11*sin(ph*11.0)+ a12*sin(ph*12.0);
}

// ---- SC-55 bowed-string / string-ensemble additive resynthesis (strings_ens family) ----
// harmonic amplitude (relative to h1) for each SC-55 string profile.
float bowProfileH(int h, int kind){
    if(kind==1){                                   // cello / contrabass (bright 2nd harmonic)
        if(h==1) return 1.0;  if(h==2) return 1.55; if(h==3) return 0.42;
        if(h==4) return 0.18; if(h==5) return 0.10; if(h==6) return 0.17;
        if(h==7) return 0.03; if(h==8) return 0.01; if(h==9) return 0.043;
        if(h==10) return 0.017; if(h==11) return 0.012; if(h==12) return 0.02;
        return 0.01;
    } else if(kind==2){                            // string ensemble (fundamental-dominant + mid formant)
        if(h==1) return 1.0;   if(h==2) return 0.33;  if(h==3) return 0.06;
        if(h==4) return 0.04;  if(h==5) return 0.072; if(h==6) return 0.082;
        if(h==7) return 0.063; if(h==8) return 0.022; if(h==9) return 0.011;
        if(h==10) return 0.011; if(h==11) return 0.012; if(h==12) return 0.004;
        return 0.006;
    }                                              // kind 0: violin / viola (rolled saw + upper formant)
    if(h==1) return 1.0;   if(h==2) return 0.28;  if(h==3) return 0.141;
    if(h==4) return 0.035; if(h==5) return 0.126; if(h==6) return 0.101;
    if(h==7) return 0.071; if(h==8) return 0.054; if(h==9) return 0.0;
    if(h==10) return 0.045; if(h==11) return 0.031; if(h==12) return 0.022;
    return 0.02;
}
vec2 bowEns(float f, float age, float held, float vel, int kind){
    // detuned-unison additive: center voice exact (clean spectrum) + 2 shimmer copies
    vec2 s = vec2(0.0);
    for(int v=0; v<3; v++){
        float det = (v==0)? 0.0 : (v==1)? 0.020 : -0.020;
        float amp = (v==0)? 1.0 : 0.5;
        float pan = (v==1)? 0.62 : (v==2)? 0.38 : 0.5;
        float w = TAU*f*(1.0+det)*age;
        float o = 0.0;
        for(int h=1;h<=14;h++){ if(f*float(h) > 16000.0) break;
            o += bowProfileH(h,kind)*sin(w*float(h)); }
        s += vec2(1.0-pan, pan)*o*amp;
    }
    // slow bowed attack (ensembles swell), steady sustain, gentle release
    float atk = (kind==2)? 0.28 : 0.05;
    float env = gmADSR(age, held, atk, 0.12, 0.88, 0.25);
    env *= 1.0 + 0.045*sin(TAU*5.3*age);           // tremolo (AM only -> keeps harmonics sharp)
    return s*env*(0.5+0.6*vel)*0.05;
}

// additive reed tone: hardcoded SC-55-matched partial gains (h1..h16), RMS-normalized
float reedAdd(float ph, int prog){
    float g[16] = float[16](1.0,1.2,0.85,0.55,0.35,0.28,0.20,0.15,0.11,0.08,0.06,0.04,0.03,0.02,0.015,0.01); // default sax-ish
    if(prog==64)      g = float[16](1.0,1.862,0.996,0.451,0.169,0.447,0.061,0.062,0.065,0.048,0.053,0.012,0.010,0.034,0.012,0.035); // soprano sax
    else if(prog==65) g = float[16](1.0,1.30,0.95,0.60,0.30,0.35,0.15,0.12,0.09,0.07,0.05,0.03,0.02,0.02,0.012,0.010); // alto sax
    else if(prog==66) g = float[16](1.0,0.887,0.870,1.274,0.391,0.404,0.458,0.494,0.159,0.278,0.119,0.113,0.040,0.042,0.009,0.029); // tenor sax
    else if(prog==67) g = float[16](1.0,0.030,0.900,0.090,0.480,0.090,0.300,0.080,0.170,0.060,0.100,0.030,0.050,0.020,0.020,0.010); // bari/bassoon hollow
    else if(prog==68) g = float[16](1.0,1.587,1.464,2.269,1.699,1.576,0.401,0.179,0.131,0.281,0.205,0.134,0.129,0.051,0.013,0.015); // oboe (leakage-compensated)
    else if(prog==69) g = float[16](1.0,1.40,1.15,1.60,1.05,0.85,0.35,0.16,0.13,0.22,0.16,0.11,0.09,0.04,0.02,0.012); // english horn
    else if(prog==70) g = float[16](1.0,0.050,0.900,0.120,0.500,0.100,0.280,0.090,0.150,0.060,0.100,0.030,0.040,0.020,0.020,0.010); // bassoon hollow
    else if(prog==71) g = float[16](1.0,0.021,1.073,0.110,0.501,0.071,0.262,0.116,0.129,0.127,0.066,0.023,0.027,0.006,0.011,0.010); // clarinet (leakage-compensated)
    float s=0.0, e=0.0;
    for(int h=1;h<=16;h++){ s += g[h-1]*sin(ph*float(h)); e += g[h-1]*g[h-1]; }
    return s * (0.5/sqrt(e));
}

// Additive mellow detune-unison pad voice — direct harmonic control (a2..a7)
// so SC-55 pad profiles (mellow, few harmonics) match without PWM buzz.
vec2 padAdd(float t, float f, float held, float atk,
            float a2,float a3,float a4,float a5,float a6,float a7){
    vec2 s = vec2(0.0);
    for(int n=0; n<5; n++){
        float fo = f*0.01*0.7*float(n-2);            // ~0.7% detune spread (lush width)
        float b  = 0.45*float(n-2);
        vec2 bal = vec2(clamp(1.-b,0.,1.), clamp(b+1.,0.,1.));
        float ff = f+fo;
        float pl = TAU*ff*t; float pr = pl + 1.7;    // stereo decorrelation
        float wl = sin(pl)+a2*sin(2.*pl)+a3*sin(3.*pl)+a4*sin(4.*pl)+a5*sin(5.*pl)+a6*sin(6.*pl)+a7*sin(7.*pl);
        float wr = sin(pr)+a2*sin(2.*pr)+a3*sin(3.*pr)+a4*sin(4.*pr)+a5*sin(5.*pr)+a6*sin(6.*pr)+a7*sin(7.*pr);
        s += bal*vec2(wl, wr);
    }
    float env = smoothstep(0.,atk,t) * (t<held ? 1.0 : smoothstep(held+0.4,held,t));
    return s*env*0.02;
}

vec2 gmVoice(uint prog, float f, float age, float held, float vel){
    // ---- complete voices (own baked envelope) ----
    if(prog>=32u && prog<=39u)                       // BASS -> bassStab
        return bassStab(age, age, f, prog) * (0.6+0.9*vel);
    if(prog>=56u && prog<=63u)                       // BRASS -> analogBrass + note-off release
        return analogBrass(age, age, f, prog) * (0.7+0.6*vel) * (age<held?1.0:exp(-(age-held)/0.12));
    if(prog>=80u && prog<=87u)                       // SYNTH LEAD -> sawLead + release
        return sawLead(age, age, f, prog) * (0.6+0.6*vel) * (age<held?1.0:exp(-(age-held)/0.15));
    if(prog==4u || prog==5u)                         // E-PIANO -> glassy epiano (the "bellsy" one)
        return epiano(f, age, 0.6+0.8*vel);
    if(prog>=88u && prog<=95u){                      // SYNTH PAD -> additive mellow detune-unison
        if(prog==88u) return padAdd(age,f,held,0.12, 0.945,0.78,0.19,0.079,0.028,0.0075)*(0.7+0.5*vel); // new age
        if(prog==89u) return padAdd(age,f,held,1.0,  0.263,0.055,0.021,0.0088,0.0033,0.0)*(0.7+0.5*vel);// warm
        if(prog==91u) return padAdd(age,f,held,0.11, 1.43,0.19,0.0,  0.007,0.0,0.0    )*(0.7+0.5*vel);  // choir
        if(prog==90u) return padAdd(age,f,held,0.05, 0.5,0.32,0.20,0.12,0.07,0.04)*(0.7+0.5*vel);        // polysynth
        if(prog==94u) return padAdd(age,f,held,0.15, 0.5,0.40,0.26,0.16,0.10,0.06)*(0.7+0.5*vel);        // halo (brighter)
        if(prog==95u) return padAdd(age,f,held,0.2,  0.5,0.36,0.26,0.20,0.12,0.07)*(0.7+0.5*vel);        // sweep (brighter)
        return padAdd(age,f,held,0.25, 0.4,0.28,0.16,0.09,0.05,0.02)*(0.7+0.5*vel);                      // 92 bowed / 93 metallic
    }
    if((prog>=40u && prog<=44u) || (prog>=48u && prog<=51u)){  // bowed strings / ensembles -> rolled-saw additive
        int bk = (prog==42u||prog==43u) ? 1 : (prog>=48u) ? 2 : 0;   // cello/CB / ensemble / violin-viola
        return bowEns(f, age, held, vel, bk);
    }
    float ph  = TAU*f*age;
    float ph2 = TAU*f*1.008*age;                 // detune partner (reeds)
    float sig, env, vib = 0.0;
    if(prog >= 16u && prog <= 23u){
        if(prog == 16u){                          // Drawbar Organ (SC-55 octave stack 1,2,4,8)
            sig = drawbar(ph, 0.,0.,8.,6.5,0.,4.5,0.,0.,2.);
            env = gmADSR(age, held, 0.008, 0.0, 1.0, 0.05);
        } else if(prog == 17u){                   // Percussive Organ (strong 2nd harm + click)
            sig = drawbar(ph, 8.,0.,4.,7.94,0.13,0.83,0.08,0.,0.)
                + 0.0005*sin(ph*6.0)
                + 0.45*sin(ph*4.0)*exp(-age*16.0);
            env = gmADSR(age, held, 0.004, 0.0, 1.0, 0.04);
        } else if(prog == 18u){                   // Rock Organ (mellow drawbar + Leslie)
            sig = drawbar(ph, 8.,8.,8.,2.42,1.80,2.45,0.31,0.0375,0.06)
                + 0.0023*sin(ph*7.0) + 0.00067*sin(ph*9.0);
            env = gmADSR(age, held, 0.006, 0.0, 1.0, 0.06);
            vib = 1.0;
        } else if(prog == 19u){                   // Church Organ (principal chorus, even harmonics)
            sig = drawbar(ph, 4.,0.,4.,8.41,0.0,4.40,0.32,3.67,3.15)
                + 0.0035*sin(ph*9.0) + 0.011*sin(ph*10.0);
            env = gmADSR(age, held, 0.085, 0.06, 0.95, 0.30);
        } else if(prog == 20u){                   // Reed Organ (harmonium, odd harmonics)
            sig = 1.4*(sin(ph) + sin(ph*3.0)/3.0 + sin(ph*5.0)/5.0 + sin(ph*7.0)/7.0);
            env = gmADSR(age, held, 0.04, 0.10, 0.95, 0.12);
        } else if(prog == 21u){                   // Accordion (musette: 2 detuned reed banks)
            sig = 0.275*(drawbar(ph,  8.,0.,8.,9.32,21.0,3.17,7.60,6.55,1.94)
                       + drawbar(ph2, 8.,0.,8.,9.32,21.0,3.17,7.60,6.55,1.94))
                + 0.01155*sin(ph*7.0) + 0.0198*sin(ph*9.0);
            env = gmADSR(age, held, 0.03, 0.05, 0.95, 0.10);
            vib = 0.4;
        } else if(prog == 22u){                   // Harmonica (bright reed + vibrato)
            sig = 0.6*sin(ph)+0.3*sin(ph*2.0)+0.2*sin(ph*3.0)+0.12*sin(ph*4.0);
            env = gmADSR(age, held, 0.02, 0.05, 0.95, 0.08);
            vib = 1.5;
        } else {                                   // 23 Tango Accordion (bandoneon)
            sig = 0.33*(drawbar(ph,  8.,0.,8.,10.07,21.77,3.20,2.16,2.87,1.83)
                      + drawbar(ph2, 8.,0.,8.,10.07,21.77,3.20,2.16,2.87,1.83))
                + 0.00363*sin(ph*7.0) + 0.0165*sin(ph*9.0) + 0.03036*sin(ph*10.0);
            env = gmADSR(age, held, 0.05, 0.08, 0.95, 0.12);
            vib = 0.7;
        }
        if(vib > 0.0) env *= 1.0 + vib*0.06*sin(TAU*6.2*age);   // Leslie/vibrato AM
        return vec2(sig*env*vel);
    }

    uint fam = prog >> 3u;   // 0 pf,1 chromperc,2 org,3 gtr,4 bass,5 str,6 ens,
                             // 7 brass,8 reed,9 pipe,10 lead,11 pad,12 fx,13 eth,14 perc,15 sfx
    float g = 0.85;
    if(fam == 0u){                                   // ---- PIANO / EP / harpsi / clav
        if(prog==4u||prog==5u){                       // e-piano (DX tine)
            sig = sin(ph + 2.2*exp2(-age*5.0)*sin(ph));
            env = gmStruck(age, held, 2.0, 0.10);
        } else if(prog==6u||prog==7u){                // harpsichord / clav (SC-55 profile)
            sig = sin(ph) + 0.508*sin(ph*2.0) + 0.440*sin(ph*3.0) + 0.199*sin(ph*4.0)
                + 0.0385*sin(ph*5.0) + 0.184*sin(ph*6.0) + 0.068*sin(ph*7.0)
                + 0.148*sin(ph*8.0) + 0.180*sin(ph*9.0);
            env = gmStruck(age, held, 1.3, 0.06) * smoothstep(0.0,0.07,age); g=0.5;
        } else {                                      // acoustic / electric grand (SC-55 profile)
            sig = sin(ph) + 0.242*sin(ph*2.0) + 0.033*sin(ph*3.0) + 0.111*sin(ph*4.0)
                + 0.125*sin(ph*5.0) + 0.030*sin(ph*6.0) + 0.0022*sin(ph*7.0) + 0.0048*sin(ph*8.0);
            env = gmStruck(age, held, 3.0, 0.10) * smoothstep(0.0,0.04,age);
        }
    } else if(fam == 1u){                            // ---- CHROMATIC PERCUSSION
        if(prog==11u){                                // vibraphone: metal-bar 4th mode dominant
            sig = sin(ph) + 0.692*sin(TAU*fract(4.0*f*age))
                + 0.010*sin(TAU*fract(2.0*f*age)) + 0.0273*sin(TAU*fract(3.0*f*age))
                + 0.0040*sin(TAU*fract(5.0*f*age));
            env = gmStruck(age, held, 1.30, 0.10); g=0.6; vib=1.0;   // rings + rotor tremolo
        } else if(prog==12u){                          // marimba: fundamental + tuned 4:1 & 10:1 bar modes
            float a = sin(TAU*fract(f*age) + 1.5*exp(-80.*age)*sin(TAU*fract(7.*f*age)))
                    + 0.0491*sin(TAU*fract(4.0*f*age))
                    + 0.0019*sin(TAU*fract(7.0*f*age))
                    + 0.0286*sin(TAU*fract(10.0*f*age));
            return vec2(a * exp(-3.2*age)) * (0.7+0.6*vel);
        } else if(prog==13u){                          // xylophone -> user's FM marimba (fast decay)
            return vec2(sin(TAU*fract(f*age) + 1.5*exp(-80.*age)*sin(TAU*fract(7.*f*age)))
                        * exp(-9.0*age)) * (0.7+0.6*vel);
        } else if(prog==14u){                          // tubular bell: 2f/4f dominant, near-absent fundamental
            sig = 1.485e-5*sin(TAU*fract(f*age)) + 0.7026*sin(TAU*fract(2.0*f*age))
                + 0.001328*sin(TAU*fract(3.0*f*age)) + 0.2859*sin(TAU*fract(4.0*f*age))
                + 5.347e-5*sin(TAU*fract(5.0*f*age)) + 2.178e-5*sin(TAU*fract(6.0*f*age))
                + 1.003e-5*sin(TAU*fract(7.0*f*age)) + 1.041e-5*sin(TAU*fract(8.0*f*age))
                + 9.80e-6*sin(TAU*fract(9.0*f*age)) + 9.18e-6*sin(TAU*fract(10.0*f*age));
            env = gmStruck(age, held, 1.00, 0.20); g=1.0;
        } else {                                       // celesta/glock/musicbox / dulcimer
            float dec = (prog<=10u) ? 2.5 : 7.0;
            float mi  = (prog<=10u) ? 4.0 : 2.0;                   // bell inharmonicity
            sig = sin(ph + mi*exp2(-age*6.0)*sin(ph*3.51));
            env = gmStruck(age, held, dec, 0.05); g=1.0;
        }
    } else if(fam == 3u){                            // ---- GUITAR (plucked)
        if(prog==24u){                                // nylon acoustic (mellow)
            sig = 0.30*gtrHarm(ph, 0.606,0.197,0.183,0.027,0.040,0.062,0.091,0.003,0.033,0.007,0.007);
            env = gmStruck(age, held, 1.1, 0.08); g=1.0;
        } else if(prog==25u){                         // steel acoustic (bright mid formant)
            sig = 0.26*gtrHarm(ph, 0.607,0.151,0.390,0.348,0.230,0.077,0.033,0.024,0.009,0.020,0.025);
            env = gmStruck(age, held, 0.9, 0.08); g=1.0;
        } else if(prog==27u){                         // clean electric (very bright, comb formants)
            sig = 0.105*gtrHarm(ph, 1.595,3.884,0.747,1.172,3.245,0.246,0.015,0.017,0.142,0.210,1.732);
            env = gmStruck(age, held, 0.7, 0.08); g=1.0;
        } else if(prog==30u){                         // distortion (rich, sustained)
            sig = 0.16*gtrHarm(ph, 1.420,1.815,1.370,0.233,0.825,0.641,0.658,0.311,0.244,0.167,0.457);
            env = gmStruck(age, held, 0.45, 0.10); g=1.0;
        } else {                                       // 26 jazz, 28 muted, 29 overdrive, 31 harmonics
            float br = (prog>=28u) ? 0.14 : 0.22;      // muted/od less bright decay tail
            sig = pluckAdd(ph, age, br);
            if(prog==29u) sig = tanh(sig*3.0);          // overdrive
            env = gmStruck(age, held, 0.8, 0.08); g=1.0;
        }
    } else if(fam == 4u){                            // ---- BASS (user's vBass shape)
        float cut = 3.0 + 11.0*exp2(-age*6.0); float s=0.0;
        for(int h=1;h<=10;h++){ float hh=float(h); if(f*hh>15000.0) break;
            s += (1.0/hh)*exp(-0.07*max(hh-cut,0.0))*sin(ph*hh); }
        sig = tanh(s*1.6)*0.7 + sin(ph)*0.6;
        bool syn = prog>=38u;
        env = gmADSR(age, held, syn?0.006:0.004, 0.18, syn?0.75:0.5, 0.09); g=0.95;
    } else if(fam == 5u || fam == 6u){               // ---- STRINGS / ENSEMBLE / CHOIR
        if(prog==45u){                                // pizzicato: low-passed pluck (SC-55 profile)
            sig = sin(ph) + 0.40*sin(ph*2.0) + 0.25*sin(ph*3.0) + 0.155*sin(ph*4.0)
                + 0.045*sin(ph*5.0) + 0.03*sin(ph*6.0) + 0.007*sin(ph*9.0);
            env = gmStruck(age, held, 3.0, 0.05); g=0.5;
        } else if(prog==46u){                         // harp
            sig = pluckAdd(ph, age, 0.18); env = gmStruck(age, held, 0.7, 0.08); g=1.0;
        } else if(prog==47u){                         // timpani (modal drum)
            sig = tanh(sin(TAU*(f*age + f*0.1*(1.0-exp2(-age*20.0))))*exp2(-age*5.0)*1.5);
            env = 1.0; g=1.2;
        } else {                                      // bowed strings / string ensemble / choir
            float det = (fam==6u) ? 0.008 : 0.004;    // ensembles wider
            sig = 0.34*(sawAdd(ph,14) + sawAdd(TAU*f*(1.0+det)*age,14) + sawAdd(TAU*f*(1.0-det)*age,14));
            bool choir = (prog>=52u && prog<=54u);
            if(choir) sig = 0.42*(sin(ph) + 0.95*sin(ph*2.0) + 0.70*sin(ph*3.0) + 0.05*sin(ph*4.0)
                                + 0.012*sin(ph*5.0) + 0.004*sin(ph*6.0) + 0.013*sin(ph*7.0)
                                + 0.032*sin(ph*8.0) + 0.031*sin(ph*9.0) + 0.021*sin(ph*10.0)); // SC-55 vowel formants
            env = gmADSR(age, held, choir?0.16:0.14, 0.2, 0.9, 0.25); vib = 0.6;
        }
    } else if(fam == 7u){                            // ---- BRASS (FM, attack swell)
        float bright = (prog==56u||prog==59u) ? 2.4 : (prog==58u) ? 0.9 : 1.6; // tpt bright, tuba mellow
        float I = (bright + 1.6*vel)*(1.0 - exp2(-age*26.0))*exp2(-age*1.2) + 0.7;
        sig = sin(ph + I*gmfb(ph, 0.4));
        env = gmADSR(age, held, 0.045, 0.10, 0.85, 0.10); vib = 0.3;
    } else if(fam == 8u){                            // ---- REED (sax/oboe/clarinet/bassoon)
        sig = reedAdd(ph, int(prog));                 // SC-55-matched additive partials
        sig += nse(floor(age*30000.0))*0.03*exp2(-age*10.0);   // breath attack
        float ra=0.035, rs=0.85;
        if(prog==64u){ ra=0.045; rs=0.88; }
        else if(prog==66u){ ra=0.035; rs=0.82; }
        else if(prog==68u){ ra=0.033; rs=0.73; }
        else if(prog==71u||prog==67u){ ra=0.028; rs=0.66; }
        env = gmADSR(age, held, ra, 0.06, rs, 0.10); vib = 0.6;
    } else if(fam == 9u){                            // ---- PIPE (flute/recorder/whistle/pan)
        if(prog < 75u){                              // flute / recorder: strong 2nd & 3rd
            sig = sin(ph) + 1.05*sin(ph*2.0) + 0.78*sin(ph*3.0)
                + 0.076*sin(ph*4.0) + 0.023*sin(ph*5.0) + 0.026*sin(ph*6.0)
                + 0.013*sin(ph*7.0) + 0.0038*sin(ph*8.0) + 0.0018*sin(ph*9.0);
            sig *= 0.30;
            sig += (nse(floor(age*30000.0)) - nse(floor(age*30000.0)-1.0))*0.035;
        } else {                                     // pan flute / airy: near-sine, breathy 3rd
            sig = sin(ph) + 0.022*sin(ph*2.0) + 0.100*sin(ph*3.0)
                + 0.006*sin(ph*4.0) + 0.015*sin(ph*5.0)
                + 0.0018*sin(ph*7.0) + 0.0015*sin(ph*9.0);
            sig *= 0.72;
            sig += (nse(floor(age*30000.0)) - nse(floor(age*30000.0)-1.0))*0.085;  // airier breath
        }
        env = gmADSR(age, held, 0.05, 0.05, 0.95, 0.10); vib = 1.0; g=0.95;
    } else if(fam == 10u){                           // ---- SYNTH LEAD
        sig = (prog==80u||prog==87u) ? sqrAdd(ph,15)      // square lead / bass-lead
            : (prog==81u||prog==84u) ? sawAdd(ph,16)      // saw / charang
            : sin(ph + 2.0*gmfb(ph,0.5));                 // FM-ish
        env = gmADSR(age, held, 0.01, 0.1, 0.8, 0.10); vib = 0.5;
    } else if(fam == 11u){                           // ---- SYNTH PAD (lush, slow)
        sig = 0.34*(sawAdd(ph,12) + sawAdd(TAU*f*1.010*age,12) + sawAdd(TAU*f*0.990*age,12));
        env = gmADSR(age, held, 0.35, 0.4, 0.85, 0.6); vib = 0.3; g=0.8;
    } else if(fam == 13u){                           // ---- ETHNIC (mostly plucked)
        sig = pluckAdd(ph, age, 0.25); env = gmStruck(age, held, 0.9, 0.08); g=1.0;
    } else if(fam == 14u){                           // ---- PERCUSSIVE (112-119), SC-55 fitted
        if(prog==114u){                               // Steel Drums (metallic: strong h2 + h4)
            sig = sin(ph) + 1.772*sin(2.0*ph) + 0.792*sin(4.0*ph)
                + 0.081*sin(5.0*ph) + 0.036*sin(6.0*ph) + 0.038*sin(8.0*ph);
            env = gmStruck(age, held, 2.9, 0.10) * smoothstep(0.0,0.02,age); g=0.62;
        } else if(prog==116u){                        // Taiko drum
            sig = sin(ph) + 0.61*sin(2.0*ph) + 0.37*sin(3.0*ph) + 0.79*sin(4.0*ph) + 0.62*sin(5.0*ph);
            env = gmStruck(age, held, 3.2, 0.08); g=0.75;
        } else if(prog==117u || prog==118u){          // Melodic / Synth Tom (near-pure, drum decay)
            sig = sin(ph) + 0.20*sin(2.0*ph);
            env = gmStruck(age, held, 3.0, 0.08); g=1.0;
        } else if(prog==115u){                        // Woodblock (short bright click)
            sig = sin(ph) + 1.38*sin(2.0*ph) + 0.30*sin(5.0*ph) + 0.32*sin(6.0*ph);
            env = gmStruck(age, held, 22.0, 0.02); g=0.85;
        } else if(prog==112u || prog==113u){          // Tinkle Bell / Agogo (h2-dominant metallic ping)
            sig = 0.25*sin(ph) + sin(2.0*ph) + 0.10*sin(3.0*ph) + 0.06*sin(4.0*ph);
            env = gmStruck(age, held, 6.0, 0.04); g=0.85;
        } else {                                      // Reverse Cymbal (119) / Synth Drum — noisy swell
            sig = sin(ph + 3.0*exp2(-age*4.0)*sin(ph*2.5)) + nse(floor(age*20000.0))*0.3;
            env = smoothstep(0.0,0.4,age)*exp2(-max(age-held,0.0)*3.0); g=0.6;
        }
    } else {                                         // ---- SYNTH FX / SFX (best-effort)
        sig = sin(ph + 3.0*exp2(-age*4.0)*sin(ph*2.5));
        env = gmStruck(age, held, 1.5, 0.15);
    }
    if(vib > 0.0) env *= 1.0 + vib*0.05*sin(TAU*5.5*age);
    return vec2(sig*env*vel*g);
}

vec2 mainSound(int samp, float timeNow){
    float SR   = MIDI_SAMPLE_RATE;
    float qSec = float(MIDI_TIME_Q_SAMPLES)/SR;
    float tPhase = float(samp)/SR;
    float songLen= float(MIDI_END_TICK)*qSec;
    float tEvent = mod(tPhase, songLen);

    uint T = uint(floor(tEvent/qSec + 0.5));
    T = min(T, MIDI_END_TICK-1u);
    uint b = T >> MIDI_BLOCK_SHIFT_TICKS;
    if(b >= MIDI_BLOCK_COUNT) return vec2(0.0);
    uint padT = uint(1.2/qSec + 0.5);

    float L=0.0, R=0.0, nrg=0.0;
    for(uint j=0u; j<(samp<0 ? 99999u : MIDI_LOOKBACK_BLOCKS); j++){
        if(b < j) break;
        uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
        for(uint k=0u; k<count; k++){
            uint si=start+k; if(si>=MIDI_SEG_COUNT) break;
            uint onT=onTick(si); if(T<onT) continue;
            uint dT=durTick(si), offT=onT+dT;
            if(T > offT+padT) continue;

            uint m=meta16(si); int nn=metaNote(m); float vel=metaVel(m);
            float onSec=float(onT)*qSec;
            float ageOn=max(0.0, tEvent-onSec);
            float held =float(dT)*qSec;
            uint prog=noteProg(si);
            if(prog==128u){                                    // channel-10 drum hit
                if(ageOn > 2.5) continue;                      // one-shot decay window
                vec2 d = gmDrum(nn, ageOn, vel)*vel;           // drums carry their own stereo
                L += d.x; R += d.y;
                nrg += 0.35*vel*exp(-ageOn*3.0);
                continue;
            }
            if(ageOn > held + 1.2) continue;
            float nfx = ageOn/max(float(gateTick(si))*qSec, 1e-4);
            float bnd = noteBend(si, nfx);
            float xg  = noteExpr(si, nfx);
            float f   = NoteToHz(nn)*exp2(bnd/12.0);

            vec2 v = gmVoice(prog, f, ageOn, held, vel) * xg;
            float pan = clamp((float(nn)-64.0)/48.0, -1.0, 1.0)*0.25;
            L += v.x*(0.5-pan*0.5); R += v.y*(0.5+pan*0.5);
            nrg += 0.3*vel*xg;
        }
    }
    float lvl = nrg*0.5;
    float g   = 1.0/sqrt(1.0 + lvl*lvl);
    vec2 lr = softLimit(vec2(L,R)*g*0.9);
    return lr*smoothstep(0.0, 0.4, tPhase);
}
"""

def build_gm(tpq, tempo, tracks, bpm=None, pad_sec=1.2, tempo_map=None, trim_s=None):
    data, notes, info = build_buffer(tpq, tempo, tracks, bpm, pad_sec=pad_sec,
                                     tempo_map=tempo_map, trim_s=trim_s, gm=True)
    return data + GM_PLAYER_CODE, info

def build_player(tpq, tempo, tracks, bpm=None, pad_sec=1.2, tempo_map=None, trim_s=None,
                 budget_vec4=None):
    return build_buffer(tpq, tempo, tracks, bpm, pad_sec=pad_sec, tempo_map=tempo_map,
                        trim_s=trim_s, budget_vec4=budget_vec4)[0] + PLAYER_CODE

# ============================================================ VIZ (--viz)
# "LED Band Spectro 3D" (Merry Xmas '25, Orblivious — based on shadertoy wcVBRh)
# as a Buffer A pass. ShaderToy does NOT expose the Sound tab's FFT as a texture
# input, so row 0's spectrum is computed straight from the embedded MIDI note
# data (midiSpec: active notes -> gaussian bumps on a log-frequency axis, plus
# a few harmonics so it reads like a real FFT). Everything else — the 32-row
# history buffer, attack/release smoothing, the mirrored 5-wall tunnel — is the
# original shader unchanged, self-fed via iChannel0 = Buffer A.
VIZ_BUFFERA_CODE = r"""
// ── MIDI-driven spectrum (replaces the iChannel1 FFT texture) ────────────────
// Same block-indexed scan as the Sound tab; a separate ShaderToy pass, so the
// helper names may repeat the Sound tab's — they never collide.
float NoteToHz(int n){ return 440.0*exp2((float(n)-69.0)/12.0); }
uint  onTick(uint i){ return midiOM(i)&0xFFFFu; }
uint  meta16(uint i){ return midiOM(i)>>16; }
uint  durTick(uint i){ uint w=midiDur(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
int   metaNote(uint m){ return MIDI_NOTE_MIN+int(m&0x7Fu); }
float metaVel(uint m){ return float((m>>7)&127u)/127.0; }

// envelope mirror of the Sound tab (the bars must track what you HEAR)
#ifndef ENV_ATTACK
#define ENV_ATTACK  0.02
#endif
#ifndef ENV_DECAY
#define ENV_DECAY   0.09
#endif
#ifndef ENV_SUSTAIN
#define ENV_SUSTAIN 0.72
#endif
#ifndef ENV_RELEASE
#define ENV_RELEASE 0.18
#endif
float envLvl(float age){
    if(age < ENV_ATTACK) return smoothstep(0.0, ENV_ATTACK, age);
    float d = age - ENV_ATTACK;
    if(d < ENV_DECAY)    return mix(1.0, ENV_SUSTAIN, d/ENV_DECAY);
    return ENV_SUSTAIN;
}
float vizNoteEnv(float age, float held){
    if(age < 0.0) return 0.0;
    return (age < held) ? envLvl(age) : envLvl(held)*exp(-(age-held)/ENV_RELEASE);
}

// energy near band u (0..1 across VIZ_NOTE_LO..HI, i.e. log-frequency) at iTime
float midiSpec(float u){
    float SR = MIDI_SAMPLE_RATE, qSec = float(MIDI_TIME_Q_SAMPLES)/SR;
    float songLen = float(MIDI_END_TICK)*qSec;
    float tEvent  = mod(iTime, songLen);
    uint T = uint(floor(tEvent/qSec + 0.5));
    T = min(T, MIDI_END_TICK-1u);
    uint b = T >> MIDI_BLOCK_SHIFT_TICKS;
    if(b >= MIDI_BLOCK_COUNT) return 0.0;
    uint padT = uint(1.2/qSec + 0.5);
    float target = VIZ_NOTE_LO + u*(VIZ_NOTE_HI - VIZ_NOTE_LO);
    float e = 0.0;
    for(uint j=0u; j<(iFrame<0 ? 99999u : MIDI_LOOKBACK_BLOCKS); j++){  // de-unroll: iFrame<0 is never true at runtime but ANGLE can't fold it (see the Sound tab)
        if(b < j) break;
        uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
        for(uint k=0u; k<count; k++){
            uint si=start+k; if(si>=MIDI_SEG_COUNT) break;
            uint onT=onTick(si); if(T<onT) continue;
            uint dT=durTick(si), offT=onT+dT;
            if(T > offT+padT) continue;
            uint m=meta16(si); float fn=float(metaNote(m)); float vel=metaVel(m);
            float env = vizNoteEnv(tEvent-float(onT)*qSec, float(dT)*qSec)*vel;
            if(env < 0.003) continue;
            // fundamental + octave/fifth harmonics on the semitone axis
            float d0=(fn       - target)/VIZ_BAND_SEMIS;
            float d1=(fn+12.0  - target)/VIZ_BAND_SEMIS;
            float d2=(fn+19.02 - target)/VIZ_BAND_SEMIS;
            float d3=(fn+24.0  - target)/VIZ_BAND_SEMIS;
            e += env*( exp(-d0*d0) + 0.45*exp(-d1*d1) + 0.25*exp(-d2*d2) + 0.15*exp(-d3*d3) );
        }
    }
    return 1.0 - exp(-e*VIZ_GAIN);   // soft-normalize into 0..1 like an FFT bin
}

/*
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓                                                      ▓
    ▓                 LED Band Spectro 3D                  ▓
    ▓                                                      ▓
    ▓         -** Merry Xmas '25 from Orblivious **-       ▓
    ▓                                                      ▓
    ▓    Based on: https://www.shadertoy.com/view/wcVBRh   ▓
    ▓                                                      ▓
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
*/


const int numWalls = 5;

#define HISTORY_ROWS 32
#define DECAY_RATE 0.1

mat2 R2(float a){ return mat2(cos(a), -sin(a), sin(a), cos(a)); }

vec3 spectrumColor(float t, float energy) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.33, 0.67);
    vec3 col = a + b * cos(6.283 * (c * t + d));
    return col * (1.0 + energy * 1.5);
}

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hitPlane(vec3 ro, vec3 rd, vec3 n, float dist) {
    float denom = dot(rd, n);
    if (denom < 0.0) {
        float t = (-dist - dot(ro, n)) / denom;
        if (t > 0.0) return t;
    }
    return 1e8;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    ivec2 ifc = ivec2(fragCoord);
    ivec2 res = ivec2(iResolution.xy);
    float t = iTime;

    // ============ SPECTRUM HISTORY BUFFER (top rows) ============

    float alpha = 0.0;   // [mid2glsl] out params are write-only; original read fragColor.a here
    if (ifc.y >= res.y - HISTORY_ROWS) {
        int row = res.y - 1 - ifc.y;


        if (row == 0) {
            // Row 0: Calculate band spectrum by averaging frequency bins
            float u = float(ifc.x) / float(res.x);

            // Number of bands we want
            float numBands = 32.0;
            float bandWidth = 1.0 / numBands;

            // Which band is this pixel in?
            float bandIndex = floor(u * numBands);
            float bandStart = bandIndex / numBands;
            float bandEnd = (bandIndex + 1.0) / numBands;

            // Sample multiple points within this band and average
            float spec = 0.0;
            const int samplesPerBand = 8;

            for (int s = 0; s < samplesPerBand; s++) {
                float sampleU = bandStart + (float(s) + 0.5) / float(samplesPerBand) * bandWidth;
                spec += midiSpec(sampleU);   // [mid2glsl] was texture(iChannel1, ...).r — MIDI data instead of FFT
            }
            spec /= float(samplesPerBand);

            // Shape the response
            spec =  pow(spec, 0.6) * 2.5;

            // Get previous value for smoothing
            float prev = texelFetch(iChannel0, ifc, 0).a;

            // Fast attack, slow release
            if (spec > prev) {
                alpha = mix(prev, spec, 0.5);
            } else {
                alpha = max(spec, prev * DECAY_RATE);
            }
        } else {
            // Other rows: Copy from row above with decay
            ivec2 aboveCoord = ivec2(ifc.x, ifc.y + 1);
            float aboveVal = texelFetch(iChannel0, aboveCoord, 0).a;
            alpha = aboveVal * DECAY_RATE;
        }

        // Only write alpha, RGB stays zero


    }

    // ============ MAIN TUNNEL RENDERING ============
    vec3 col = vec3(0.0);

    // Helper to read spectrum from buffer (stored in alpha)
    #define getSpectrum(freq, histRow) texelFetch(iChannel0, ivec2(clamp(int(freq * float(res.x)), 0, res.x-1), res.y - 1 - histRow), 0).a

    float bassHit = getSpectrum(0.08, 0);

    for(int m = 0; m < 8; m++)
    {
        vec2 offset = vec2(m % 3, m / 3) / 3.0 - 0.33;
        vec2 uv = (2.0 * (fragCoord + offset) - iResolution.xy) / iResolution.y;

        float time = t + float(m) * 0.015;
        vec3 ro = vec3(0, 0, time * 1.8);
        vec3 rd = normalize(vec3(uv, 1.0));

        float alph = 1.0;
        float fogDt = 1.0;

        for(int i = 0; i < 4; i++)
        {
            const float PI = 3.14159265;

            float dt = 1e8;
            vec3 n;
            vec3 p;

            for(int w = 0; w < numWalls; w++)
            {
                float angle = float(w) * 2.0 * PI / float(numWalls);
                float spiralTwist = ro.z * 0.25;
                angle += spiralTwist;

                vec3 wallNormal = vec3(cos(angle), sin(angle), 0.0);
                float tunnelR = 2.3 + bassHit * 0.5;
                float dist = hitPlane(ro, rd, wallNormal, tunnelR);

                if(dist < dt)
                {
                    dt = dist;
                    n = wallNormal;
                    p = ro + rd * dist;
                }
            }

            if (i == 0) fogDt = dt;

            float wallAngle = atan(n.y, n.x);
            float spiralOffset = p.z * 0.25;

            vec2 pRot = vec2(
                p.x * cos(-wallAngle) - p.y * sin(-wallAngle),
                p.x * sin(-wallAngle) + p.y * cos(-wallAngle)
            );

            vec2 tuv = vec2(p.z, pRot.y + spiralOffset);

            // ========== SPECTRUM GRID ==========
            float numBands = 12.0;
            float numLevels = 8.0;

            vec2 gridScale = vec2(numBands * 0.18, numLevels * 0.35);
            vec2 gridUv = tuv * gridScale;

            vec2 cellId = floor(gridUv);
            vec2 cellUv = fract(gridUv) - 0.5;

            // ========== READ FROM HISTORY BUFFER ==========
            float bandIndex = mod(cellId.x, numBands);
            float freqBand = bandIndex / numBands;

            // Sample from history - deeper = older history
            int historyIndex = clamp(int(mod(p.z * 0.3, float(HISTORY_ROWS - 1))), 0, HISTORY_ROWS - 1);
            float audioLevel = getSpectrum(freqBand, historyIndex);

            // Also get current for blending
            float audioCurrent = getSpectrum(freqBand, 0);
            audioLevel = mix(audioCurrent, audioLevel, 0.5);

            // Boost it!


            // Bass boost

            if ((i^m)%2==0)
              audioLevel = 1.-min(audioLevel,1.);
            audioLevel *= 1.0 + (1.0 - freqBand) * 0.6;
           audioLevel*=audioLevel;
            // ========== WAVE PROPAGATION ==========
            float wavePhase = t * 3.0 - bandIndex * 0.15;
            float wave = sin(wavePhase) * 0.5 + 0.5;
            float wave2 = sin(t * 2.3 + bandIndex * 0.2) * 0.5 + 0.5;

            float dynamicLevel = audioLevel * (0.6 + wave * 0.4) * (0.8 + wave2 * 0.2);

            // ========== LEVEL ANIMATION ==========
            float levelPos = mod(cellId.y, numLevels);
            float normalizedLevel = levelPos / numLevels;
            float centeredLevel = abs(normalizedLevel - 0.5) * 2.0;

            float flutter = sin(t * 8.0 + bandIndex * 0.5 + levelPos * 0.3) * 0.05;
            float threshold = centeredLevel + flutter;

            float dotOn = smoothstep(threshold + 0.12, threshold - 0.08, dynamicLevel);

            // ========== PEAK INDICATOR ==========
            float peakPos = dynamicLevel * numLevels * 0.5;
            float distFromPeak = abs(levelPos - numLevels * 0.5) - peakPos;
            float isPeak = smoothstep(0.8, 0.0, abs(distFromPeak));
            isPeak *= sin(t * 12.0 + bandIndex) * 0.3 + 0.7;

            // ========== DOT SIZE ==========
            float basePulse = sin(t * 6.0 + bandIndex * 0.4) * 0.1 + 0.9;
            float audioPulse = 1.0 + audioLevel * 0.4;

            float dynamicRad = 0.28 * basePulse * audioPulse;
            dynamicRad *= 0.8 + dotOn * 0.4;
            dynamicRad += isPeak * 0.1;

            float sphere = length(cellUv) - dynamicRad;
            float sphereMask = smoothstep(0.06, -0.03, sphere);

            // ========== GLOW ==========
            float glowStrength = 1.0 + audioLevel * 3.0;
            float glow = exp(-sphere * glowStrength) * (0.4 + audioLevel * 0.6);
            glow += exp(-sphere * 2.0) * pow(bassHit, 2.0) * 2.0 * dotOn;

            // ========== COLORS ==========
            float colorShift = freqBand * 0.8 + t * 0.1;
            vec3 bandColor = spectrumColor(colorShift, audioLevel);
            vec3 peakColor = spectrumColor(colorShift + 0.3, 1.0) * 2.0;
            vec3 hotColor = vec3(1.0, 0.95, 0.9);

            // ========== MIX ==========
            // Only draw if dot is on
            if (dotOn > 0.01) {
                float intensity = dotOn * (dynamicLevel * 3.0);

                vec3 layerCol = bandColor * sphereMask * intensity;
                layerCol = mix(layerCol, peakColor, sphereMask * isPeak);
                layerCol = mix(layerCol, hotColor * 3.0, sphereMask * intensity * audioLevel * 0.5);

                layerCol += bandColor * glow * dotOn * 0.7;
                layerCol += peakColor * glow * isPeak * 0.5;
                layerCol += bandColor * exp(-sphere * 1.2) * dotOn * 0.15;

                col += layerCol * alph;
            }
            alph *= 0.45;

            rd = reflect(rd, n);
            rd.xy *= R2(float((i) % 2 == 0 ? -1 : 1) * ro.z * 0.5 * (1.0 + float(i) * 0.5));
            // Each layer spins at different speed
            ro = p + n * 0.001;
        }

        float fogDensity = 0.12 + bassHit * 0.08;
        float fog = 1.2 / (1.0 + fogDt * fogDt * fogDensity);
        col *= fog;
    }

    // Post-processing
    col /= 6.5/exp(.5 * bassHit);

    vec2 q = fragCoord / iResolution.xy;
    col *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);

    col = pow(col, vec3(0.9));
    col = mix(vec3(dot(col, vec3(0.299, 0.587, 0.114))), col, 1.2);

    // Final output with alpha = 1
    fragColor = vec4(col, alpha);
}
"""

VIZ_IMAGE_CODE = r"""// mid2glsl --viz: display pass. Set iChannel0 = Buffer A.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    fragColor = vec4(texelFetch(iChannel0, ivec2(fragCoord), 0).rgb, 1.0);
}
"""

# ============================================================ VIZ "melee" (--viz melee / 4)
# "Look behind you, a three headed monkey!" — Melee Island, ShaderToy ldKGDW by MMGS (2016). The
# shader is VERBATIM (author credit kept in the Image tab). iChannel0/1 = the artist's two textures
# (wired via _STJ_MELEE_TEX); iChannel3 "music" ← the MIDI-spectrum Buffer A (img_buf_channel=3).
VIZ_MELEE_IMAGE_CODE = r"""////
/// Look behind you, a three headed monkey!
/// "Melee Island" (ldKGDW) MMGS 2016 + choreographed running monkey.
/// mid2glsl viz 4: iChannel3 = MIDI-spectrum Buffer A (stars pulse to music).
#define HUGE 1000000.0
#define PI 3.14159

vec3 ballPos;
vec3 lightDir = normalize(vec3(1,2,-2));
float gRun = 1.0;         // 1 = running (legs march), 0 = standing still
float gAnimSpeed = 3.0;   // leg cadence (higher = faster run)
float gHeadYaw = 0.0;     // head turn (looking around) independent of the body
float gBodyYaw = 0.0;     // whole-body facing (rotated in place -> no screen shift)

// ---- monkey SDF helpers (from the running-monkey shader) ----
vec3 rotatePitch(vec3 v, float a){ return vec3(v.x, v.y*cos(a)-v.z*sin(a), v.z*cos(a)+v.y*sin(a)); }
vec3 rotateYaw(vec3 v, float a){ return vec3(v.x*cos(a)-v.y*sin(a), v.y*cos(a)+v.x*sin(a), v.z); }
vec3 rotatePitchAbout(vec3 v, vec3 c, float a){ return rotatePitch(v-c,a)+c; }
vec3 rotateYawAbout(vec3 v, vec3 c, float a){ return rotateYaw(v-c,a)+c; }
float calcLight(vec3 n, vec3 rd, float sh){
    float amb=0.3, dif=max(dot(-lightDir,n),0.0), spe=max(dot(-lightDir,reflect(rd,n)),0.0);
    spe*=spe*spe*spe*spe*spe*spe*spe*spe; return amb+dif+spe*sh;
}
float sphereSdf(vec3 p, vec3 c, float r){ return distance(p,c)-r; }
float ellipsoidSdf(vec3 pos, vec3 c, vec3 r){ vec3 p=pos-c; float k0=length(p/r), k1=length(p/(r*r)); return k0*(k0-1.0)/k1; }
float cylinderSdf(vec3 pos, vec3 c, float h, float r){ vec3 p=pos-c; vec2 d=abs(vec2(length(p.xy),p.z))-vec2(r,h); return min(max(d.x,d.y),0.0)+length(max(d,0.0)); }
float roundConeSdf(vec3 pos, vec3 c, float r1, float r2, float h){
    vec3 p=pos-c; float b=(r1-r2)/h, a=sqrt(1.0-b*b); vec2 q=vec2(length(p.xy),p.z);
    float k=dot(q,vec2(-b,a)); if(k<0.0) return length(q)-r1; if(k>a*h) return length(q-vec2(0.0,h))-r2; return dot(q,vec2(a,b))-r1;
}
float capsuleSdf(vec3 p, vec3 a, vec3 b, float r){ vec3 pa=p-a, ba=b-a; float h=clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0); return length(pa-ba*h)-r; }
float opSmoothUnion(float d1, float d2, float k){ float h=clamp(0.5+0.5*(d2-d1)/k,0.0,1.0); return mix(d2,d1,h)-k*h*(1.0-h); }
float atan_fast(float a, float b){ float r=atan(a/b); if(b<0.0) r+=PI; return r; }
float spiral(vec2 p){ float rad=length(p); float ang=atan_fast(p.y,p.x)+PI*0.75; float x=rad*4.0-ang; x=fract(x/(2.0*PI)); return 1.0+4.0*x*(x-1.0); }
float animLimbAngle(){ return gRun * sin(iTime*gAnimSpeed*PI); }   // legs/arms (0 standing, faster=run)
float animEars(){ float t=fract(iTime/3.0)*3.0; t*=50.0; if(t>=4.0*PI) t=0.0; return sin(t); }
float animBlink(){ const float bs=0.1, fr=3.5; float t=fract(iTime/fr)*fr; return 2.0*max((bs-abs(t-bs))/bs,0.0); }
float xzdiscSdf(vec3 pos, vec3 c, float radius, float s){ radius-=s; vec3 p=pos-c; vec2 d=abs(vec2(length(p.xz),p.y))-vec2(radius,0.0); return min(max(d.x,d.y),0.0)+length(max(d,0.0))-s; }

float globalSdf(vec3 pos, out vec3 color){
    pos.z -= 8.0;   // center the monkey vertically for the billboard camera
    // whole-body facing: rotate the model in place about its own vertical axis
    // (no screen translation, unlike orbiting the camera)
    { float bc=cos(-gBodyYaw), bs=sin(-gBodyYaw); pos.xy = mat2(bc,-bs,bs,bc)*pos.xy; }
    vec3 bodyColor=vec3(0.3,0.2,0.1), skinColor=vec3(0.8,0.7,0.6), shirtColor=vec3(0.8,0.4,0.2), eyeColor=vec3(0.1);
    color = bodyColor;
    vec3 ear=vec3(4.6,0,14), shoulder=vec3(1.5,0,10), elbow=vec3(3.5,0,8), hand=vec3(4.5,-1,6),
         hip=vec3(2,0,4), knee=vec3(2.1,-0.5,2), ankle=vec3(2.2,0,0), toes=vec3(2.2,-2,-0.5);
    float angle=animLimbAngle();
    pos.x += angle*0.25;
    if(pos.x<0.0) angle=-angle;
    elbow=rotatePitchAbout(elbow,shoulder,angle); hand=rotatePitchAbout(hand,shoulder,angle);
    knee=rotatePitchAbout(knee,hip,-angle); ankle=rotatePitchAbout(ankle,hip,-angle); toes=rotatePitchAbout(toes,hip,-angle);
    ear=rotateYawAbout(ear,vec3(2,0,14),animEars()*0.15);
    float blink=animBlink();
    vec3 pm=vec3(abs(pos.x),pos.y,pos.z);
    // head assembly rotates by gHeadYaw about the vertical (Z) axis -> looks around
    vec3 hp=pos; { float hc=cos(-gHeadYaw), hs=sin(-gHeadYaw); hp.xy=mat2(hc,-hs,hs,hc)*pos.xy; }
    vec3 hpm=vec3(abs(hp.x),hp.y,hp.z);
    float d=HUGE;
    float dHead=cylinderSdf(hp,vec3(0,0,15),1.0,0.8)-2.0;
    dHead=opSmoothUnion(dHead,ellipsoidSdf(hp,vec3(0,-1,12.5),vec3(2.5,2.0,1.0)),2.0);
    float dTuft=opSmoothUnion(sphereSdf(hp,vec3(0,-1.5,18.2),0.5),capsuleSdf(hp,vec3(0,0,18.5),vec3(0,1,19),0.0),2.5);
    dHead=min(dHead,dTuft); d=min(d,dHead);
    float dEar=xzdiscSdf(hpm,ear,2.1,0.5);
    if(dEar<0.1 && hpm.y<ear.y) dEar+=0.5*spiral(ear.xz-hpm.xz);
    d=min(d,dEar);
    float dBody=roundConeSdf(pos,vec3(0,0,6.5),3.2,2.2,3.0); d=min(d,dBody);
    float dArm=min(capsuleSdf(pm,shoulder,elbow,0.5),capsuleSdf(pm,elbow,hand,0.5)); d=min(d,dArm);
    float dLeg=min(capsuleSdf(pm,hip,knee,0.6),capsuleSdf(pm,knee,ankle,0.6)); d=min(d,dLeg);
    float dHand=sphereSdf(pm,hand,1.0); d=min(d,dHand);
    float dFoot=sphereSdf(pm,ankle,1.0); dFoot=opSmoothUnion(dFoot,capsuleSdf(pm,toes-vec3(0.5,0,0),toes+vec3(0.5,0,0),0.5),2.0); d=min(d,dFoot);
    // colors (face features use the head-rotated coords)
    if(dHead<=d && hp.y<-1.0){
        vec3 eye=vec3(1.2*sign(hp.x),-2.0,14.7); vec3 dist=hp-eye; dist.z*=0.6;
        if(hp.z<=eye.z+1.0-blink && dist.x*dist.x+dist.z*dist.z<0.25 && abs(dist.z)*2.0>-dist.x) color=eyeColor;
        else if(hp.z<13.3 && abs(hp.x*hp.x*hp.x*hp.x*0.0625-hp.x*hp.x*0.25-hp.z+13.0)<0.1) color=vec3(0.2);
        else if(hp.z<15.0 || distance(hp,eye)<2.2) color=skinColor;
    }
    if(dEar<=d && hpm.y<ear.y+0.1) color=skinColor;
    if(dBody<=d && pos.y>0.0 && dEar+0.3<dHead) color=skinColor;
    if(dArm<=d && distance(pm,shoulder)<2.5) color=shirtColor;
    if(dHand<=d || dFoot<=d) color=skinColor;
    return d;
}
vec3 calcNormal(vec3 pos, float d){
    float e=0.02; vec3 dm;
    vec3 v=vec3(globalSdf(pos-vec3(e,0,0),dm), globalSdf(pos-vec3(0,e,0),dm), globalSdf(pos-vec3(0,0,e),dm));
    return normalize(d-v);
}
// billboard monkey; d in local units. Body facing is set via the gBodyYaw global
// (model rotates in place -> stays screen-centered at dd regardless of facing).
vec4 renderMonkey(vec2 dd){
    ballPos=vec3(0.0);
    vec3 ro=vec3(dd.x*13.0, -34.0, 8.0 + dd.y*14.0);   // fixed ortho-ish front camera
    vec3 rd=vec3(0.0,1.0,0.0);
    float t=0.0;
    for(int i=0;i<(iChannelResolution[0].x<0.0?99999:45);i++){   // de-unroll
        vec3 pos=ro+rd*t; vec3 oc; float ds=globalSdf(pos,oc);
        if(ds<0.06){ vec3 n=calcNormal(pos,ds); return vec4(oc*calcLight(n,rd,0.5), 1.0); }
        t+=ds; if(t>70.0) break;
    }
    return vec4(0.0);
}
// road centerline (screen x) at screen height sy.
// Matches meleeScene()'s road-draw uv-chain exactly (verified vs a sentinel render):
// the backdrop shifts uv.x by a net +0.1 before the road test, so add it here too.
float roadX(float sy){
    float ry=sy-0.1;
    float rxc=0.3375 - cos(-ry*45.0 - ry)*0.09 - 0.02 + ry*0.09;
    return rxc + 0.2 - sy*0.23*cos(sy*10.2);
}

// ---- Melee Island backdrop (MMGS 2016) ----
vec3 meleeScene(vec2 fragCoord){
    vec2 uv = fragCoord.xy/iResolution.xy;
    vec3 col=vec3(0,0,-0.3*uv.y+1.9-uv.y-uv.y*.5);
    vec3 tex=texture(iChannel0,uv*10.5).xyz+texture(iChannel0,uv*11.5).xyz;
    vec3 music=texture(iChannel3,uv).xyz;
    if(uv.y>0.6){
        if(cos(uv.x*24.)>0.99993&&cos(uv.y*24.)>0.995||cos(uv.x*24.)>0.99&&cos(uv.y*24.)>0.9998) col+=vec3(music.x,music.x,0)*0.8;
        if(cos(uv.x*26.)>0.99993&&cos(uv.y*26.)>0.995||cos(uv.x*26.)>0.99&&cos(uv.y*26.)>0.9998) col+=vec3(music.y,music.y,0.3)*0.8;
        if(cos(uv.x*23.)>0.99993&&cos(uv.y*23.)>0.995||cos(uv.x*23.)>0.99&&cos(uv.y*23.)>0.9998) col+=vec3(music.z,music.z,0.5)*0.8;
        if(tex.x<0.7) col=vec3(uv.y*0.5);
    }
    vec3 cloud=vec3(0);
    if(uv.y>0.2){
        col*=1.5*uv.y; vec2 p1=vec2(0.8); float r=0.2;
        r+=cos(uv.y*12.+0.2*sin(uv.x*11.+iTime*0.01))+sin(uv.x*1.5+iTime*0.01)-cos(uv.x*4.+iTime*0.01)*0.1;
        if(length(uv-p1)-r<0.) col=col+uv.y*.1;
        p1=vec2(0.9); r=0.6; r+=cos(uv.y*12.+0.2*sin(uv.x*16.+iTime*0.02))+sin(uv.x*4.5+iTime*0.01)-cos(uv.x*4.+iTime*0.01)*0.1;
        if(length(uv-p1)-r<0.) col=col+uv.y*.4;
        if(length(uv-p1)-r<0.) col=col-uv.y*.2;
        p1=vec2(0.4); r=0.1; r+=cos(uv.y*12.+0.2*sin(uv.x*26.+iTime*0.02))+sin(uv.x*5.5+iTime*0.06)-cos(uv.x*4.+iTime*0.01)*0.1;
        if(length(uv-p1)-r<0.) col=col+uv.y*.34;
        cloud=col;
    }
    uv.x+=0.3;
    if(0.4+uv.y-0.9*uv.x+(uv.x*uv.x)*0.6+cos(uv.x*30.-uv.x*22.+cos(uv.x*25.))*0.05<.5) col=vec3(0.1,0.1,0.5)+tex*0.01;
    if(0.4+uv.y-0.9*uv.x+(uv.x*uv.x)*0.65+cos(uv.x*30.-uv.x*22.+cos(uv.x*25.))*0.05<.5) col=vec3(0.1,0.11,0.4)+tex*0.01;
    if(uv.y<0.6){
        uv.x-=0.3;
        if(0.2+uv.y-0.8*uv.x+(uv.x*uv.x)*.8+cos(uv.x*30.-uv.x*23.)*0.2<.4) col=vec3(0.1,0.12,.35)+tex*0.01;
        uv.x-=0.2;
        if(0.2+uv.y-1.3*uv.x+(uv.x*uv.x)*3.5+cos(uv.x*30.-uv.x*16.)*0.05<.6) col=vec3(0.1,0.13,0.34)+tex*0.01;
        if(uv.y-uv.x+(0.18+uv.x*(uv.x-0.5)*3.+cos(uv.x*10.-uv.x*22.)*0.04)*2.<.3) col=vec3(0.1,0.14,.45)+tex*0.01;
        if(uv.y-uv.x+(0.2+uv.x*(uv.x-0.5)*2.9+cos(uv.x*29.-uv.x*18.)*0.05)*2.<.3) col=vec3(0.1,0.15,0.35)+tex*0.01;
    }
    uv.x+=0.1;
    // road
    if(uv.y<0.61&&uv.y>0.1){
        uv.x-=0.1-uv.y*0.23*cos(uv.y*10.2); uv.y-=0.1;
        if(uv.x<0.34-cos(-uv.y*45.-uv.y)*0.09-0.02+uv.y*0.09 && uv.x>0.335-cos(-uv.y*45.-uv.y)*0.09-0.02+uv.y*0.09) col*=0.5+vec3(1.);
        if(uv.x<0.34-cos(-uv.y*45.-uv.y)*0.09-0.02+uv.y*0.09 && uv.x>0.335-cos(-uv.y*2.-uv.y)*0.09-0.03+uv.y*0.09) col*=0.95;
    }
    uv=fragCoord.xy/iResolution.xy;
    if(uv.y<0.2+cos(uv.x*22.+sin(uv.y*222.)*2.)*0.02-uv.y*2.-sin(uv.x*3.)*0.2) col=0.5+vec3(0.2,0.2,0.8);
    // outpost
    uv=fragCoord.xy/iResolution.xy; uv.x-=0.51; uv.y-=0.15;
    if(uv.y>0.46&&uv.y-.0*uv.x+uv.x*uv.x*90.<.50) col=tex*0.5+cos(iTime*21.)*0.02-uv.x*5.;
    uv.y+=0.006;
    if(uv.y>0.46&&uv.y+.22*uv.x+uv.x*uv.x*200.<.5) col=vec3(0,0.13,.9)+cloud;
    uv=fragCoord.xy/iResolution.xy;
    if(uv.y>0.6&&uv.y<0.613&&uv.x<0.54&&uv.x>0.45||uv.y>0.6&&uv.y<0.64&&uv.x<0.47&&uv.x>0.45) col=vec3(0.2,0.2,.4)*0.5+tex*0.1+cos(iTime*21.)*0.01;
    uv.x-=0.03;
    if(uv.y>0.61&&uv.y<0.62+cos(iTime*4.0)*0.001&&uv.x<0.47-cos(iTime*5.)*0.001&&uv.x>0.46) col+=vec3(1.5,0.2,0.26)*1.5;
    // melee town
    uv=fragCoord.xy/iResolution.xy;
    if(uv.y<0.25&&uv.y>0.04&&uv.x<0.45&&uv.x>0.26){
        vec3 tex2=texture(iChannel0,uv*1.5).xyz*vec3(1,1,0); uv.y-=0.1;
        vec3 t3=texture(iChannel1,uv*2.9).xyz*0.3;
        if(t3.x>0.20){col-=.12-t3; if(tex2.x>0.88) col+=tex2;}
        uv=fragCoord.xy/iResolution.xy; t3=texture(iChannel1,uv*3.).xyz*0.3;
        if(t3.x>0.2){col-=.12-t3; if(tex2.x>0.85) col+=tex2;}
    }
    if(sin(uv.y*2222.+uv.x*2.)<0.5) col*=.95;
    col*=0.5+uv.y*0.9;
    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec3 col = meleeScene(fragCoord);

    // ---- choreography: sprint across the ocean -> vanish -> climb -> rest ----
    const float T_DASH  = 2.5;    // sprint across the water
    const float T_GAP   = 0.5;    // off-screen (vanish)
    const float T_CLIMB = 15.0;   // climb the winding road
    const float SY_BOT = 0.135, SY_TOP = 0.60;          // climb all the way to the summit plateau
    const float BASE_SCALE = 0.076, TOP_SCALE = 0.065;  // ~28% smaller: mountain-bottom -> summit
    const float REST_SX = 0.43;                         // rest spot: just LEFT of the chimney/outpost
    const float T_END  = T_DASH + T_GAP + T_CLIMB;      // climb ends here
    const float T_TURN = 1.2;                           // graceful summit turn-around
    float sx, sy, scale; bool draw = true;

    if(iTime < T_DASH){                                // PHASE 1: SPRINT along the BOTTOM edge, big
        float d = iTime / T_DASH;
        sx = mix(-0.22, 1.28, d);                      // left edge -> off the right edge (vanishes)
        sy = 0.012 + 0.028*abs(sin(d*PI*7.0));         // feet on the window bottom + fast running bob
        scale = 0.173;                                 // big-ish (~28% smaller)
        gBodyYaw = -PI*0.5;                            // side profile, FACING right (run direction)
        gAnimSpeed = 8.0;                              // FAST legs (a real run)
    } else if(iTime < T_DASH + T_GAP){                 // vanished (off-screen), before the climb
        draw = false; sx=0.0; sy=0.0; scale=0.1;
    } else if(iTime < T_END){                          // PHASE 2: climb from the hill base (smaller)
        float c = (iTime - T_DASH - T_GAP) / T_CLIMB;
        sy = mix(SY_BOT, SY_TOP, c);
        sx = roadX(sy);
        float dxdy = (roadX(sy+0.01) - sx) / 0.01;      // path tangent
        scale = mix(BASE_SCALE, TOP_SCALE, c);         // half-size at the base -> summit
        // gentle lean into the curve, faded out over the last stretch so it
        // arrives at the summit cleanly back-to-us (no hard whip near the top)
        float bank = clamp(dxdy*0.4, -0.7, 0.7) * (1.0 - smoothstep(0.82, 1.0, c));
        gBodyYaw = PI + bank;
        gAnimSpeed = 3.5;
    } else if(iTime < T_END + T_TURN){                 // PHASE 3: turn around at the summit (smooth)
        float u = smoothstep(0.0, 1.0, (iTime - T_END) / T_TURN);
        sy = SY_TOP; sx = mix(roadX(SY_TOP), REST_SX, u); // step left off the road to the chimney
        scale = TOP_SCALE;
        gBodyYaw = PI + PI*u;                           // rotate back(PI) -> facing us(2PI==0)
        gRun = 1.0 - u;                                 // legs wind down to a stop
        gAnimSpeed = 3.5;
    } else {                                           // PHASE 4: rest, only the head turns
        float ti = iTime - (T_END + T_TURN);
        sy = SY_TOP; sx = REST_SX;                      // standing just left of the chimney
        scale = TOP_SCALE;
        gBodyYaw = 0.0;                                // faces us, perfectly still
        gRun = 0.0;                                    // no leg/arm motion
        gHeadYaw = 0.55*sin(ti*0.8);                   // head looks left <-> right
    }

    if(!draw){ fragColor = vec4(col, 1.0); return; }
    vec2 uv = fragCoord/iResolution.xy;
    vec2 dLoc = (uv - vec2(sx, sy)) / scale;   // model feet are at dLoc.y~0 -> feet planted at sy
    dLoc.x *= iResolution.x/iResolution.y;
    vec4 mk = renderMonkey(dLoc);
    col = mix(col, mk.rgb, mk.a);
    fragColor = vec4(col, 1.0);
}
"""

# The artist's two source textures (ShaderToy media) — iChannel0 = terrain/star grain, iChannel1 = town.
# Passed as extra_image_inputs for --viz melee (Buffer A takes iChannel3). mipmap/repeat/vflip, per ldKGDW.
_STJ_MELEE_TEX = [
    {"channel": 0, "type": "texture", "id": "4dXGzn",
     "filepath": "/media/a/0c7bf5fe9462d5bffbd11126e82908e39be3ce56220d900f633d58fb432e56f5.png",
     "sampler": {"filter": "mipmap", "wrap": "repeat", "vflip": "true", "srgb": "false", "internal": "byte"}},
    {"channel": 1, "type": "texture", "id": "4sf3Rr",
     "filepath": "/media/a/ad56fba948dfba9ae698198c109e71f118a54d209c0ea50d77ea546abad89c57.png",
     "sampler": {"filter": "mipmap", "wrap": "repeat", "vflip": "true", "srgb": "false", "internal": "byte"}},
]

def _viz_defs(note_min, note_max):
    """VIZ_* defines computed per song (band axis spans the song's own note
    range, min 3 octaves). Prepended to constant GLSL blocks — no .format on
    them, so their braces stay untouched."""
    lo, hi = float(note_min - 3), float(note_max + 7)
    if hi - lo < 36.0:
        c = (hi + lo) * 0.5
        lo, hi = c - 18.0, c + 18.0
    band = max(1.2, (hi - lo) / 32.0 * 0.9)   # gaussian half-width ~ one display band
    return (f"#define VIZ_NOTE_LO {lo:.1f}\n"
            f"#define VIZ_NOTE_HI {hi:.1f}\n"
            f"#define VIZ_BAND_SEMIS {band:.2f}\n"
            "#define VIZ_GAIN 1.4\n")

def build_viz_buffera(note_min, note_max):
    return ("// --- Buffer A: LED Band Spectro 3D (set THIS pass's iChannel0 = Buffer A) ---\n"
            + _viz_defs(note_min, note_max) + VIZ_BUFFERA_CODE)

# ============================================================ VIZ "cd" (--viz cd)
# "Soundform CD" (orblivius; fork of prasound's Soundform, shadertoy 3XyBDw /
# 7fc3D8): raymarched CD with spectrum ring etched on the data band, diffraction
# rainbows, XorDev wave-field reflections/background, mouse-drag rotation.
# Port notes vs the original 4-pass ShaderToy JSON:
#   * Buffer A's audio column read a music input (512x2 FFT/wave texture);
#     ShaderToy exposes no Sound-tab audio, so col-0 FFT is midiSpec() and the
#     waveform is SYNTHESIZED by a mono mirror of the Sound tab's mainSound
#     (512 texels/frame — trivial). Everything else (mouse texels, scroll
#     history, min/max columns) is unchanged.
#   * The Image never samples its cubemap input (iChannel1) nor Buffer B
#     (iChannel2) — dead fork leftovers — so this build needs only
#     Common + Sound + Buffer A + Image. Keyboard (iChannel3) is optional:
#     keys 1/2 toggle the wave/FFT rings.
CD_BUFFERA_CODE = r"""
// ── MIDI audio feed (replaces the iChannel3 music texture) ───────────────────
// Separate ShaderToy pass — helper names may repeat the Sound tab's.
#define TAU 6.28318530718
float NoteToHz(int n){ return 440.0*exp2((float(n)-69.0)/12.0); }
uint  onTick(uint i){ return midiOM(i)&0xFFFFu; }
uint  meta16(uint i){ return midiOM(i)>>16; }
uint  durTick(uint i){ uint w=midiDur(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
int   metaNote(uint m){ return MIDI_NOTE_MIN+int(m&0x7Fu); }
float metaVel(uint m){ return float((m>>7)&127u)/127.0; }

float fbsin(float p,float fb){ float y=sin(p); y=sin(p+fb*y); y=sin(p+fb*y); return y; }
// TDM's hash (Bamboo bass, shadertoy XldcRr) — per-tap reverb jitter
vec2 rvHash(float p){ return fract(sin(p * vec2(110.082, 175.025)) * vec2(19871.8972, 11571.72)); }
float softLimit1(float x){
    const float T=0.85, HEAD=1.0-T;
    float ax=abs(x), over=max(ax-T,0.0);
    return sign(x)*(min(ax,T) + HEAD*over/(over+HEAD));
}

// envelope + voice: verbatim mirrors of the Sound tab (waveform must match what you HEAR)
#ifndef ENV_ATTACK
#define ENV_ATTACK  0.02
#endif
#ifndef ENV_DECAY
#define ENV_DECAY   0.09
#endif
#ifndef ENV_SUSTAIN
#define ENV_SUSTAIN 0.72
#endif
#ifndef ENV_RELEASE
#define ENV_RELEASE 0.18
#endif
#ifndef USE_REVERB
#define USE_REVERB 1
#endif
#ifndef REVERB_MIX
#define REVERB_MIX 0.42
#endif
#ifndef REVERB_TAPS
#define REVERB_TAPS 16        // stochastic echo taps; raise (48-128) for a denser tail
#endif
#ifndef REVERB_DUR
#define REVERB_DUR 0.9        // reverb length, seconds
#endif
#ifndef REVERB_FALL
#define REVERB_FALL 3.0       // tap decay (exp2 exponent per second); lower = longer tail
#endif
#ifndef PAD_SEC
#define PAD_SEC 1.2           // how long a dead note stays in the scan (must cover the reverb tail)
#endif
#ifndef MASTER_GAIN
#define MASTER_GAIN 0.45
#endif
#ifndef POLY_COMP
#define POLY_COMP   0.5
#endif
float envLvl(float age){
    if(age < ENV_ATTACK) return smoothstep(0.0, ENV_ATTACK, age);
    float d = age - ENV_ATTACK;
    if(d < ENV_DECAY)    return mix(1.0, ENV_SUSTAIN, d/ENV_DECAY);
    return ENV_SUSTAIN;
}
float vizNoteEnv(float age, float held){
    if(age < 0.0) return 0.0;
    return (age < held) ? envLvl(age) : envLvl(held)*exp(-(age-held)/ENV_RELEASE);
}
#ifndef VOICE
#define VOICE 0
#endif
#if VOICE==1
// "piano man" piano — same injected block as the Sound tab (waveform must match)
//__PIANOMAN__
#elif VOICE==2
//__EPIANO__
#elif VOICE==3
//__ORGAN__
#elif VOICE==4
//__PIANO2__
#elif VOICE==5
//__EPIANO2__
#elif VOICE==6
//__PIANO3__
#elif VOICE==7
//__PIANO4__
#elif VOICE==8
//__PIANO5__
#elif VOICE==9
//__PIANO6__
#elif VOICE==10
//__PIANO7__
#else
float voice(float f, float ageOn, float vel){
    float ph = TAU*f*ageOn;
    float I  = (1.0+2.4*vel)*exp2(-ageOn*3.5) + 0.6;
    float m  = I*fbsin(ph, 0.5);
    float dec= 0.55 + 0.45*exp2(-ageOn*0.7);
    return (sin(ph+m)+sin(ph*1.004+m))*0.5*dec;
}
#endif
#if VOICE==0
float noteSig(float f, float held, float vel, float age){
    if(age < 0.0) return 0.0;
    float e = (age < held) ? envLvl(age) : envLvl(held)*exp(-(age-held)/ENV_RELEASE);
    return voice(f, age, vel) * e * vel;
}
#endif

// energy near band u (0..1 across VIZ_NOTE_LO..HI, log-frequency) at iTime
float midiSpec(float u){
    float SR = MIDI_SAMPLE_RATE, qSec = float(MIDI_TIME_Q_SAMPLES)/SR;
    float songLen = float(MIDI_END_TICK)*qSec;
    float tEvent  = mod(iTime, songLen);
    uint T = uint(floor(tEvent/qSec + 0.5));
    T = min(T, MIDI_END_TICK-1u);
    uint b = T >> MIDI_BLOCK_SHIFT_TICKS;
    if(b >= MIDI_BLOCK_COUNT) return 0.0;
    uint padT = uint(1.2/qSec + 0.5);
    float target = VIZ_NOTE_LO + u*(VIZ_NOTE_HI - VIZ_NOTE_LO);
    float e = 0.0;
    for(uint j=0u; j<(iFrame<0 ? 99999u : MIDI_LOOKBACK_BLOCKS); j++){  // de-unroll: iFrame<0 never true at runtime, opaque to ANGLE
        if(b < j) break;
        uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
        for(uint k=0u; k<count; k++){
            uint si=start+k; if(si>=MIDI_SEG_COUNT) break;
            uint onT=onTick(si); if(T<onT) continue;
            uint dT=durTick(si), offT=onT+dT;
            if(T > offT+padT) continue;
            uint m=meta16(si); float fn=float(metaNote(m)); float vel=metaVel(m);
            float env = vizNoteEnv(tEvent-float(onT)*qSec, float(dT)*qSec)*vel;
            if(env < 0.003) continue;
            float d0=(fn       - target)/VIZ_BAND_SEMIS;
            float d1=(fn+12.0  - target)/VIZ_BAND_SEMIS;
            float d2=(fn+19.02 - target)/VIZ_BAND_SEMIS;
            float d3=(fn+24.0  - target)/VIZ_BAND_SEMIS;
            e += env*( exp(-d0*d0) + 0.45*exp(-d1*d1) + 0.25*exp(-d2*d2) + 0.15*exp(-d3*d3) );
        }
    }
    return 1.0 - exp(-e*VIZ_GAIN);
}

// mono synth sample at song-time tPhase — mirrors mainSound (L+R)/2, so the
// waveform ring shows the actual audio incl. ADSR, reverb, poly gain, limiter
float synthWave(float tPhase){
    float SR   = MIDI_SAMPLE_RATE;
    float qSec = float(MIDI_TIME_Q_SAMPLES)/SR;
    float songLen= float(MIDI_END_TICK)*qSec;
    float tEvent = mod(tPhase, songLen);
    uint T = uint(floor(tEvent/qSec + 0.5));
    T = min(T, MIDI_END_TICK-1u);
    uint b = T >> MIDI_BLOCK_SHIFT_TICKS;
    if(b >= MIDI_BLOCK_COUNT) return 0.0;
    uint padT = uint(PAD_SEC/qSec + 0.5);
    float M=0.0, nrg=0.0;
#if VOICE>=1
    int ZERO = iFrame>>31;            // 0 at runtime; opaque → keeps voice loops rolled
#endif
    for(uint j=0u; j<(iFrame<0 ? 99999u : MIDI_LOOKBACK_BLOCKS); j++){  // de-unroll
        if(b < j) break;
        uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
        for(uint k=0u; k<count; k++){
            uint si=start+k; if(si>=MIDI_SEG_COUNT) break;
            uint onT=onTick(si); if(T<onT) continue;
            uint dT=durTick(si), offT=onT+dT;
            if(T > offT+padT) continue;
            uint m=meta16(si); int nn=metaNote(m); float vel=metaVel(m);
            float onSec=float(onT)*qSec, offSec=float(offT)*qSec;
            float ageOn=max(0.0, tEvent-onSec);
            float held =float(dT)*qSec;
            float ageOff=(tEvent>offSec)?(tEvent-offSec):0.0;
#if VOICE>=1
            if(ageOn > 6.0) continue;
            vec2 pno = pianoStrings(float(nn), ageOn, held, vel, ZERO);
            M += (pno.x + pno.y);         // mono sum (body modes omitted: visual-only)
            nrg += 0.35*vel*exp(-ageOn*2.0);
#if VOICE>=2 && VOICE!=6 && USE_REVERB
            {   // bamboo-style reverb (mono mirror)
                float rdt = REVERB_DUR/float(REVERB_TAPS);
                vec2 rsum = vec2(0.0), rout = vec2(0.0);
                for(int ri=0; ri<(ZERO + REVERB_TAPS); ri++){
                    vec2 rnd = rvHash(float(ri));
                    vec2 rt  = float(ri)*rdt + rdt*rnd*5.0;
                    vec2 ra  = exp2(-rt*REVERB_FALL);
                    ra *= vec2(rnd.x < 0.5 ? -1.0 : 1.0, rnd.y < 0.5 ? -1.0 : 1.0);  // polarity: no coherent LF pileup
                    rout.x += pianoWet(float(nn), ageOn-rt.x, held, vel, ZERO).x*ra.x;
                    rout.y += pianoWet(float(nn), ageOn-rt.y, held, vel, ZERO).y*ra.y;
                    rsum += abs(ra);
                }
                rout /= rsum;
                M += REVERB_MIX*(rout.x + rout.y);
            }
#endif
#else
            float f = NoteToHz(nn);
            float e = (tEvent<offSec) ? envLvl(ageOn) : envLvl(held)*exp(-ageOff/ENV_RELEASE);
            if(e < 0.0004) continue;
            float s = voice(f, ageOn, vel)*e*vel;
            nrg += e*vel;
            M += s;                       // pan cancels in the mono (L+R) sum
#if USE_REVERB
            {   // bamboo-style stochastic reverb (mono mirror of the Sound tab)
                float rdt = REVERB_DUR/float(REVERB_TAPS);
                vec2 rsum = vec2(0.0), rout = vec2(0.0);
                for(int ri=0; ri<(iFrame<0 ? 99999 : REVERB_TAPS); ri++){  // de-unroll
                    vec2 rnd = rvHash(float(ri));
                    vec2 rt  = float(ri)*rdt + rdt*rnd*5.0;
                    vec2 ra  = exp2(-rt*REVERB_FALL);
                    ra *= vec2(rnd.x < 0.5 ? -1.0 : 1.0, rnd.y < 0.5 ? -1.0 : 1.0);  // polarity: no coherent LF pileup
                    rout.x += noteSig(f,held,vel,ageOn-rt.x)*ra.x;
                    rout.y += noteSig(f,held,vel,ageOn-rt.y)*ra.y;
                    rsum += abs(ra);
                }
                rout /= rsum;
                M += REVERB_MIX*(rout.x + rout.y);
            }
#endif
#endif
        }
    }
    float lvl = nrg*POLY_COMP;
    float g   = MASTER_GAIN * inversesqrt(1.0 + lvl*lvl);
    return softLimit1(0.5*M*g) * smoothstep(0.0, 0.4, tPhase);
}

#define NN 64
#define R iResolution.xy
#define PI 3.1415926
void mainImage( out vec4 o, in vec2 p ) {
    // Shadertoy loves resizing buffers without notice
    if (iFrame < 9) { o = vec4(0); return; }

    o = texelFetch(iChannel0, ivec2(p) - ivec2(1,0), 0);

      // texel (3,0): (yaw, pitch, prevMouseX, prevMouseY)
   // (3,0): (yaw, pitch, spin, _)
   if (int(p.x) == 3 && int(p.y) == 0) {
        vec4 st = texelFetch(iChannel0, ivec2(3,0), 0);
        vec2 pm = texelFetch(iChannel0, ivec2(4,0), 0).xy;
        float yaw=st.x, pitch=st.y, spin=st.z;

        if (iFrame < 10) { yaw = 0.2; pitch = 0.6; spin = 0.0; }

        if (iMouse.z > 0.) {
            vec2 m = iMouse.xy;
            vec2 delta = (pm.x < 0.) ? vec2(0.) : (m - pm);
            yaw   +=  delta.x * 6.2831853 / R.x;
            pitch +=  delta.y * 6.2831853 / R.y;
        }
        spin -= clamp(iTimeDelta, 0., 0.05) * 0.5;
        o = vec4(yaw, pitch, spin, 0.);
    }

    // (4,0): prevMouse
    if (int(p.x) == 4 && int(p.y) == 0) {
        o = vec4((iMouse.z > 0.) ? iMouse.xy : vec2(-1.), 0., 0.);
    }

    if (int(p.x) == 0) {
        int py = int(p.y);   // [mid2glsl --viz cd] was texelFetch(iChannel3 music, 512x2); rows >=512 read 0 there too
        o.x = (py < 512) ? midiSpec(float(py)/511.0) : 0.0;                                   // abs(FFT) = 0..1, 512 bins
        o.y = (py < 512) ? 0.5 + 0.5*synthWave(iTime + (float(py)-511.0)/MIDI_SAMPLE_RATE) : 0.0; // waveform = 0..1, 512 samples
    }

    // No vertex shaders on Shadertoy...
    if (int(p.x) == 1 && int(p.y) % NN == 0 && int(p.y) < 512) {
        o.zw = vec2(2,-1); // min..max of waveform[x..x+32]

        for (int i = 0; i < (iChannelResolution[0].x<0.0?99999:NN); i++) {   // de-unroll
            float wave = texelFetch(iChannel0, ivec2(p) + ivec2(-1,i), 0).y;
            o.z = min(o.z, wave);
            o.w = max(o.w, wave);
        }
    }

    // No vertex shaders on Shadertoy...
    if (int(p.x) == 2) {
        float r = p.y/R.y; // find wave samples that are close to r
        o.z = 0.03 * smoothstep(0.003, 0., abs(r - iTimeDelta*120.));

        for (int y = 0; y < (iChannelResolution[0].x<0.0?99999:512); y += NN) {   // de-unroll
            vec2 range = texelFetch(iChannel0, ivec2(1,y), 0).zw;
            if (r < range.x || r > range.y) continue;
            for (int i = 0; i < (iChannelResolution[0].x<0.0?99999:NN); i++) {   // de-unroll
                float wave = texelFetch(iChannel0, ivec2(1,y+i), 0).y;
                o.z += 0.03 * smoothstep(0.003, 0., abs(wave - r));
            }
        }
    }
}
"""

# (voice tokens in CD_BUFFERA_CODE are resolved by inject_voices() at build)

CD_IMAGE_CODE = r"""// Fork of "Soundform" by prasound. https://shadertoy.com/view/3XyBDw
// CD spectrum viz + reflections + XorDev field background
//
// Channels (Image tab):
//   iChannel0 = Buffer A    (audio cols 0-2, orientation texel (3,0))
//   iChannel3 = keyboard    (optional; keys 1/2 toggle the wave/FFT rings)
// [mid2glsl --viz cd] audio comes from the embedded MIDI data via Buffer A;
// the original cubemap (iChannel1) and Buffer B (iChannel2) inputs were never
// sampled by this pass, so they are omitted.

#define MAX_STEPS 100
#define MAX_DIST  20.0
#define SURFACE_DIST 0.001
#define WAVELENGTH_MULT 500.0
#define AA 2
#define PI 3.1415926

#define CD_SCALE   1.5
#define ETCHING_SCALE 1.4

#define VIZ_GAIN   1.0
#define REFL_STEPS 60.
#define BG_STEPS   90.
#define REFR_STEPS 30.

mat3 gRot;

mat3 rotateX(float t){ float c=cos(t),s=sin(t); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 rotateY(float t){ float c=cos(t),s=sin(t); return mat3(c,0,s, 0,1,0, -s,0,c); }
mat3 rotateZ(float t){ float c=cos(t),s=sin(t); return mat3(c,s,0, -s,c,0, 0,0,1); }
mat3 identity(){ return mat3(1,0,0, 0,1,0, 0,0,1); }

mat3 cdRot(){
    vec4 s = texelFetch(iChannel0, ivec2(3,0), 0);
    return rotateX(s.y) * rotateY(s.x) * rotateZ(s.z);
}

float sdCylinder(vec3 p, vec3 a, vec3 b, float r){
    vec3 ba=b-a, pa=p-a;
    float baba=dot(ba,ba), paba=dot(pa,ba);
    float x=length(pa*baba-ba*paba)-r*baba;
    float y=abs(paba-baba*0.5)-baba*0.5;
    float x2=x*x, y2=y*y*baba;
    float d=(max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
    return sign(d)*sqrt(abs(d))/baba;
}

vec2 surfaceMin(vec2 a, vec2 b){ return (a.x>b.x)?b:a; }

vec2 getDist(vec3 p){
vec3 diskPos=vec3(0,0,3);
    float scale = CD_SCALE;                        // <-- >1 bigger, <1 smaller
    mat3 rot = gRot;
    vec3 dp=((p-diskPos)/scale)*rot+diskPos;  // divide by scale



    float outerDisk         = sdCylinder(dp, diskPos+vec3(0,0,0.01), diskPos-vec3(0,0,0.01), 1.015);
    float outerDiskSubtract = sdCylinder(dp, diskPos+vec3(0,0,0.1 ), diskPos-vec3(0,0,0.1 ), 1.0);
    float disk              = sdCylinder(dp, diskPos+vec3(0,0,0.01), diskPos-vec3(0,0,0.01), 1.0);
    float diskSubtract      = sdCylinder(dp, diskPos+vec3(0,0,0.1 ), diskPos-vec3(0,0,0.1 ), 0.35);
    float mirrorBand        = sdCylinder(dp, diskPos+vec3(0,0,0.01), diskPos-vec3(0,0,0.01), 0.5);
    float mirrorBandSub     = sdCylinder(dp, diskPos+vec3(0,0,0.1 ), diskPos-vec3(0,0,0.1 ), 0.295);
    float plasticHub        = sdCylinder(dp, diskPos+vec3(0,0,0.01), diskPos-vec3(0,0,0.01), 0.295);
    float plasticHubDivot   = sdCylinder(dp, diskPos+vec3(0,0,0.1 ), diskPos-vec3(0,0,0.1 ), 0.26);
    float phDivotSub0       = sdCylinder(dp, diskPos+vec3(0,0,0.1 ), diskPos-vec3(0,0,0.1 ), 0.2575);
    float phDivotSub1       = sdCylinder(dp, diskPos+vec3(0,0,0.008),diskPos-vec3(0,0,0.008),0.5);
    float hole              = sdCylinder(dp, diskPos+vec3(0,0,1   ), diskPos-vec3(0,0,1   ), 0.115);

    vec2 outerDiskM = vec2(max(outerDisk,-outerDiskSubtract), 0.0);
    vec2 diskM      = vec2(max(disk,-diskSubtract),           1.0);
    vec2 mirrorM    = vec2(max(mirrorBand,-mirrorBandSub),    2.0);
    plasticHubDivot = max(plasticHubDivot, -min(phDivotSub0, phDivotSub1));
    float plasticHubSub = min(plasticHubDivot, hole);
    vec2 plasticHubM = vec2(max(plasticHub,-plasticHubSub),  0.0);


    return surfaceMin(surfaceMin(surfaceMin(outerDiskM, diskM), mirrorM), plasticHubM);
}

vec3 getNormal(vec3 p){
    float d=getDist(p).x; vec2 e=vec2(0.01,0);
    vec3 n=d-vec3(getDist(p-e.xyy).x, getDist(p-e.yxy).x, getDist(p-e.yyx).x);
    return normalize(n);
}

vec2 rayMarch(vec3 ro, vec3 rd){
    float dO=0.0; vec2 distCol=vec2(0);
    for(int i=0;i<(iChannelResolution[0].x<0.0?99999:MAX_STEPS);i++){   // de-unroll
        vec3 p=ro+rd*dO; distCol=getDist(p);
        float dS=distCol.x; dO+=dS;
        if(dO>MAX_DIST||abs(dS)<SURFACE_DIST) break;
    }
    return vec2(dO,distCol.y);
}

vec3 getLight(vec3 p, vec3 n, vec3 c, vec3 lp){
    vec3 l=normalize(lp-p); return clamp(dot(n,l)*c,0.0,1.0);
}
vec3 getSpecular(vec3 p, vec3 n, vec3 c, vec3 lp, vec3 ro, float sp){
    vec3 l=normalize(lp-p), h=normalize(l+normalize(ro-p));
    return c*pow(max(0.0,dot(n,h)),sp);
}

vec3 bump3y(vec3 x, vec3 yo){ vec3 y=vec3(1.0)-x*x; return clamp(y-yo,0.0,1.0); }
vec3 spectralZucconi6(float w){
    float x=clamp((w-400.0)/300.0,0.0,1.0);
    const vec3 c1=vec3(3.54585104,2.93225262,2.41593945);
    const vec3 x1=vec3(0.69549072,0.49228336,0.27699880);
    const vec3 y1=vec3(0.02312639,0.15225084,0.52607955);
    const vec3 c2=vec3(3.90307140,3.21182957,3.96587128);
    const vec3 x2=vec3(0.11748627,0.86755042,0.66077860);
    const vec3 y2=vec3(0.84897130,0.88445281,0.73949448);
    if(w<400.0||w>700.0) return vec3(0);
    return bump3y(c1*(x-x1),y1)+bump3y(c2*(x-x2),y2);
}
vec3 getDiffraction(vec3 l, vec3 rd, float d, vec3 difr){
    float u=abs(dot(l,difr)-dot(rd,difr));
    if(u==0.0) return vec3(0);
    vec3 color=vec3(0);
    for(int i=1;i<=12;i++) color+=spectralZucconi6(u*d/float(i)*WAVELENGTH_MULT);
    return clamp(pow(color,vec3(2.2)),0.0,1.0);
}

vec4 flame(float t){ return max(t,0.)*vec4(9,3,1,1); }
bool keypress(int key){ return texelFetch(iChannel3, ivec2(key,0),0).x > 0.; }

vec4 drawViz(float r, float a){
    if(r > 1.0) return vec4(0);
    vec2 ar = vec2(a + float(a < 0.), r*512./iResolution.y);
    vec4 o = vec4(0);
    float wave = texture(iChannel0, vec2(ar.x, r)).z;
    if(!keypress(0x31)) o += flame(wave).bgra;
    vec4 fft = texture(iChannel0, ar);
    if(!keypress(0x32)) o += flame(fft.x*fft.x) * max(1.-o.a, 0.);
    return o;
}

vec3 diskViz(vec3 p){
    vec3 diskPos=vec3(0,0,3);
     float scale = ETCHING_SCALE;                              // match getDist scale
    vec2 q = ((p - diskPos) / scale * gRot).xy;
    float r = length(q);
    float a = atan(q.x, q.y) / PI;
  r = clamp((r - 0.35) / (0.65), 0.0, 1.0);
    return drawViz(r, a).rgb;
}


vec3 waveScene(vec3 dir, float steps){
    vec4 O = vec4(0);
    float d = 1e-3, z = 1e-3, r;
    for(float i=0.; i++ < steps;
        // color from DIRECTION, fully continuous (no z, no atan wrap)
         O += (cos(z*.45 + iTime + vec4(0,2,4,3))*1.5 + 1.0)/d/z)
    {
        vec3 p = z * dir;
        r = max(-++p, 0.).y;
        float hx = (p.x+6.5)/15.;
        hx = abs(fract(hx*0.5)*2.0 - 1.0);              // mirrored audio
        float snd = texelFetch(iChannel0, ivec2(0, int(hx*511.)), 0).x;
        p.y += r + r - 2.*snd;
        z += d = .1*(.1*r + abs(p.y)/(1.5+r*r) + max(d = p.z+3., -d*.05));
        d = max(d, 1e-3);
    }
    return tanh(O/900.).rgb;
}

void mainCubemap(out vec4 O, in vec2 U, in vec3 ro, in vec3 rd){
    vec3 d = abs(rd);                    // 8-fold symmetric fold — every face mirrors
    O = vec4(waveScene(normalize(d), 60.), 1.0);
}


vec3 bgWave(vec2 I){
    vec3 R = iResolution.xyy;
    return waveScene(normalize(vec3(I+I,0) - R), BG_STEPS);
}

void getMat(in vec3 p, in float id, inout vec3 col, inout float sp,
            inout float refl, inout float refr, inout vec3 difr){
    int i=int(id);
    if(i==0){ col=vec3(0.1); sp=12.0; refl=0.5; refr=1.0; difr=vec3(0); }
    else if(i==1){
        col=vec3(0.012,0.012,0.02); sp=9.0; refl=1.0; refr=0.0;
        vec3 diskPos=vec3(0,0,3);
        vec3 dp=(p-diskPos)*gRot+diskPos;
        difr = normalize(vec3(-normalize(dp).y, 0.0, normalize(dp).x));
    }
    else if(i==2){ col=vec3(0.05); sp=9.0; refl=1.0; refr=0.0; difr=vec3(0); }
}



vec3 bgWaveRaw(vec2 I){
    vec3 R = iResolution.xyy;
    return waveScene(normalize(vec3(I+I,0) - R), BG_STEPS);
}


vec3 waveSceneOsc(vec3 dir, float steps){
    vec4 O = vec4(0);
    float d = 1e-3, z = 1e-3, r;
    for(float i=0.; i++ < steps; O += (cos(z*.5+iTime+vec4(0,2,4,3))*1.5+1.0)/d/z){
        vec3 p = z * dir;
        r = max(-++p, 0.).y;
        float bin = clamp((p.x+6.5)/15., 0., 1.) * 511.;
        float wav = texelFetch(iChannel0, ivec2(0, int(bin)), 0).y;   // .y = WAVEFORM
        p.y += r + r - 2.*(wav - 0.5);                                 // centered
        z += d = .1*(.1*r + abs(p.y)/(1.5+r*r) + max(d=p.z+3., -d*.05));
        d = max(d, 1e-3);
    }
    return tanh(O/900.).rgb;
}


vec3 sampleSphereRoom(vec3 ro, vec3 rd){
    float R = 8.0;
    float b = dot(ro, rd);
    float c = dot(ro, ro) - R*R;
    float t = -b + sqrt(b*b - c);
    vec3 dir = normalize(ro + rd * t);

    // hemisphere coverage: map the full direction range onto the field
    float lon = atan(dir.z, dir.x) / (2.0*PI) + 0.5;    // 0..1 around

    // stretch latitude so the field fills the ENTIRE hemisphere (pole to pole)

  // was: float lat = fract(acos(clamp(dir.y,-1.,1.))/PI * 3.0);
    float latRaw = acos(clamp(dir.y,-1.,1.))/PI * 3.0;
    float lat = abs(fract(latRaw * 0.5) * 2.0 - 1.0);   // triangle wave = mirrored tiling, no tear

   // lat = fract(lat * 3.0);              // repeat the field 3x vertically -> fills hemisphere
    vec2 I = vec2(lon, lat) * iResolution.xy;
    return bgWaveRaw(I);


}



// Intersect a ray from origin `ro` dir `rd` with an axis-aligned box of half-size `b`.
vec3 sampleCubeRoom(vec3 ro, vec3 rd){
  vec3 b = vec3(4.0);                    // room half-size (walls at ±8)
    vec3 t = (sign(rd) * b - ro) / rd;      // distance to each far wall
    float tHit = min(min(t.x, t.y), t.z);   // nearest wall the ray exits through
    vec3 hit = ro + rd * tHit;              // point on the wall

    vec2 uv;
    if(t.x <= t.y && t.x <= t.z)      uv = hit.zy;   // X face
    else if(t.y <= t.z)               uv = hit.xz;   // Y face
    else                              uv = hit.xy;   // Z face

    vec2 I = (uv / b.x * 0.5 + 0.5) * iResolution.xy;
    return bgWaveRaw(I);
}

#ifdef AA
void mainImageRaw(out vec4 fragColor, in vec2 fragCoord)
#else
void mainImage(out vec4 fragColor, in vec2 fragCoord)
#endif
{
    gRot = cdRot();
    vec2 uv=(fragCoord-.5*iResolution.xy)/iResolution.y;
    vec3 ro=vec3(0.0);
    vec3 rd=normalize(vec3(uv.x,uv.y,1.0));
    vec2 rs=rayMarch(ro,rd);



    if(rs.x >= MAX_DIST){
        fragColor = vec4(bgWave(fragCoord), 1.0);
        return;
    }

    vec3 p=ro+rd*rs.x;
    vec3 n=getNormal(p);
    vec3 lp=vec3(2,1,-6);

    vec3 col=vec3(0); float sp=0.0, refl=0.0, refr=0.0; vec3 difr=vec3(0);
    getMat(p,rs.y,col,sp,refl,refr,difr);

    vec3 diff=clamp(getLight(p,n,vec3(1),lp),0.0,1.0);
    vec3 spec=clamp(getSpecular(p,n,vec3(1),lp,ro,sp),0.0,1.0);
    float fresnel=pow(1.0-dot(n,-rd),2.0);

vec3 reflTex=vec3(0), refrTex=vec3(0);
if(refl > 0.0){
        vec3 rdir = reflect(rd, n);
        vec3 env = sampleSphereRoom(p, rdir);      // <-- sphere room, NOT cubemap!
        env = env / (1.0 + env);
        reflTex = refl * env * 2.;
    }
    if(refr > 0.0){
        vec3 tdir = refract(rd, n, 0.99);
        tdir.z = -tdir.z;
        tdir = normalize(tdir);
        refrTex = refr * waveScene(tdir, REFR_STEPS);
    }

vec3 diffract=getDiffraction(vec3(dot(n,normalize(-p))),rd,rs.x,difr);

  vec3 color = col * 0.2;
    color += reflTex;
    color = clamp(color, 0.0, 1.0);

    if(rs.y > 0.5 && rs.y < 1.5)
        color = clamp(color + diskViz(p)*VIZ_GAIN, 0.0, 1.0);

    color += diffract * 0.8;         // <-- rainbow bows on top, prominent
    color = clamp(color, 0.0, 1.0);
    color = pow(color, vec3(0.4545));
    fragColor = vec4(color, 1.0);
}

#ifdef AA
void mainImage(out vec4 O, vec2 U){
    vec4 T; O=vec4(0);
    for(int k=0;k<AA*AA;k++,O+=T)
        mainImageRaw(T, U + 0.33*vec2(k%AA - AA/2, k/AA - AA/2));
    O /= float(AA*AA);
}
#endif
"""

def build_viz_cd_buffera(note_min, note_max, reverb=True):
    head = "// --- Buffer A: Soundform CD state (set THIS pass's iChannel0 = Buffer A) ---\n"
    if not reverb:
        head += "#define USE_REVERB 0\n"   # keep the etched waveform matching the audio
    return head + _viz_defs(note_min, note_max) + CD_BUFFERA_CODE

# --inst piano: selects the additive piano voice + piano ADSR (no sustain stage —
# the voice carries the decay; short damper release). Prepended to the Sound tab,
# the viz Buffer A (waveform/energy must match), and single-file builds — the
# templates' #ifndef guards make these win.
PIANO_DEFS  = ("#define VOICE 1\n"
               "#define MASTER_GAIN 0.62\n")
EPIANO_DEFS = ("#define VOICE 2\n"
               "#define MASTER_GAIN 0.70\n")
ORGAN_DEFS  = ("#define VOICE 3\n"
               "#define MASTER_GAIN 0.55\n")
PIANO2_DEFS = ("#define VOICE 4\n"
               "#define MASTER_GAIN 0.60\n")
EPIANO2_DEFS = ("#define VOICE 5\n"
                "#define MASTER_GAIN 0.85\n")
PIANO3_DEFS = ("#define VOICE 6\n"
               "#define MASTER_GAIN 0.95\n")
PIANO4_DEFS = ("#define VOICE 7\n"
               "#define MASTER_GAIN 0.60\n")
PIANO5_DEFS = ("#define VOICE 8\n"
               "#define MASTER_GAIN 0.55\n")
PIANO6_DEFS = ("#define VOICE 9\n"
               "#define MASTER_GAIN 0.55\n"
               # v7: loops + pedal carry the REAL sustain now — the old long
               # reverb-as-decay (MIX .55 / DUR 2.2 / FALL 1.6 / TAPS 48) was
               # fighting it, smearing the pedal texture. Back to a room.
               "#define REVERB_MIX 0.32\n"
               "#define REVERB_DUR 1.0\n"
               "#define REVERB_FALL 3.0\n"
               "#define REVERB_TAPS 16\n"
               "#define PAD_SEC 1.5\n"        # reverb tail scan pad (was the decay window)
               "#define P6_SYNTH 0.5\n")      # FM stays sample->room glue
PIANO7_DEFS = ("#define VOICE 10\n"
               "#define MASTER_GAIN 0.55\n")
INST_DEFS   = {"piano": PIANO_DEFS, "epiano": EPIANO_DEFS, "organ": ORGAN_DEFS,
               "piano2": PIANO2_DEFS, "epiano2": EPIANO2_DEFS, "piano3": PIANO3_DEFS,
               "piano4": PIANO4_DEFS, "piano5": PIANO5_DEFS, "piano6": PIANO6_DEFS,
               "piano7": PIANO7_DEFS}

# ============================================================ VIZ "piano" (--viz 3)
# Raymarched grand piano (user-supplied ShaderToy, integrated 2026-07-12) whose
# KEYS PLAY THE MIDI. Buffer A bakes all 88 key states once per frame (one texel
# per key, x = midi-21) so the Image pass SDF only pays a texelFetch per eval.
# Presses use the RAW key gate (midiGT) — pedal-extended durations would leave
# whole runs visually sunk; the pedal sustains the SOUND, not the key.
PIANO_BUFA_CODE = r"""// mid2glsl --viz 3 (piano): Buffer A — 88 key states from the MIDI data.
// One texel per key (x = midi-21, row 0). Image reads via iChannel3.

uint  onTick(uint i){ return midiOM(i)&0xFFFFu; }
uint  meta16(uint i){ return midiOM(i)>>16; }
uint  durTick(uint i){ uint w=midiDur(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
uint  gateTick(uint i){ uint w=midiGT(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
int   metaNote(uint m){ return MIDI_NOTE_MIN+int(m&0x7Fu); }
float metaVel(uint m){ return float((m>>7)&0x7Fu)/127.0; }

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    int key = int(fragCoord.x);
    if(fragCoord.y >= 1.0 || key >= 89){
        // ---- rows >= 4: the logo's glyph distance field, RELAYED from
        // Buffer B (which owns the font bake; iChannel1 = Buffer B). Keeps
        // Image's channel wiring unchanged while the font core lives in B.
        if (fragCoord.y >= 4.0) fragColor = texelFetch(iChannel1, ivec2(fragCoord.xy), 0);
        else fragColor = vec4(0.0);
        return;
    }
    int wantNote = 21 + key;
    float qSec = float(MIDI_TIME_Q_SAMPLES)/MIDI_SAMPLE_RATE;
    float now = clamp(iTime/qSec, 0.0, float(MIDI_END_TICK));
    uint b = uint(now)>>MIDI_BLOCK_SHIFT_TICKS;
    if(key == 88){                       // texel 88 = damper pedal (CC64 state):
        float ped = 0.0;                 // any note sounding PAST its key gate
        for(uint j=0u; j<(iFrame<0 ? 99999u : MIDI_LOOKBACK_BLOCKS); j++){
            if(j>b) break;
            uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
            for(uint k=0u; k<(iFrame<0 ? 99999u : count); k++){
                uint i=start+k;
                float tt=(now-float(onTick(i)))*qSec;
                if(tt<0.0) continue;
                float gate=float(gateTick(i))*qSec, ring=float(durTick(i))*qSec;
                ped=max(ped, smoothstep(gate, gate+0.12, tt)
                            *(1.0-smoothstep(ring-0.06, ring+0.03, tt)));
            }
        }
        fragColor=vec4(ped,0.0,0.0,1.0); return;
    }
    float press = 0.0;
    for(uint j=0u; j<(iFrame<0 ? 99999u : MIDI_LOOKBACK_BLOCKS); j++){  // de-unroll: keeps the scan a runtime loop under ANGLE
        if(j>b) break;
        uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
        for(uint k=0u; k<(iFrame<0 ? 99999u : count); k++){
            uint i=start+k;
            uint m=meta16(i);
            if(metaNote(m)!=wantNote) continue;
            float tt=(now-float(onTick(i)))*qSec;          // s since onset
            if(tt<0.0) continue;
            float gate=float(gateTick(i))*qSec;            // RAW key-down time (pre-pedal)
            // key kinematics: ~18ms dip, hold while the key is down, ~90ms return
            float p=smoothstep(0.0,0.018,tt)*(1.0-smoothstep(gate,gate+0.09,tt));
            press=max(press, p*(0.75+0.25*metaVel(m)));
        }
    }
    fragColor=vec4(press,0.0,0.0,1.0);
}"""

PIANO_IMAGE_CODE = r"""#define MARCH_STEPS 50      // 128 = finest key silhouettes (~25% slower)
#define REFLECT_STEPS 24
#define EPSILON 0.05
#ifndef SONG_END
#define SONG_END 180.0     // lid closes here (the 180s playback limit); --end N overrides
#endif
#define AA_EDGE 3           // extra rays on EDGE pixels only (0 = off, 3 = 4x on edges)
#define AA_EDGE_T 0.10      // luma-contrast (fwidth) threshold that marks an edge
#define KEY_REFLECT 0.5     // key lacquer vs the lid's: 1.0 = identical mirror, 0.0 = matte

// ---- 3D maker logo: extruded Bezier glyphs standing on the piano ----------
// World-anchored (orbits with the scene). Shows for LOGO_FADE_T seconds, then
// alpha-fades over LOGO_FADE_D via a continuation ray, then costs nothing.
#define LOGO_FADE_T 30.0
#define LOGO_FADE_D 3.0
#define LOGO_SCALE  2.6
#define LOGO_CX     -12.0
#define LOGO_BASE_Y 96.0
#define LOGO_Z      -20.0
#define LOGO_DEPTH  1.6
#define LOGO_ADVW   54.276
bool gSkipLogo = false;
float logoAlpha(){ return 1.0 - smoothstep(LOGO_FADE_T, LOGO_FADE_T + LOGO_FADE_D, iTime); }
#define PI 3.141592653

// ---- mid2glsl --viz 3: keys + pedal play the MIDI --------------------------
// Bind: Image iChannel3 = Buffer A (88 key states + damper on texel 88).
#define KEYSIDE_TINT vec3(1.35, 0.80, 0.45)   // saturated wood tint for the key sides
#define KEYSIDE_GAIN 0.18   // includes ~2.7x LightIntensity compensation near the keys
#define KEY_DIP_W 2.4
#define KEY_DIP_B 1.9
float keyState(int midi){
    if(midi < 21 || midi > 108) return 0.0;
    return texelFetch(iChannel3, ivec2(midi - 21, 0), 0).r;
}
float pedalState(){ return texelFetch(iChannel3, ivec2(88, 0), 0).r; }  // damper (CC64)
// white index from the BASS side (0 = A0) -> midi
int white2midi(int w){
    const int d[7] = int[7](0,2,3,5,7,8,10);
    return 21 + (w/7)*12 + d[w%7];
}
// ---- maker nameboard decal (front vertical panel, normal -Z) ----
const float DECAL_Z     = -40.0;              // front wall plane (world z). local z=0
const float DECAL_ZTOL  =  1.5;               // plane hit tolerance
const float DECAL_CX    = -12.0;              // text center X (world) = keyboard center
const float DECAL_CY    =  87.5;             // text center Y (world), in the black riser
const float DECAL_TH    =  6.5;               // world height that maps to LINE_H(=12 font units)
const vec3  DECAL_COLOR = vec3(0.90, 0.86, 0.74);  // whitish-grey ink
const float DECAL_ADVW  = 54.276;             // total advance of "Boesendorfer"

// boe  --  Bezier SDF font  (NEVESD-compatible, no texture)
//  Font: boe (boe.woff)  --  chars: "Boesendorfer"
// iquilezles.org/articles/distance -- exact quadratic bezier SDF
// Fast compile: const-array loops + switch() tables
// Scale vec2 sc in mainImage -- animate freely with sin(iTime)

#define _    32,
#define _EX  33,
#define _DBQ 34,
#define _NUM 35,
#define _DOL 36,
#define _PER 37,
#define _AMP 38,
#define _QT  39,
#define _LPR 40,
#define _RPR 41,
#define _MUL 42,
#define _ADD 43,
#define _COM 44,
#define _SUB 45,
#define _DOT 46,
#define _DIV 47,
#define _COL 58,
#define _SEM 59,
#define _LES 60,
#define _EQ  61,
#define _GE  62,
#define _QUE 63,
#define _AT  64,
#define _LBR 91,
#define _ANTI 92,
#define _RBR 93,
#define _HAT 94,
#define _UN  95,
#define _GRV 96,
#define _0 48,
#define _1 49,
#define _2 50,
#define _3 51,
#define _4 52,
#define _5 53,
#define _6 54,
#define _7 55,
#define _8 56,
#define _9 57,
#define _A 65,
#define _B 66,
#define _C 67,
#define _D 68,
#define _E 69,
#define _F 70,
#define _G 71,
#define _H 72,
#define _I 73,
#define _J 74,
#define _K 75,
#define _L 76,
#define _M 77,
#define _N 78,
#define _O 79,
#define _P 80,
#define _Q 81,
#define _R 82,
#define _S 83,
#define _T 84,
#define _U 85,
#define _V 86,
#define _W 87,
#define _X 88,
#define _Y 89,
#define _Z 90,
#define _a  97,
#define _b  98,
#define _c  99,
#define _d 100,
#define _e 101,
#define _f 102,
#define _g 103,
#define _h 104,
#define _i 105,
#define _j 106,
#define _k 107,
#define _l 108,
#define _m 109,
#define _n 110,
#define _o 111,
#define _p 112,
#define _q 113,
#define _r 114,
#define _s 115,
#define _t 116,
#define _u 117,
#define _v 118,
#define _w 119,
#define _x 120,
#define _y 121,
#define _z 122,

#define makeStr(name) \
float name(vec2 _FSTU) { \
    if(_FSTU.x<0.||_FSTU.y<0.||_FSTU.y>LINE_H) return 0.0; \
    const int _FSTC[] = int[](

#define _end 0); \
    float _FSTX=0.0; \
    for(int _FSTK=0;_FSTK<(iFrame<0?9999:_FSTC.length()-1);_FSTK++) { \
        vec4 _FSTM=getGlyphMeta(_FSTC[_FSTK]); \
        if(_FSTU.x<_FSTX+_FSTM.z) { \
            vec4 _FSTR=getGlyphRect(_FSTC[_FSTK]); \
            vec2 _FSTB=vec2(_FSTM.x,LINE_H-_FSTM.y-_FSTM.w); \
            vec2 _FSTL=(_FSTU-vec2(_FSTX,0.)-_FSTB)/vec2(_FSTR.z*ATLAS_W,_FSTM.w); \
            return (all(greaterThanEqual(_FSTL,vec2(0.)))&&all(lessThanEqual(_FSTL,vec2(1.)))) \
                ? _atlasAlpha(_FSTR.xy+_FSTL*_FSTR.zw) : 0.0; \
        } \
        _FSTX+=_FSTM.z; \
    } \
    return 0.0; \
}

#define makeColorStr2D(name) \
vec4 name(vec2 _C2DU) { \
    if(_C2DU.x<0.||_C2DU.y<0.||_C2DU.y>LINE_H) return vec4(0.); \
    const int _C2DC[] = int[](

#define _endC2D 0); \
    float _C2DX=0.0; \
    for(int _C2DK=0;_C2DK<(iFrame<0?9999:_C2DC.length()-1);_C2DK++) { \
        vec4 _C2DM=getGlyphMeta(_C2DC[_C2DK]); \
        if(_C2DU.x<_C2DX+_C2DM.z) { \
            vec4 _C2DR=getGlyphRect(_C2DC[_C2DK]); \
            vec2 _C2DB=vec2(_C2DM.x,LINE_H-_C2DM.y-_C2DM.w); \
            vec2 _C2DL=(_C2DU-vec2(_C2DX,0.)-_C2DB)/vec2(_C2DR.z*ATLAS_W,_C2DM.w); \
            float _C2DA=(all(greaterThanEqual(_C2DL,vec2(0.)))&&all(lessThanEqual(_C2DL,vec2(1.)))) \
                ? _atlasAlpha(_C2DR.xy+_C2DL*_C2DR.zw) : 0.; \
            return vec4(1.,1.,1.,_C2DA); \
        } \
        _C2DX+=_C2DM.z; \
    } \
    return vec4(0.); \
}


//__FONTCORE__
float _atlasAlpha(vec2 pos) {
    float d=cSDF(pos), fw=max(fwidth(d),0.001);
    return smoothstep(fw,-fw,d);
}

vec3 tintGradient(vec3 f,vec3 d,vec3 l){return mix(d,l,dot(f,vec3(.299,.587,.114)));}
vec3 rgb2hsl(vec3 c){float mx=max(c.r,max(c.g,c.b)),mn=min(c.r,min(c.g,c.b)),d=mx-mn;
  float h=0.,s=0.,l=(mx+mn)*.5;if(d>.001){s=d/(1.-abs(2.*l-1.));
  if(mx==c.r)h=mod((c.g-c.b)/d,6.);else if(mx==c.g)h=(c.b-c.r)/d+2.;
  else h=(c.r-c.g)/d+4.;h/=6.;}return vec3(h,s,l);}
vec3 hsl2rgb(vec3 h){float c=(1.-abs(2.*h.z-1.))*h.y,
  x=c*(1.-abs(mod(h.x*6.,2.)-1.)),m=h.z-c*.5;float hi=floor(h.x*6.);
  vec3 r=hi<1.?vec3(c,x,0):hi<2.?vec3(x,c,0):hi<3.?vec3(0,c,x):
         hi<4.?vec3(0,x,c):hi<5.?vec3(x,0,c):vec3(c,0,x);return r+m;}
vec3 tintHueShift(vec3 f,float s){vec3 h=rgb2hsl(f);h.x=fract(h.x+s);return hsl2rgb(h);}

vec2 drawChar(vec2 frag,vec2 cursor,int c,vec2 sc,inout vec3 col,inout float alpha){
    vec4 uv=getGlyphRect(c); vec4 mt=getGlyphMeta(c);
    vec2 bl=cursor+vec2(mt.x,FONT_BASE-mt.y-mt.w)*sc;
    vec2 sz=vec2(uv.z*ATLAS_W,mt.w)*sc;
    vec2 lc=(frag-bl)/sz;
    if(all(greaterThanEqual(lc,vec2(0.)))&&all(lessThanEqual(lc,vec2(1.)))){
        float a=_atlasAlpha(uv.xy+lc*uv.zw);
        col=mix(col,vec3(1.),a); alpha=max(alpha,a);
    }
    return vec2(cursor.x+mt.z*sc.x,cursor.y);
}
vec2 fontUV(vec2 lab,float cx,float cy,float ch,float adv,vec2 sc){
    float p=LINE_H/ch;
    return vec2((lab.x-cx)*p/sc.x+adv*.5,(lab.y-cy)*p/sc.y+LINE_H*.5);
}

// -- Edit text here -------------------------------------------------------
makeStr(line1)       _B _o _e _s _e _n _d _o _r _f _e _r  _end
makeColorStr2D(col1) _B _o _e _s _e _n _d _o _r _f _e _r  _endC2D


/*
"Magic particles" by Emmanuel Keller aka Tambako - December 2015
License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
Contact: tamby@tambako.ch
   (mid2glsl: particle mechanics kept — launch staggering, lifetimes/rebirth,
   sparkle, hue drift, the eight-branch star glare; the 2D harmonic path is
   replaced by a 3D flight around/through the logo letters, projected through
   the scene camera and DEPTH-TESTED against the raymarch, so the swarm truly
   weaves in and out of the letters. Multipass motion blur -> swarm trailing.)
*/
#define twopi 6.28319
#define PARTICLES_T0 10.0            // fade-in start (s); fades out with the logo
// 1: mac / 2: normal gpu / 3: good gpu / 4: gaming gpu — be careful going up!
#define tb_complexity 1
#if tb_complexity == 1
const int nb_particles = 95;
const float tb_lt_min = 0.9, tb_lt_max = 1.9;
#elif tb_complexity == 2
const int nb_particles = 160;
const float tb_lt_min = 1.0, tb_lt_max = 2.5;
#elif tb_complexity == 3
const int nb_particles = 280;
const float tb_lt_min = 1.1, tb_lt_max = 3.2;
#else
const int nb_particles = 500;
const float tb_lt_min = 1.2, tb_lt_max = 4.0;
#endif
const float tb_timefact_min = 6., tb_timefact_max = 20.;
const float tb_time_factor = 0.75;
const float tb_start_time = 2.5;
const float tb_grow_time_factor = 0.15;
const float tb_int_div = 40000.;
const float tb_int_factor_min = 0.1, tb_int_factor_max = 3.2;
const float tb_spark_min_int = 0.25, tb_spark_max_int = 0.88;
const float tb_spark_min_freq = 2.5, tb_spark_max_freq = 6.0;
const float tb_spark_tff = 0.35;
const float tb_mp_int = 12.;
const float tb_dist_factor = 3.;
const float tb_ppow = 2.3;
const float tb_min_hue = -0.13, tb_max_hue = 0.13;
const float tb_min_sat = 0.5, tb_max_sat = 0.9;
const float tb_hue_time_factor = 0.035;
const float tb_mp_hue = 0.5, tb_mp_sat = 0.18;
const vec2 tb_starhv_dfac = vec2(9., 0.32);
const float tb_starhv_ifac = 0.25;
const vec2 tb_stardiag_dfac = vec2(13., 0.61);
const float tb_stardiag_ifac = 0.19;

float gDepth = 1e6;                  // primary-ray hit depth (set by Render)

vec3 tb_hsv2rgb (vec3 hsv) {
	hsv.yz = clamp (hsv.yz, 0.0, 1.0);
	return hsv.z*(0.63*hsv.y*(cos(twopi*(hsv.x + vec3(0.0, 2.0/3.0, 1.0/3.0))) - 1.0) + 1.0);
}
float tb_random(float co){ return fract(sin(co*12.989) * 43758.545); }

// 3D flight of the MAIN particle: orbit the letters, snake through them, orbit
// again; tp is the (stretched, staggered) particle time since PARTICLES_T0
vec3 tb_path (float tp)
{
	vec3 C = vec3(LOGO_CX, LOGO_BASE_Y + 0.5*LINE_H*LOGO_SCALE, LOGO_Z);
	float hx = 0.5*LOGO_ADVW*LOGO_SCALE + 10.0;
	float th = tp*0.85;
	vec3 orbit = C + vec3(hx*cos(th), 7.0*sin(th*0.63), (LOGO_DEPTH + 11.0)*sin(th));
	// snake: sweep the text left-right while z dives through the letter plane
	vec3 snake = C + vec3(hx*0.92*sin(tp*0.42),
	                      9.0*sin(tp*1.6),
	                      (LOGO_DEPTH + 5.5)*sin(tp*2.4));
	float w = smoothstep(5.5, 7.5, tp) - smoothstep(12.5, 14.5, tp);   // orbit->snake->orbit
	return mix(orbit, snake, clamp(w, 0.0, 1.0));
}

vec3 tb_color (int partnr, float pint, float t2, float runnr)
{
	float saturation = mix(tb_min_sat, tb_max_sat, tb_random(float(partnr*6 + 44) + runnr*3.3))*0.45/pint;
	float hue = mix(tb_min_hue, tb_max_hue, tb_random(float(partnr + 124) + runnr*1.5)) + tb_hue_time_factor*t2;
	return tb_hsv2rgb(vec3(hue, saturation, pint));
}

// star glare of one particle at screen-space (tuv) distance vectors
float tb_star (vec2 uvppos, float dist)
{
	float distv = length(uvppos*tb_starhv_dfac);
	float disth = length(uvppos*tb_starhv_dfac.yx);
	vec2 uvpposd = 0.7071*vec2(dot(uvppos, vec2(1., 1.)), dot(uvppos, vec2(1., -1.)));
	float distd1 = length(uvpposd*tb_stardiag_dfac);
	float distd2 = length(uvpposd*tb_stardiag_dfac.yx);
	return 1./(dist*tb_dist_factor + 0.015)
	     + tb_starhv_ifac/(disth*tb_dist_factor + 0.01)
	     + tb_starhv_ifac/(distv*tb_dist_factor + 0.01)
	     + tb_stardiag_ifac/(distd1*tb_dist_factor + 0.01)
	     + tb_stardiag_ifac/(distd2*tb_dist_factor + 0.01);
}

vec3 drawParticles3D (vec2 tuv, vec3 cam_pos, vec3 rgt, vec3 up, vec3 fwd)
{
	float t2 = tb_time_factor*(iTime - PARTICLES_T0);
	vec3 pcol = vec3(0.);
	for (int i = 1; i < (iFrame<0 ? 99999 : nb_particles); i++)
	{
		float pst = tb_start_time*tb_random(float(i*2));
		float plt = mix(tb_lt_min, tb_lt_max, tb_random(float(i*2-35)));
		float t4 = mod(t2 - pst, plt);
		float t3 = t4 + pst;
		float runnr = floor((t2 - pst)/plt);
		float tf = mix(tb_timefact_min, tb_timefact_max, tb_random(float(i*2 + 94) + runnr*1.5));
		float ptime = (runnr*plt + pst)*(-1./tf + 1.) + t2/tf;
		vec3 P = tb_path(ptime);
		// 3D drift away from the trajectory + slight gravity droop
		P += 7.0*(vec3(tb_random(float(i*3-23) + runnr*4.),
		               tb_random(float(i*7+632) - runnr*2.5),
		               tb_random(float(i*5+117) + runnr*7.1)) - 0.5)*(t3 - pst);
		P.y -= 1.6*t4*t4;
		// project through the scene camera
		vec3 v = P - cam_pos;
		float z = dot(v, fwd);
		if (z < 5.0) continue;
		vec2 sp = 0.5*vec2(dot(v, rgt), dot(v, up))/z;
		vec2 uvppos = tuv - sp;
		if (dot(uvppos, uvppos) > 0.25) continue;          // glare reach
		float dist = length(uvppos);
		float pint1 = tb_star(uvppos, dist);
		float pint0 = mix(tb_int_factor_min, tb_int_factor_max, tb_random(runnr*4. + float(i-55)));
		float pint = pint0*(pow(pint1, tb_ppow)/tb_int_div)*(-t4/plt + 1.);
		pint *= smoothstep(0., tb_grow_time_factor*plt, t4);
		float sparkfreq = clamp(tb_spark_tff*t4, 0., 1.)*tb_spark_min_freq
		                + tb_random(float(i*5 + 72) - runnr*1.8)*(tb_spark_max_freq - tb_spark_min_freq);
		pint *= mix(tb_spark_min_int, tb_spark_max_int, tb_random(float(i*7 - 621) - runnr*12.))*sin(sparkfreq*twopi*t2)/2. + 1.;
		if (z > gDepth + 1.0) pint *= 0.08;                // occluded: behind the scene
		pcol += tb_color(i, pint, t2, runnr);
	}
	// the bright main particle at the head of the swarm
	{
		vec3 P = tb_path(t2);
		vec3 v = P - cam_pos;
		float z = dot(v, fwd);
		if (z >= 5.0) {
			vec2 sp = 0.5*vec2(dot(v, rgt), dot(v, up))/z;
			vec2 uvppos = tuv - sp;
			float dist = length(uvppos);
			float pint1 = tb_star(uvppos, dist);
			if (tb_int_factor_max*pint1 > 6.)
			{
				float pint = tb_int_factor_max*(pow(pint1, tb_ppow)/tb_int_div)*tb_mp_int;
				if (z > gDepth + 1.0) pint *= 0.08;
				float saturation = 0.75/pow(pint, 2.5) + tb_mp_sat;
				pcol += tb_hsv2rgb(vec3(tb_hue_time_factor*t2 + tb_mp_hue, saturation, pint));
			}
		}
	}
	// fade in at T0, fade out WITH the logo ("both fade out")
	return pcol * smoothstep(PARTICLES_T0, PARTICLES_T0 + 2.5, iTime) * logoAlpha();
}

vec3 place_pos = vec3(10.0,0.0,40.0);

vec3 get_light_pos()
{
    return vec3(100.0*sin(iTime*.5),200.0,300.0*cos(iTime*.5));
    //return vec3(100.0,200.0,300.0);
}

struct Obj
{
	float m_dist;
	int m_obj_idx;//0:floor 1:wall 2:piano body 3:w keys 4:b keys  5: sound board 6:gold mat 7:string
};

void Choose(in Obj obj1, in Obj obj2, out Obj obj)
{
	if (obj1.m_dist> obj2.m_dist)
	{
		obj = obj2;
	}
	else
	{
		obj = obj1;
	}
}

float LightIntensity(in vec3 world_pos)
{
	float temp = length(get_light_pos() - world_pos);
	return 1.0 / (pow(temp, 1.5)*0.0002);
}
//------------------------------------------------------
float Combine(float re1, float re2)
{
	if (re1<0.0 || re2 <0.0)
		return max(re1, re2);
	return sqrt(re1*re1 + re2*re2);
}

float Subtract(float re1, float sub)
{
	if (sub>0.0)
		return re1;
	return max(-sub, re1);
}

float MapBox(in vec3 pos, in vec3 half_size)
{//center at (0,0,0)
	vec3 v = abs(pos)-half_size;
	
	if (v.x<0.0 || v.y <0.0 || v.z<0.0)
		return max(max(v.x, v.y), v.z);
	return length(v);
}

float MapBoxSim(in vec3 pos, in vec3 half_size)//no inside info
{//center at (0,0,0)
	return length(max(abs(pos)-half_size,0.0))-0.1;
}

float MapRoundBox(in vec3 pos, in vec3 half_size, in float r)//no inside info
{
	return length(max(abs(pos)-half_size,0.0))-r;
}

float Map2Box(in vec3 pos, in vec2 top_half_size, in vec2 bottom_half_size, in float half_h)
{//center at (0,0,0)
	float y = abs(pos.y) - half_h;
	float p = pos.y*0.5/half_h + 0.5;
	p = clamp(p, 0.0,1.0);//bottom---top
	float x = abs(pos.x) - mix(bottom_half_size.x, top_half_size.x, p);
	float z = abs(pos.z) - mix(bottom_half_size.y, top_half_size.y, p);
	
	if (x<0.0 || y <0.0 || z<0.0)
		return max(max(x, y), z);
	return sqrt(x*x + y*y + z*z);
}

float MapCylinder(in vec3 pos, in float r, in float half_h)
{
	float y = abs(pos.y) - half_h;
	float rr = length(pos.xz) - r;
	return Combine(y, rr);
}
//-------------------------------------------------------------------------------------------
//unit 100 == 1 meter
void MapFloor(in vec3 world_pos, out Obj obj)
{
	float dist = world_pos.y;   // floor plane only — the ceiling belongs to MapWall (cubemap)
	if (dist <= EPSILON)
	{
		obj.m_obj_idx = 0;
	}
	
	obj.m_dist = dist;
}

void MapWall(in vec3 world_pos, out Obj obj)
{
	// inside-out CYLINDER (was a 200x300 box) + the CEILING cap at y=300.
	// Both share the wall material = cubemap sampled by direction, so overhead
	// you see the cubemap sky, not the stone floor stretched into a "tarp".
	// R must stay > ~185: the auto-orbit camera reaches ~178 from origin in xz.
	float dist = min(250.0 - length(world_pos.xz), 300.0 - world_pos.y);
	if (dist <= EPSILON)
	{
		obj.m_obj_idx = 1;
	}
	obj.m_dist = dist;
}


float MapPianoBodyShapeDist(in float x, in float y)
{
	if (y > 118.0)
	{//semi circle
		return sqrt((x+30.0)*(x+30.0)+(y-118.0)*(y-118.0)) - 45.0;
	}
	if (y>42.0)
	{
		//box
		float vx = -x - 75.0;
		//sin shape
		float sinv = sin(((y - 42.0)/76.0 + 0.5) * PI);
		sinv = x - (sinv*30.0 + 45.0);
		
		if (x<-30.0)
		{
			return vx;
		}
		return max(sinv, vx);
	}
	float xx = abs(x) - 75.0;
	return Combine(xx, 42.0-y);
}

float MapCover0(in vec3 pos)
{
	float re = MapPianoBodyShapeDist(-pos.x, pos.z);
	float re_2 = abs(pos.y) - 1.0;
	return Combine(re, re_2);
}
float MapBody(in vec3 pos)
{
	float re = MapPianoBodyShapeDist(-pos.x, pos.z);
	float re_2 = abs(pos.y) - 15.0;
	return Combine(re, re_2);
}

void MapPianoBody(in vec3 world_pos, out Obj obj)
{
	vec3 pos = world_pos + place_pos;
	
	//backfoot
	float re = Map2Box(pos - vec3(30.0, 40.0, 155.0), vec2(15.0,3.2), vec2(3.0,3.0), 35.0);
	
	//foot1
	float re_2 = Map2Box(pos - vec3(-67.0, 40.0, 12.0), vec2(5.0,12.5), vec2(3.0,3.0), 35.0);
	re = min(re, re_2);
	//foot2
	re_2 = Map2Box(pos - vec3(67.0, 40.0, 12.0), vec2(5.0,12.5), vec2(3.0,3.0), 35.0);
	re = min(re, re_2);
	
	//pedal box
	re_2 = MapBoxSim(pos - vec3(0.0, 10.0, 12.0), vec3(14.0,5.0,5.0));
	re = min(re, re_2);
	//pedal box2
	re_2 = Map2Box(pos - vec3(0.0, 42.5, 14.0), vec2(14.0,4.0), vec2(5.0,2.0), 35.0);
	re = min(re, re_2);
	
	//keyboard bottom
	re_2 = MapBoxSim(pos - vec3(0.0, 74.5, -9.0), vec3(75.0, 4.5, 9.0));
	re = min(re, re_2);
	
	//keyboard two side
	re_2 = MapBoxSim(pos - vec3(68.0, 77.0, -9.0), vec3(7.0, 7.0, 9.0));
	re = min(re, re_2);
	re_2 = MapBoxSim(pos - vec3(-70.0, 77.0, -9.0), vec3(5.0, 7.0, 9.0));
	re = min(re, re_2);
	
	{//open cover: rises as the piece starts, closes at SONG_END (--end N,
	 // default 180 = the playback time limit)
		vec3 pos_trans = pos + vec3(-75, -99.0, 0.0);
		float songLen = SONG_END;
		float open01 = smoothstep(0.0, 4.0, iTime)*(1.0 - smoothstep(songLen - 4.0, songLen, iTime));
		float angle = 0.4975*open01;
		float sinv = sin(angle);
		float cosv = cos(angle);
		pos_trans.xy = pos_trans.xy * mat2(cosv, -sinv, sinv, cosv);
		pos_trans.x += 75.0;
		re_2 = MapCover0(pos_trans);
		re = min(re, re_2);
		
		re_2 = MapBoxSim(pos_trans - vec3(0.0, 3.0, 63.0), vec3(75.0, 1.0, 21.0));
		re = min(re, re_2);
	}
	
	{
		vec3 pos_temp = pos - vec3(0.0,83.0,21.0);
		float x = abs(pos_temp.x) - 75.0;
		float y = abs(pos_temp.y) - 13.0;
		float z = abs(pos_temp.z) - 21.0;
		z = Combine(x,z);//use to cal color
		if (pos_temp.z>0.0 && pos_temp.z<22.0)
		{
			z = x;
		}
		z = abs(z + 4.0) - 4.0;
		re_2 = Combine(z, y);
		re = min(re, re_2);
		
		pos_temp = pos - vec3(0.0,83.0,0.0);
		y = MapPianoBodyShapeDist(-pos_temp.x, pos_temp.z);//use to cal color
		if (pos_temp.z >40.0 && pos_temp.z<44.0)
		{
			y = abs(pos_temp.x) - 75.0;
		}
		y = abs(y + 4.0) - 4.0;
		z = abs(pos_temp.y) - 13.0;
		re_2 = Combine(y, z);
		re = min(re, re_2);
	
		if (re <= EPSILON)
		{
			obj.m_obj_idx = 2;
		}
	}
	obj.m_dist = re;
}

void MapBodySoundboard(in vec3 world_pos, out Obj obj)
{
	vec3 pos = world_pos + place_pos -  vec3(0.0,80.0,-10.0);
	float re_3 = MapPianoBodyShapeDist(-pos.x, pos.z);
	float re_2 = abs(pos.y) - 6.0;
	float re = Combine(re_3, re_2) + 4.0;
	
	if (re< EPSILON)
	{
		obj.m_obj_idx = 5;
	}
	obj.m_dist = re;
}

void MapGoldObjs(in vec3 world_pos, out Obj obj)
{
	vec3 pos = world_pos + place_pos;
	
	//wheel
	float re = MapCylinder((pos-vec3(-67.0, 3.0, 12.0)).xzy, 2.5, 5.0);
	float re_2 = MapCylinder((pos-vec3(67.0, 3.0, 12.0)).zxy, 2.5, 5.0);
	re = min(re, re_2);
	re_2 = MapCylinder((pos-vec3(30.0, 3.0, 155.0)).xzy, 2.5, 5.0);
	re = min(re, re_2);
	
	//pedal — the -x one is the player's RIGHT (treble side) = the damper;
	//it presses with the sustain-pedal state derived from the MIDI
	re_2 = MapBoxSim(pos - vec3(0.0,7.0,3.0), vec3(2.0, 0.5, 5.0));
	re = min(re, re_2);
	re_2 = MapBoxSim(pos - vec3(9.0,7.0,3.0), vec3(2.0, 0.5, 5.0));
	re = min(re, re_2);
	{
		vec3 pp = pos - vec3(-9.0, 7.0, 3.0);
		float lever = clamp((5.0 - pp.z)/10.0, 0.0, 1.0);   // toe dips, heel pivots
		pp.y += pedalState()*1.4*lever;
		re_2 = MapBoxSim(pp, vec3(2.0, 0.5, 5.0));
	}
	re = min(re, re_2);
	
	//inner box
	re_2 = MapBoxSim(pos - vec3(0.0, 80.0, 18.0), vec3(70.0, 10.0, 14.0));
	re = min(re, re_2);
	
	//inner
	{
		vec3 pos_1 = pos -  vec3(0.0,85.0,0.0);
		float re_3 = MapPianoBodyShapeDist(-pos_1.x, pos_1.z);
		float re_2 = abs(pos_1.y) - 14.0;
		re_2 = Combine(re_3, re_2) + 10.0;
		
		float sinv = sin(-0.66);
		float cosv = cos(-0.66);
		pos_1.x -= 40.0;
		pos_1.z -=30.0;
		pos_1.xz = pos_1.xz * mat2(cosv, -sinv, sinv, cosv);
		re_3 = MapBox(pos_1, vec3(110.0,25.0,60.0));
		re_2 = Subtract(re_2, re_3);
		re_2 += 3.0;
		
		//line
		re_3 = MapBoxSim(pos - vec3(-59.0, 85.0, 27.0), vec3(1.0, 10.0, 23.0));
		re_2 = min(re_2, re_3);
		re_3 = MapBoxSim(pos - vec3(-30.0, 85.0, 40.0), vec3(1.0, 10.0, 36.0));
		re_2 = min(re_2, re_3);
		re_3 = MapBoxSim(pos - vec3(-4.0, 85.0, 50.0), vec3(1.0, 10.0, 46.0));
		re_2 = min(re_2, re_3);
		re_3 = MapBoxSim(pos - vec3(26.0, 85.0, 70.0), vec3(1.0, 10.0, 66.0));
		re_2 = min(re_2, re_3);
		re_3 = MapBoxSim(pos - vec3(62.0, 85.0, 75.0), vec3(1.0, 10.0, 71.0));
		re_2 = min(re_2, re_3);
		
		//hole
		re_3 = MapCylinder(pos - vec3(-42.0, 85.0, 60.0), 4.0, 10.0);
		re_2 = Subtract(re_2, re_3);
		re_3 = MapCylinder(pos - vec3(-17.0, 85.0, 82.0), 5.0, 10.0);
		re_2 = Subtract(re_2, re_3);
		re_3 = MapCylinder(pos - vec3(9.0, 85.0, 115.0), 6.0, 10.0);
		re_2 = Subtract(re_2, re_3);
		
		re = min(re, re_2);
	}
	
	if (re< EPSILON)
	{
		obj.m_obj_idx = 6;
	}
	obj.m_dist = re;
}

void MapString(in vec3 world_pos, out Obj obj)
{
	vec3 pos = world_pos + place_pos - vec3(0.0, 85.0,80.0);
	
	float re = MapBox(world_pos + place_pos - vec3(0.0, 85.0,80.0), vec3(56.5,0.001,54.0));
	
	float sinv = sin(-0.66);
	float cosv = cos(-0.66);
	pos.z -= 80.0;
	pos.xz = pos.xz * mat2(cosv, -sinv, sinv, cosv);
	float re_2 = MapBox(pos, vec3(130.0,25.0,60.0));
	re = Subtract(re, re_2);
	
	if (re< EPSILON)
	{
		re_2 = (world_pos.x + 56.5)/113.0*88.0;
		re_2 = abs(fract(re_2) - 0.5);
		if (re_2 < 0.499)
			re = max(re, re_2*2.5681818181818181818181818181818);
	}
	
	if (re< EPSILON)
	{
		obj.m_obj_idx = 7;
	}
	obj.m_dist = re;
}

void MapBlackKeys(in vec3 world_pos, out Obj obj)
{
	vec3 pos = world_pos + place_pos - vec3(-2.0, 82.75, -5.0);

	// ---- mid2glsl: press the black key under this x (MIDI-driven) ----
	// same group/segment math as the carve below; a segment's white-boundary
	// index turns into the lower white key, black = that white + 1 semitone
	{
		float xg = clamp(pos.x/126.0 + 0.5, 0.0, 1.0);
		float press = 0.0;
		if (xg > 0.03175 && xg < 0.984)
		{
			float x2 = clamp(xg - 0.02, 0.0, 1.0);
			int grp = int(x2*7.4285714285714285714285714285714);
			float fx = fract(x2*7.4285714285714285714285714285714);
			int bl = 0;
			if (fx > 0.1    && fx < 0.16)   bl = 1;      // #A
			else if (fx > 0.2476 && fx < 0.3343) bl = 2; // #G
			else if (fx > 0.395  && fx < 0.4817) bl = 3; // #F
			else if (fx > 0.6408 && fx < 0.7452) bl = 5; // #D
			else if (fx > 0.8159 && fx < 0.8861) bl = 6; // #C
			if (bl > 0)
			{
				int wlow = 51 - (grp*7 + bl + 1);        // lower white of the pair
				if (wlow >= 0 && wlow < 52)
					press = keyState(white2midi(wlow) + 1);
			}
		}
		float lever = clamp((5.0 - pos.z)/10.0, 0.0, 1.0);   // front dips, back pivots
		pos.y += press*KEY_DIP_B*lever;
	}
	float re = MapBoxSim(pos, vec3(63.0, 0.75, 5.0));
	
	// gate widened to EPSILON*2.5: the conservative distance scale below makes
	// the march accept hits while the RAW re is still ~EPSILON/0.42 — with the
	// old EPSILON gate the gap carve never ran and all 36 black keys fused
	// into one solid slab ("black keys gone")
	if (re <= EPSILON*2.5)
	{
		float x = clamp(pos.x/126.0 + 0.5, 0.0, 1.0);//0--1
		if (x<0.03175)//high pitch clamp
		{
			re = max(re, (0.03175-x)*126.0);
		}
		else if (x> 0.984)//low pitch clamp
		{
			re = max(re, (1.0-x)*126.0);
		}
		else
		{
			x = clamp(x-0.02, 0.0,1.0);
			x = fract(x*7.4285714285714285714285714285714);//group num
			if (x<0.1)//#A -- B
				re = max(re, (0.1-x)*7.4285714285714285714285714285714);
			else if (x>0.8861)//C---#C
				re = max(re, (x -0.8861)*7.4285714285714285714285714285714);
			else if (x>0.16 && x < 0.2476)//#G -- #A
				re = max(re, (0.2038-abs(x-0.2038))*7.4285714285714285714285714285714);
			else if (x>0.3343 && x< 0.395)//#F --- #G
				re = max(re, (0.36465-abs(x-0.36465))*7.4285714285714285714285714285714);
			else if (x>0.4817 && x< 0.6408)//#D -- #F
				re = max(re, (0.56125-abs(x-0.56125))*7.4285714285714285714285714285714);
			else if (x>0.7452 && x< 0.8159)//#C-- #D
				re = max(re, (0.78055-abs(x-0.78055))*7.4285714285714285714285714285714);
		}
	}
	
	re = max(re*0.42, re - KEY_DIP_B);   // conservative distance — see MapWhiteKeys (narrower keys → stronger factor)
	if (re <= EPSILON)
	{
		obj.m_obj_idx = 4;
	}
	obj.m_dist = re;
}

void MapWhiteKeys(in vec3 world_pos, out Obj obj)
{
	vec3 pos = world_pos + place_pos - vec3(-2.0, 80.5, -7.5);

	// ---- mid2glsl: press the white key under this x (MIDI-driven) ----
	{
		float xn = clamp(pos.x/126.0 + 0.5, 0.0, 0.99999);
		int wt = int(xn*52.0);                               // white cell, treble side first
		float press = keyState(white2midi(51 - wt));
		float lever = clamp((7.5 - pos.z)/15.0, 0.0, 1.0);   // front dips, back pivots
		pos.y += press*KEY_DIP_W*lever;
	}
	float re = MapBoxSim(pos, vec3(63.0, 1.5, 7.5));
	// press displacement jumps by up to KEY_DIP_W between adjacent keys, which
	// breaks the SDF's Lipschitz bound — rays overshoot INTO pressed keys and
	// shade their interiors as dark blotches. max(re/2, re-DIP) is conservative
	// (never tunnels) yet keeps FULL step size away from the keyboard.
	re = max(re*0.5, re - KEY_DIP_W);
	if (re <= EPSILON)
	{
		// the gap-zone band (key SIDES + a sliver of top/front at each gap)
		// is its own material: saturated wood — where the gap lines live
		float xn2 = clamp(pos.x/126.0 + 0.5, 0.0, 0.99999)*52.0;
		float d2b = (0.5 - abs(fract(xn2) - 0.5)) * (126.0/52.0);
		obj.m_obj_idx = (d2b < 0.14) ? 11 : 3;
	}
	obj.m_dist = re;
}

void MapLogo (in vec3 world_pos, out Obj obj)
{
	// AABB early-out: rays away from the logo pay ~nothing
	vec3 q = world_pos - vec3(LOGO_CX, LOGO_BASE_Y + 0.5*LINE_H*LOGO_SCALE, LOGO_Z);
	vec3 hs = vec3(0.5*LOGO_ADVW*LOGO_SCALE + 2.0, 0.6*LINE_H*LOGO_SCALE, LOGO_DEPTH);
	float db = length(max(abs(q) - hs, 0.0));
	if (db > 2.0) { obj.m_dist = db + 0.5; return; }
	// text reads left-to-right toward -x (front view), like the nameboard decal
	vec2 uvf = vec2((LOGO_CX + 0.5*LOGO_ADVW*LOGO_SCALE - world_pos.x)/LOGO_SCALE,
	                (world_pos.y - LOGO_BASE_Y)/LOGO_SCALE);
	// the glyph distance field is PREBAKED by Buffer A (rows >= 4) — the
	// letters are a real 3D object now: one texture fetch replaces re-running
	// the Bezier font math on every march step
	vec2 lm = (uvf - vec2(-2.0, -1.5)) / vec2(58.5, 14.0);
	float d2 = 1.0;
	if (all(greaterThan(lm, vec2(0.0))) && all(lessThan(lm, vec2(1.0)))){
		// quintic-warped bilinear (iq): C2-smooth field, no texel faceting
		vec2 tuv = vec2(lm.x, (4.0 + lm.y*(iResolution.y - 4.0))/iResolution.y);
		vec2 rp = tuv*iResolution.xy - 0.5;
		vec2 fp2 = fract(rp);
		fp2 = fp2*fp2*fp2*(fp2*(fp2*6.0 - 15.0) + 10.0);
		d2 = texture(iChannel3, (floor(rp) + fp2 + 0.5)/iResolution.xy).r;
	}
	vec2 w = vec2(d2*LOGO_SCALE, abs(q.z) - LOGO_DEPTH);
	float d = min(max(w.x, w.y), 0.0) + length(max(w, 0.0));   // exact extrusion
	obj.m_dist = d;
	if (d <= EPSILON) obj.m_obj_idx = 10;
}

//--------------------------------------------------------------------------------------------
void ReflectMap(in vec3 world_pos, out Obj obj)//just used duiring reflect rendering
{
	Obj obj_2;
	obj_2.m_obj_idx = -1;
	
	MapFloor(world_pos, obj);
	MapWall(world_pos, obj_2);
	Choose(obj, obj_2, obj);
	
	MapWhiteKeys(world_pos, obj_2);
	Choose(obj, obj_2, obj);
	
	MapBlackKeys(world_pos, obj_2);
	Choose(obj, obj_2, obj);
	
	MapBodySoundboard(world_pos,obj_2);
	Choose(obj, obj_2, obj);
	
	MapString(world_pos,obj_2);
	Choose(obj, obj_2, obj);
	
	MapGoldObjs(world_pos, obj_2);
	Choose(obj, obj_2, obj);

	if (!gSkipLogo && logoAlpha() > 0.0)
	{
		MapLogo(world_pos, obj_2);
		Choose(obj, obj_2, obj);
	}
}

//negative means inside
void Map(in vec3 world_pos, out Obj obj)//all rendering
{
	Obj obj_2;
	obj_2.m_obj_idx = -1;
	
	ReflectMap(world_pos, obj);
	MapPianoBody(world_pos, obj_2);
	Choose(obj, obj_2, obj);
}

void GetNormal(in vec3 pos, in vec3 cam_up, in vec3 cam_right, in vec3 cam_forward, in Obj obj, out vec3 normal)
{
	// KEYS: analytic box-face normal. The 4-tap gradient over the guarded,
	// carved, press-displaced key field jitters pixel-to-pixel, and that
	// jitter through the spec^32 + fresnel reflection terms IS the key
	// "hairs" (path diagnostic showed hit classification perfectly clean).
	if (obj.m_obj_idx == 3 || obj.m_obj_idx == 4 || obj.m_obj_idx == 11)
	{
		bool w = (obj.m_obj_idx != 4);
		vec3 p = pos + place_pos - (w ? vec3(-2.0, 80.5, -7.5) : vec3(-2.0, 82.75, -5.0));
		vec3 h = w ? vec3(63.0, 1.5, 7.5) : vec3(63.0, 0.75, 5.0);
		vec3 d = abs(p) - h;
		if (d.y >= d.x && d.y >= d.z)      normal = vec3(0.0, 1.0, 0.0);   // tops (undersides never visible)
		else if (d.z >= d.x)               normal = vec3(0.0, 0.0, sign(p.z));
		else                               normal = vec3(sign(p.x), 0.0, 0.0);
		return;
	}
	// 4-tap tetrahedron gradient (iq) — one fewer scene eval pair than central
	// differences, same quality. Signature kept; cam args unused.
	Obj o = obj;
	float h = EPSILON*0.6;
	vec2 k = vec2(1.0, -1.0);
	vec3 n = vec3(0.0);
	o.m_obj_idx = -1; Map(pos + k.xyy*h, o); n += k.xyy*o.m_dist;
	o.m_obj_idx = -1; Map(pos + k.yyx*h, o); n += k.yyx*o.m_dist;
	o.m_obj_idx = -1; Map(pos + k.yxy*h, o); n += k.yxy*o.m_dist;
	o.m_obj_idx = -1; Map(pos + k.xxx*h, o); n += k.xxx*o.m_dist;
	normal = normalize(n + vec3(1e-6, 0.0, 0.0));
}

void Reflect(in vec3 dir, in vec3 pos, out Obj obj, out vec3 reflect_pos)
{
	float march_len = 1.0;
	for (int i=0; i< (iFrame<0 ? 9999 : REFLECT_STEPS); ++i)   // iFrame sentinel: see MARCH_STEPS
	{
		reflect_pos = pos + dir * march_len;
		ReflectMap(reflect_pos, obj);
		if (obj.m_dist<= EPSILON)
		{
			reflect_pos += dir * obj.m_dist;
			break;
		}
		march_len += max(obj.m_dist, EPSILON*0.5);   // min-step (see Render)
	}
}

vec3 GetColorDiffuse(in Obj obj, in vec3 world_pos, in float c)
{
	if (obj.m_obj_idx == 0)
	{
		return texture(iChannel0, (world_pos.xz * 0.005)).rgb * c;
	}
	else if (obj.m_obj_idx == 1)
	{
		// wall + ceiling: ONE spherical panorama projection from the room
		// center — geometrically correct on the cylinder (the old
		// position*0.05-as-direction was a box-era mapping that warped), and
		// wall/ceiling are continuous across the rim (same projection center)
		return texture(iChannel1, normalize(world_pos - vec3(0.0, 100.0, 0.0)).zyx).rgb * c;
	}
	else if (obj.m_obj_idx == 2)
    {
        vec3 base = vec3(0.02*c, 0.02*c, 0.02*c);

        // Only the front wall lies on this plane; grab hits near it.
        if (abs(world_pos.z - DECAL_Z) < DECAL_ZTOL)
        {
            float u2f   = DECAL_TH / LINE_H;          // world units per font unit
            float halfW = 0.5 * DECAL_ADVW * u2f;     // half text width in world units

            // This panel faces -Z, so world +X reads to screen-LEFT.
            // => text u must grow as world.x DECREASES (keeps "Boesendorfer" un-mirrored).
            float fu = (DECAL_CX + halfW - world_pos.x) / u2f;
            float fv = (world_pos.y - (DECAL_CY - 0.3*DECAL_TH)) / u2f;

            if (fu >= 0.0 && fu <= DECAL_ADVW && fv >= 0.0 && fv <= LINE_H)
            {
                float a = col1(vec2(fu, fv)).a;       // reuse your existing color-string
                base = mix(base, DECAL_COLOR * (1. - abs(LINE_H * .5  - fv) / (LINE_H)), a);
            }
        }
        return base;
    }
	else if (obj.m_obj_idx == 3)
	{
		// gap lines as FILTERED COVERAGE of the 0.1-wide pulse train (fwidth
		// on the CONTINUOUS cell coordinate — after the fract fold it spikes
		// at every wrap = the old 1px "hairs"). The coverage blends toward
		// the key-side WOOD material: it IS the far-field LOD of the sides.
		// Golus-grid gap lines (bgolus "best darn grid shader"): width
		// CLAMPED to the pixel footprint with energy conservation (distant
		// lines get FAINTER, never noisier) + moire fade to the pattern
		// average as the footprint nears the cell size. LOD color = FLAT
		// wood average (a textured sample smears grain across the fronts).
		float xc = clamp((world_pos.x+place_pos.x+2.0)/126.0 + 0.5, 0.0, 1.0) * 52.0;
		float fw = max(fwidth(xc), 1e-5);
		float lineW = 0.10;                              // gap width, cell units
		float drawW = clamp(lineW, fw, 0.5);
		float d2b = abs(fract(xc + 0.5) - 0.5) * 2.0;    // 0 at boundary, 1 mid-key
		float cov = 1.0 - smoothstep(drawW - fw, drawW + fw, d2b);
		cov *= clamp(lineW/drawW, 0.0, 1.0);             // energy conservation
		cov = mix(cov, lineW, clamp(fw*2.0 - 0.5, 0.0, 1.0));   // moire fade
		vec3 woodAvg = KEYSIDE_TINT * KEYSIDE_GAIN * 0.7;
		return mix(vec3(1.0, 1.0, 0.90), woodAvg, cov) * c;
	}
	else if (obj.m_obj_idx == 11)
	{
		// key sides: saturated wood (KEYSIDE_TINT / KEYSIDE_GAIN to taste).
		// Once the side band is SUB-PIXEL, converge to exactly what obj3
		// paints there — otherwise the per-pixel obj3/obj11 id flip is
		// itself a lottery (the residual speckle at distance).
		vec3 w = texture(iChannel2, vec2(world_pos.x*0.05, world_pos.y*0.12)).rgb
		         * KEYSIDE_TINT * KEYSIDE_GAIN;
		float xc = clamp((world_pos.x+place_pos.x+2.0)/126.0 + 0.5, 0.0, 1.0) * 52.0;
		float fw = max(fwidth(xc), 1e-5);
		float lineW = 0.10;
		float drawW = clamp(lineW, fw, 0.5);
		float d2b = abs(fract(xc + 0.5) - 0.5) * 2.0;
		float cov = 1.0 - smoothstep(drawW - fw, drawW + fw, d2b);
		cov *= clamp(lineW/drawW, 0.0, 1.0);
		cov = mix(cov, lineW, clamp(fw*2.0 - 0.5, 0.0, 1.0));
		vec3 far3 = mix(vec3(1.0, 1.0, 0.90), KEYSIDE_TINT * KEYSIDE_GAIN * 0.7, cov);
		return mix(w, far3, clamp(fw*12.0 - 0.4, 0.0, 1.0)) * c;
	}
	else if (obj.m_obj_idx == 4)
	{
		return vec3(0.03*c, 0.03*c, 0.03*c);
	}
	else if (obj.m_obj_idx == 5)
	{
		// soundboard + gold materials imported 1:1 from the user's live
		// ShaderToy (Nc3GzB) — tuned against the 1f7dca9c wood on iChannel2
		float sinv = sin(0.7);
		float cosv = cos(0.7);
		vec2 uv = world_pos.xz * mat2(cosv, -sinv, sinv, cosv);
		uv.x = sin(uv.x*0.1);
		uv.y = sin(uv.y*0.002);
		return 2. * vec3(0.294,0.286,0.149)*c*texture(iChannel2, uv ).rgb;
	}
	else if (obj.m_obj_idx == 6)
	{
		float sinv = sin(0.7);
		float cosv = cos(0.7);
		vec2 uv = world_pos.xz * mat2(cosv, -sinv, sinv, cosv);
		uv.x = sin(uv.x*0.1);
		uv.y = sin(uv.y*0.02);
		return vec3(0.31,0.2,0.09)*texture(iChannel2, uv ).rgb * c;
	}
	else if (obj.m_obj_idx == 10)
	{
		float v = clamp((world_pos.y - LOGO_BASE_Y)/(LINE_H*LOGO_SCALE), 0.0, 1.0);
		return mix(vec3(0.55,0.20,0.05), vec3(1.00,0.55,0.12), v) * c;
	}
	return vec3(0.31,0.31,0.31)*c;
}

vec3 GetColor(in Obj obj, in vec3 world_pos, in vec3 dir, in vec3 cam_up, in vec3 cam_forward, in vec3 cam_right)
{
	float c = LightIntensity(world_pos);
	vec3 re = GetColorDiffuse(obj, world_pos, c);
	
	//specular
	vec3 normal = vec3(1.0,0.0,0.0);
	GetNormal(world_pos, cam_up, cam_right, cam_forward, obj, normal);
	if (obj.m_obj_idx == 10)   // logo: extrusion WALLS darker than the faces
		re *= mix(0.45, 1.0, smoothstep(0.35, 0.75, abs(normal.z)));
	vec3 ref_dir = normalize(reflect(dir, normal));
	float spec = dot(normalize(get_light_pos() - world_pos), ref_dir);
	spec = max(0.0,spec);
	
	//
	
	if (obj.m_obj_idx == 2 || ((obj.m_obj_idx == 3 || obj.m_obj_idx == 11) && KEY_REFLECT > 0.0))
	{
		// lacquered surfaces: the lid and the WHITE keys — tight specular +
		// a real raymarched reflection with fresnel falloff. Black keys stay
		// matte (reflections there just read as noise).
		spec *= 1.01;
		spec = pow(spec, 32.0);
		re += vec3(spec, spec, spec);
		
		vec3 reflect_color;
		if (obj.m_obj_idx == 3 || obj.m_obj_idx == 11)
		{
			// keys: clean ENVIRONMENT reflection from the cubemap — the
			// marched secondary ray grazes into the key gaps and lotteries
			reflect_color = texture(iChannel1, ref_dir).rgb * 0.8;
		}
		else
		{
			Obj obj_r;
			obj_r.m_obj_idx = -1;
			obj_r.m_dist = 9999.9;
			vec3 reflect_pos = world_pos;
			Reflect(ref_dir, world_pos + normal*(EPSILON*2.0), obj_r, reflect_pos);
			float cc = LightIntensity(reflect_pos);
			reflect_color = GetColorDiffuse(obj_r, reflect_pos, cc);
		}
		
		float brdf = 1.2- dot(ref_dir, normal);
		brdf = pow(brdf, 4.0);
		brdf = min(brdf, 0.9);   // grazing fresnel hits ~2.1 at silhouettes ->
		                         // the dotted bright OUTLINE around the body/legs
		if (obj.m_obj_idx != 2) brdf *= KEY_REFLECT;

		re += reflect_color * brdf;
	}
	else if (obj.m_obj_idx !=4 && obj.m_obj_idx !=0 && obj.m_obj_idx !=1)
	{
		spec *= 1.11;
		spec = pow(spec, 10.0);
		re += vec3(spec, spec, spec);
	}
	return re;
}

vec4 Render(in vec3 cam_pos, in vec3 cam_up, in vec3 cam_forward, in vec3 cam_right, in vec2 vp_pos)
{
	Obj obj;
	obj.m_obj_idx = -1;
	obj.m_dist = 9999.9;
	
	vec3 dir = normalize(cam_forward + vp_pos.x * cam_right + vp_pos.y * cam_up);
	vec3 pos;
	float march_len = 10.0;
	float best_d = 1e9, best_len = 10.0;                 // closest approach so far
	for (int i=0; i< (iFrame<0 ? 9999 : MARCH_STEPS); ++i)   // iFrame sentinel: keeps the loop ROLLED under ANGLE/D3D (unrolling 60x Map() explodes compile / crashes weak drivers)
	{
		pos = cam_pos + dir * march_len;
		Map(pos, obj);
		if (obj.m_dist<= EPSILON)
		{
			pos += dir * obj.m_dist;
			break;
		}
		if (obj.m_dist < best_d){ best_d = obj.m_dist; best_len = march_len; }
		march_len += max(obj.m_dist, EPSILON*0.5);   // min-step: grazing rays stall-crawl and exhaust the budget as dark bands
	}
	
	// budget exhausted while GRAZING a surface: a ray sliding ~0.1 units above
	// the key tops advances only ~0.1/step, so NO step budget survives shallow
	// angles. Shade the closest-approach surface instead of returning black —
	// this is what kills the dark bands across the keyboard.
	if (obj.m_obj_idx == -1 && best_d <= EPSILON*8.0)
	{
		pos = cam_pos + dir * best_len;
		vec3 n;
		GetNormal(pos, cam_up, cam_right, cam_forward, obj, n);
		// the key fields are GUARDED (report ~half the true distance), so one
		// projection step lands only halfway -> Map still > EPSILON -> no
		// material id -> the gray pixel-lottery "hairs". Iterate to converge.
		Map(pos, obj);
		for (int r = 0; r < (iFrame<0 ? 9 : 5); r++)
		{
			if (obj.m_dist <= EPSILON) break;
			pos -= n * obj.m_dist;
			Map(pos, obj);
		}
	}
	
	gDepth = (obj.m_obj_idx == -1) ? 1e6 : march_len;
	if (obj.m_obj_idx == -1)
		return vec4(0.000,0.000,0.000,1.0);
	
	vec3 color = GetColor(obj, pos, dir, cam_up, cam_forward, cam_right);
	float lA = logoAlpha();
	if (obj.m_obj_idx == 10 && lA < 1.0)
	{
		// the logo is fading: continue the ray BEHIND it and blend = true alpha
		gSkipLogo = true;
		Obj o2; o2.m_obj_idx = -1; o2.m_dist = 9999.9;
		vec3 p2 = pos; float t2 = 1.0;
		for (int i = 0; i < (iFrame<0 ? 9999 : 48); ++i)
		{
			p2 = pos + dir * t2;
			Map(p2, o2);
			if (o2.m_dist <= EPSILON) { p2 += dir * o2.m_dist; break; }
			t2 += max(o2.m_dist, EPSILON*0.5);
			if (t2 > 600.0) break;
		}
		vec3 behind = (o2.m_obj_idx == -1) ? vec3(0.0)
		            : GetColor(o2, p2, dir, cam_up, cam_forward, cam_right);
		gSkipLogo = false;
		color = mix(behind, color, lA);
	}
	return vec4(color, 1.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	float aspect = iResolution.y / iResolution.x;
	const float fov = 35.0 * 0.017453292519943295;
	
	vec3 target = vec3(0,75,0);//look at (0,75,0)
	vec3 cam_pos;
	if (length(iMouse.xy) > 10.0)
	{
		// mouse owns the camera: drag to orbit, auto-rotation stops.
		// Releasing keeps your angle; rewind/reload resumes the auto path.
		float az = 6.2831853*(iMouse.x/iResolution.x - 0.5);
		float el = clamp(1.4*(iMouse.y/iResolution.y - 0.35), -0.05, 1.0);
		cam_pos = target + vec3(130.0*cos(el)*sin(az), 25.0 + 130.0*sin(el), -130.0*cos(el)*cos(az));
	}
	else
	{
		float t = iTime * 0.08;
		cam_pos = vec3(-10.+sin(t)*100.0, 150.0+10.0*sin(0.33*t+9.876), -20.+cos(t+2.333333)*120.0);
	}
	vec3 cam_forward = normalize(target - cam_pos);
	vec3 cam_up = vec3(0.0,1.0,0.0);
	vec3 cam_right = normalize(cross(cam_forward, cam_up));
	cam_up = cross(cam_right, cam_forward);
	
	// ---- EDGE-ADAPTIVE supersampling --------------------------------------
	// One centered sample first (uniform control flow -> valid 2x2-quad
	// derivatives), then fwidth(luma) flags pixels sitting on a visible edge —
	// key gaps, lid silhouette, logo serifs, floor tile joints — and ONLY
	// those march AA_EDGE extra jittered rays. Flat areas (most of the frame)
	// stay single-sample: a few ms instead of the old whole-frame 4x.
	// SINGLE Render call site: the scene function is huge — two inlined copies
	// (one for the center tap, one in the extra loop) double the code and the
	// register pressure and slow EVERY pixel down, edge or not.
	vec2 vp_c = (fragCoord.xy/ iResolution.xy - 0.5) * 2.0 * vec2(1.0, aspect);
	// KEYS get unconditional supersampling (2x2, like AA): the sub-pixel slits
	// between key boxes pixel-lottery into "hairs" that the fwidth edge test
	// alone doesn't always flag. Ray-vs-AABB over the keyboard region.
	bool nearKeys;
	{
		vec3 kro = cam_pos - vec3(-12.0, 83.0, -46.0);
		vec3 khs = vec3(68.0, 9.0, 14.0);
		vec3 kdir = normalize(cam_forward + vp_c.x * cam_right + vp_c.y * cam_up);
		vec3 kiv = 1.0/kdir;
		vec3 kt0 = (-khs - kro)*kiv, kt1 = (khs - kro)*kiv;
		vec3 ktn = min(kt0, kt1), ktf = max(kt0, kt1);
		nearKeys = min(min(ktf.x, ktf.y), ktf.z) >= max(max(max(ktn.x, ktn.y), ktn.z), 0.0);
	}
	const vec2 ROOK[4] = vec2[4](vec2(0.0), vec2(0.375,0.125), vec2(-0.125,0.375), vec2(-0.3,-0.3));
	vec4 tot = vec4(0.0);
	float ns = 0.0;
	int want = 1;                                    // grows to 1+AA_EDGE on edge pixels
	for(int k=0; k<(iFrame<0?9:AA_EDGE+1); k++){     // sentinel keeps the loop rolled
		if (k >= want) break;
		vec2 off = ROOK[k>3?3:k];
		vec2 vp_pos = ((fragCoord.xy + off)/ iResolution.xy - 0.5) * 2.0;
		vec4 c = Render(cam_pos, cam_up, cam_forward, cam_right, vp_pos * vec2(1.0, aspect));
		tot += c; ns += 1.0;
		if (k == 0 && (nearKeys || fwidth(dot(c.rgb, vec3(0.299, 0.587, 0.114))) > AA_EDGE_T))
			want = 1 + AA_EDGE;                      // k=0 runs on ALL quad lanes -> fwidth valid
	}
	vec3 col = tot.rgb/ns;
	// ---- Magic particles: fade in at PARTICLES_T0, out with the logo ----
	if (iTime > PARTICLES_T0 && iTime < LOGO_FADE_T + LOGO_FADE_D + 0.5)
	{
		vec2 tuv = 0.5*vec2(vp_c.x, vp_c.y);
		col += drawParticles3D(tuv, cam_pos, cam_right, cam_up, cam_forward)*0.9;
	}
	fragColor = vec4(col, 1.0);
}"""


# Credit header stamped on every emitted tab (except the sacrificial Buffer B,
# which is deleted right after import).
TAB_HEADER = """/* MID2GLSL v1.3 (c) 2026 Orblivius
   Contact: orblivius@protonmail.com
   --------------------------------------
   GIT: https://github.com/mewza/mid2glsl

   ---- __TRACK__ --------------------------------------
   The sound engine was created by me and original concept I was playing with
   and here it is - FM synthesized Bösendorfer grand piano running on ShaderToy
   nonetheless!
   
   Piano model borrowed from https://www.shadertoy.com/view/lslGWf by jedi_cy */

"""

PIANO_FONT_CORE = r"""const float ATLAS_W   = 1.0;
const float LINE_H    = 11.0000;
const float FONT_BASE = 2.4000;

float dot2(vec2 v){return dot(v,v);}
float LD(vec2 p,vec2 a,vec2 b){
    vec2 pa=p-a,ba=b-a;
    return length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.,1.));
}
int LW(vec2 p,vec2 a,vec2 b){
    if((a.y>p.y)==(b.y>p.y)) return 0;
    float t=(p.y-a.y)/(b.y-a.y);
    if(a.x+t*(b.x-a.x)<p.x) return 0;
    return (b.y>a.y)?1:-1;
}
float QD(vec2 pos,vec2 p0,vec2 p1,vec2 p2){
    vec2 a=p1-p0,b=p0-2.*p1+p2,c=a*2.,d=p0-pos;
    float kk=1./dot(b,b),kx=kk*dot(a,b),
          ky=kk*(2.*dot(a,a)+dot(d,b))/3.,kz=kk*dot(d,a);
    float p=ky-kx*kx,q=kx*(2.*kx*kx-3.*ky)+kz;
    float p3=p*p*p,h=q*q+4.*p3,res;
    if(h>=0.){
        h=sqrt(h); vec2 x=(vec2(h,-h)-q)/2.;
        vec2 uv=sign(x)*pow(abs(x),vec2(1./3.));
        float t=clamp(uv.x+uv.y-kx,0.,1.);
        res=dot2(d+(c+b*t)*t);
    }else{
        float z=sqrt(-p),v=acos(q/(p*z*2.))/3.,m=cos(v),n=sin(v)*1.732050808;
        vec3 t3=clamp(vec3(m+m,-n-m,n-m)*z-kx,0.,1.);
        float dx=dot2(d+(c+b*t3.x)*t3.x),dy=dot2(d+(c+b*t3.y)*t3.y);
        res=min(dx,dy);
    }
    return sqrt(max(res,0.));
}
int QW(vec2 p,vec2 p0,vec2 p1,vec2 p2){
    float A=p0.y-2.*p1.y+p2.y,B=2.*(p1.y-p0.y),C=p0.y-p.y;
    int cnt=0;
    if(abs(A)<1e-6){
        if(abs(B)>1e-6){float t=-C/B;
            if(t>=0.&&t<=1.){float x=mix(mix(p0.x,p1.x,t),mix(p1.x,p2.x,t),t);
                if(x>p.x) cnt+=(p2.y>p0.y)?1:-1;}}
    }else{float disc=B*B-4.*A*C;
        if(disc>=0.){float sq=sqrt(disc);
            for(int s=-1;s<=1;s+=2){float t=(-B+float(s)*sq)/(2.*A);
                if(t>=0.&&t<=1.){float x=mix(mix(p0.x,p1.x,t),mix(p1.x,p2.x,t),t);
                    if(x>p.x){float dy=2.*((1.-t)*(p1.y-p0.y)+t*(p2.y-p1.y));
                        cnt+=(dy>0.)?1:-1;}}}}}
    return cnt;
}

float sdf66(vec2 p){
  if(!all(lessThan(abs(p-vec2(4.7520,5.4570)),vec2(4.5800,6.1100)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[37]=vec4[](vec4(5.2440,1.0200,5.3520,0.8880),vec4(3.9120,4.4280,3.9720,6.3120),vec4(3.6720,6.7080,3.6720,6.8040),vec4(3.9600,7.1400,3.9720,8.6160),vec4(3.9720,8.6160,3.6720,8.4960),vec4(5.8440,10.3200,6.7440,10.9560),vec4(7.3080,9.9840,7.5120,9.9840),vec4(8.5680,5.7360,8.8320,5.5080),vec4(8.8320,5.5080,8.8200,5.4120),vec4(8.8200,5.4120,8.6040,5.0880),vec4(7.1760,3.3720,6.6960,3.0720),vec4(6.6960,3.0720,5.6160,2.4240),vec4(5.6160,2.4240,5.5320,2.4240),vec4(4.3560,1.0080,4.4640,0.9960),vec4(5.8320,10.8360,5.9880,10.8360),vec4(5.9880,10.8360,6.0000,10.7040),vec4(5.4360,3.6840,5.4120,9.7080),vec4(5.4120,9.7080,4.9680,9.4200),vec4(4.9680,9.4200,4.9560,5.0400),vec4(3.9960,3.9960,3.9960,3.9600),vec4(6.5880,7.4760,6.5880,3.2280),vec4(6.5880,3.2280,6.6000,3.2400),vec4(7.2000,5.5080,7.2000,5.6280),vec4(5.7120,6.1440,5.7240,3.4920),vec4(6.2520,3.1200,6.2520,6.0240),vec4(6.1200,6.2160,6.1080,6.2160),vec4(6.2280,7.2840,6.2640,9.0000),vec4(6.2640,9.0000,6.2280,9.3720),vec4(5.8440,10.0680,5.7240,9.9720),vec4(5.7240,9.9720,5.7480,7.2600),vec4(5.7480,7.2600,6.2280,7.2840),vec4(6.5880,9.0000,6.5880,7.6440),vec4(6.5880,7.6440,6.6000,7.6560),vec4(6.9720,8.5200,6.9720,8.5320),vec4(6.2640,6.4680,6.2640,6.9720),vec4(6.2640,6.9720,5.9880,7.0200),vec4(5.7240,7.0320,5.7240,6.4680));
  for(int i=0;i<(iFrame<0?9999:37);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[71]=vec4[](vec4(4.4640,0.9960,4.8180,0.7650),vec4(5.3520,0.8880,4.5090,-0.1530),vec4(3.0360,0.3360,1.5780,1.0530),vec4(2.0640,2.5440,2.5650,3.7170),vec4(3.8040,3.9360,3.5970,4.1550),vec4(3.9720,6.3120,3.9210,6.5730),vec4(3.6720,6.8040,3.8340,6.8820),vec4(3.6720,8.4960,3.4950,8.4600),vec4(3.5160,8.6760,3.8404,8.9239),vec4(4.5315,9.4305,5.2226,9.9371),vec4(6.7440,10.9560,7.0050,11.0670),vec4(6.9600,10.9080,6.8760,10.7130),vec4(6.7560,10.6440,6.9780,10.3410),vec4(7.5120,9.9840,7.6380,9.9120),vec4(7.6200,9.8400,7.5480,9.6810),vec4(7.5480,9.4680,7.5870,9.3360),vec4(7.6440,8.9520,7.6680,8.1300),vec4(7.0440,7.6680,7.4520,7.4940),vec4(7.8960,7.1040,8.2590,6.7200),vec4(8.5680,5.8680,8.5830,5.8290),vec4(8.5800,5.8080,8.5650,5.7720),vec4(8.6040,5.0880,8.4120,4.4400),vec4(8.0760,4.0800,7.7340,3.6900),vec4(5.5320,2.4240,5.0520,2.9070),vec4(4.3920,3.1560,3.9120,3.2790),vec4(3.6120,3.2400,2.7862,3.1676),vec4(2.4780,2.5125,2.1698,1.8574),vec4(2.4600,1.2720,2.7698,0.7560),vec4(3.3360,0.6270,3.9022,0.4980),vec4(2.0160,9.3360,1.9410,9.1410),vec4(1.8120,9.1080,1.7610,9.0810),vec4(1.5840,9.1080,1.4370,9.1500),vec4(1.3800,9.3360,1.3440,9.6300),vec4(1.3800,9.6720,1.4670,9.8730),vec4(1.6800,9.9480,1.8870,9.9600),vec4(1.9320,9.9360,2.1990,9.8310),vec4(2.2680,9.4920,2.3040,9.2460),vec4(2.2680,9.1080,2.0310,8.5290),vec4(1.4160,8.5440,0.9300,8.6220),vec4(0.7680,9.0240,0.6840,9.0630),vec4(0.6720,9.6960,0.7350,10.0080),vec4(0.7800,10.0320,0.9240,10.3770),vec4(1.1760,10.5600,1.5510,10.8150),vec4(1.8720,10.8720,2.2680,10.9890),vec4(2.8080,10.9080,2.9884,10.8641),vec4(3.4995,10.6515,4.0106,10.4389),vec4(4.3080,10.3680,4.5480,10.2960),vec4(4.7520,10.3320,4.7959,10.3444),vec4(5.1975,10.5075,5.5991,10.6706),vec4(6.0000,10.7040,5.0940,9.9120),vec4(4.5120,9.6960,3.5940,9.4320),vec4(2.7120,9.9240,2.3730,10.1130),vec4(2.0160,10.2120,2.0130,10.2120),vec4(1.9920,10.2120,1.7790,10.2750),vec4(1.5120,10.2120,1.0747,10.0046),vec4(1.0020,9.5205,0.9293,9.0364),vec4(1.3920,8.8200,1.9470,8.8080),vec4(3.9960,3.9600,4.8420,3.9750),vec4(4.9560,5.0400,4.8630,4.2840),vec4(6.6000,3.2400,7.1820,3.6641),vec4(7.4040,4.3695,7.6260,5.0749),vec4(7.2000,5.6280,7.8900,6.8280),vec4(6.6000,7.4520,6.5940,7.4550),vec4(6.1080,6.2160,5.9190,6.2970),vec4(5.7240,3.4920,6.0690,3.2880),vec4(6.2520,6.0240,6.2400,6.2280),vec4(6.2280,9.3720,6.1710,9.8730),vec4(6.6000,7.6560,7.1730,8.1600),vec4(6.9720,8.5320,6.7530,8.9550),vec4(5.9880,7.0200,5.8920,7.0620),vec4(5.7240,6.4680,6.0660,6.4050));
  const vec2 QB[71]=vec2[](vec2(5.2440,1.0200),vec2(3.0360,0.3360),vec2(2.0640,2.5440),vec2(3.8040,3.9360),vec2(3.9120,4.4280),vec2(3.6720,6.7080),vec2(3.9600,7.1400),vec2(3.5160,8.6760),vec2(4.5315,9.4305),vec2(5.8440,10.3200),vec2(6.9600,10.9080),vec2(6.7560,10.6440),vec2(7.3080,9.9840),vec2(7.6200,9.8400),vec2(7.5480,9.4680),vec2(7.6440,8.9520),vec2(7.0440,7.6680),vec2(7.8960,7.1040),vec2(8.5680,5.8680),vec2(8.5800,5.8080),vec2(8.5680,5.7360),vec2(8.0760,4.0800),vec2(7.1760,3.3720),vec2(4.3920,3.1560),vec2(3.6120,3.2400),vec2(2.4780,2.5125),vec2(2.4600,1.2720),vec2(3.3360,0.6270),vec2(4.3560,1.0080),vec2(1.8120,9.1080),vec2(1.5840,9.1080),vec2(1.3800,9.3360),vec2(1.3800,9.6720),vec2(1.6800,9.9480),vec2(1.9320,9.9360),vec2(2.2680,9.4920),vec2(2.2680,9.1080),vec2(1.4160,8.5440),vec2(0.7680,9.0240),vec2(0.6720,9.6960),vec2(0.7800,10.0320),vec2(1.1760,10.5600),vec2(1.8720,10.8720),vec2(2.8080,10.9080),vec2(3.4995,10.6515),vec2(4.3080,10.3680),vec2(4.7520,10.3320),vec2(5.1975,10.5075),vec2(5.8320,10.8360),vec2(4.5120,9.6960),vec2(2.7120,9.9240),vec2(2.0160,10.2120),vec2(1.9920,10.2120),vec2(1.5120,10.2120),vec2(1.0020,9.5205),vec2(1.3920,8.8200),vec2(2.0160,9.3360),vec2(5.4360,3.6840),vec2(3.9960,3.9960),vec2(7.4040,4.3695),vec2(7.2000,5.5080),vec2(6.6000,7.4520),vec2(6.5880,7.4760),vec2(5.7120,6.1440),vec2(6.2520,3.1200),vec2(6.1200,6.2160),vec2(5.8440,10.0680),vec2(6.9720,8.5200),vec2(6.5880,9.0000),vec2(5.7240,7.0320),vec2(6.2640,6.4680));
  for(int i=0;i<(iFrame<0?9999:71);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}
float sdf100(vec2 p){
  if(!all(lessThan(abs(p-vec2(2.2410,6.3240)),vec2(2.0810,4.4240)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[19]=vec4[](vec4(1.9080,2.4000,1.7640,2.4000),vec4(1.7640,2.4000,1.3320,2.6880),vec4(0.8760,2.9640,0.8880,3.1320),vec4(0.8880,3.1320,1.0440,3.3000),vec4(1.0440,3.3000,1.0440,6.5520),vec4(0.8760,6.7680,0.8760,6.9000),vec4(0.6840,9.2040,0.6600,9.3720),vec4(0.8280,10.2480,0.9120,10.2360),vec4(0.9120,10.2360,3.7320,7.3080),vec4(3.7320,7.1040,3.5880,6.9840),vec4(3.5880,6.9840,3.6000,3.6600),vec4(3.6000,3.6600,3.6720,3.6000),vec4(3.6720,3.6000,3.6720,3.4080),vec4(1.9200,2.4120,1.9080,2.4000),vec4(2.3400,7.4760,1.8840,7.1760),vec4(1.8840,7.1760,1.8960,3.2880),vec4(2.2080,3.0600,2.2440,3.0480),vec4(2.7120,3.3240,2.7120,7.1040),vec4(2.7120,7.1040,2.3400,7.4760));
  for(int i=0;i<(iFrame<0?9999:19);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[9]=vec4[](vec4(1.3320,2.6880,1.1040,2.8620),vec4(1.0440,6.5520,1.0320,6.6510),vec4(0.8760,6.9000,1.4250,7.1190),vec4(2.1360,7.6800,1.4460,8.5680),vec4(0.6600,9.3720,0.7980,9.6210),vec4(3.7320,7.3080,3.8220,7.2150),vec4(3.6720,3.4080,2.7600,2.9640),vec4(1.8960,3.2880,2.0610,3.1920),vec4(2.2440,3.0480,2.4330,3.0240));
  const vec2 QB[9]=vec2[](vec2(0.8760,2.9640),vec2(0.8760,6.7680),vec2(2.1360,7.6800),vec2(0.6840,9.2040),vec2(0.8280,10.2480),vec2(3.7320,7.1040),vec2(1.9200,2.4120),vec2(2.2080,3.0600),vec2(2.7120,3.3240));
  for(int i=0;i<(iFrame<0?9999:9);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}
float sdf101(vec2 p){
  if(!all(lessThan(abs(p-vec2(1.8360,5.1060)),vec2(1.6760,3.2060)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[8]=vec4[](vec4(1.7160,3.3360,1.7280,3.2160),vec4(0.6600,3.2040,0.8040,3.4440),vec4(0.8040,3.4440,0.8040,6.6240),vec4(0.8040,6.6240,0.7080,6.7560),vec4(0.7080,6.7560,0.7080,6.9600),vec4(2.2080,7.8000,2.3520,7.8120),vec4(2.9880,6.5160,2.9880,6.3720),vec4(1.5960,6.9960,1.6200,3.5760));
  for(int i=0;i<(iFrame<0?9999:8);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[17]=vec4[](vec4(1.7280,3.2160,1.4220,2.8170),vec4(1.1880,2.4000,1.1580,2.4660),vec4(1.0560,2.6760,0.8310,3.0300),vec4(0.7080,6.9600,1.6830,7.3350),vec4(2.3520,7.8120,2.7060,6.7410),vec4(2.9880,6.3720,2.1810,6.0930),vec4(2.0760,5.5440,2.0340,4.8930),vec4(2.3880,4.9080,2.3010,5.0940),vec4(2.2680,5.2440,2.2965,5.7686),vec4(2.6490,5.7015,3.0015,5.6344),vec4(3.0120,5.2680,2.7300,4.5450),vec4(2.4840,4.6680,2.4210,4.6320),vec4(2.1600,4.7040,1.8570,4.8630),vec4(1.8240,5.1480,1.7550,5.9250),vec4(2.1000,6.1800,1.9620,6.6990),vec4(1.7160,7.0920,1.6200,7.0620),vec4(1.6200,3.5760,1.7040,3.4650));
  const vec2 QB[17]=vec2[](vec2(1.1880,2.4000),vec2(1.0560,2.6760),vec2(0.6600,3.2040),vec2(2.2080,7.8000),vec2(2.9880,6.5160),vec2(2.0760,5.5440),vec2(2.3880,4.9080),vec2(2.2680,5.2440),vec2(2.6490,5.7015),vec2(3.0120,5.2680),vec2(2.4840,4.6680),vec2(2.1600,4.7040),vec2(1.8240,5.1480),vec2(2.1000,6.1800),vec2(1.7160,7.0920),vec2(1.5960,6.9960),vec2(1.7160,3.3360));
  for(int i=0;i<(iFrame<0?9999:17);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}
float sdf102(vec2 p){
  if(!all(lessThan(abs(p-vec2(2.2155,6.3960)),vec2(2.0555,4.4480)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[19]=vec4[](vec4(2.1600,3.1560,2.1720,3.0360),vec4(1.6440,2.4480,1.5120,2.4480),vec4(1.5120,2.4480,1.4400,2.6280),vec4(1.1040,3.0840,1.1040,3.1920),vec4(1.1040,3.1920,1.2480,3.4080),vec4(1.2480,3.4080,1.2480,7.3800),vec4(1.1520,7.4520,0.6960,7.4640),vec4(0.6960,7.4640,0.6600,7.6440),vec4(0.7080,7.7400,1.2360,7.7400),vec4(1.2360,7.7400,1.2360,9.0240),vec4(1.2360,9.0240,1.0320,9.2160),vec4(1.0320,9.2160,1.0320,9.3840),vec4(2.6640,10.3440,2.7720,10.3320),vec4(1.9560,9.3960,1.9560,7.8000),vec4(2.0760,7.7160,2.6280,7.7040),vec4(2.6280,7.7040,2.7240,7.4400),vec4(2.7240,7.4400,1.9920,7.4160),vec4(1.9920,7.4160,1.9560,3.4200),vec4(1.9560,3.4200,2.1600,3.1560));
  for(int i=0;i<(iFrame<0?9999:19);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[10]=vec4[](vec4(2.1720,3.0360,1.8720,2.7600),vec4(1.4400,2.6280,1.2720,2.9280),vec4(1.2480,7.3800,1.2720,7.4160),vec4(0.6600,7.6440,0.6750,7.7730),vec4(1.0320,9.3840,1.8210,9.7380),vec4(2.7720,10.3320,2.9640,9.7890),vec4(3.6960,10.1640,3.7710,9.5220),vec4(3.1440,9.3120,2.4210,9.2730),vec4(2.1840,9.6120,2.0430,9.6660),vec4(1.9560,7.8000,1.9890,7.6950));
  const vec2 QB[10]=vec2[](vec2(1.6440,2.4480),vec2(1.1040,3.0840),vec2(1.1520,7.4520),vec2(0.7080,7.7400),vec2(2.6640,10.3440),vec2(3.6960,10.1640),vec2(3.1440,9.3120),vec2(2.1840,9.6120),vec2(1.9560,9.3960),vec2(2.0760,7.7160));
  for(int i=0;i<(iFrame<0?9999:10);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}
float sdf110(vec2 p){
  if(!all(lessThan(abs(p-vec2(2.0940,5.0220)),vec2(1.9340,3.1820)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[16]=vec4[](vec4(3.5160,3.1320,3.5040,3.0240),vec4(3.0480,2.3760,2.9280,2.3760),vec4(2.4360,3.0840,2.4360,3.1680),vec4(2.4360,3.1680,2.6040,3.2640),vec4(2.6040,3.2640,2.6040,6.4440),vec4(1.6560,6.5640,1.6440,3.3600),vec4(1.6440,3.3600,1.8600,3.1560),vec4(1.8600,3.1560,1.8480,3.0600),vec4(1.3440,2.3400,1.2480,2.3400),vec4(0.6600,2.9880,0.6600,3.1920),vec4(0.9240,3.5280,0.9240,6.6480),vec4(1.1280,7.4520,1.2600,7.6800),vec4(1.2600,7.6800,1.3560,7.6800),vec4(2.9160,7.6920,3.0000,7.7040),vec4(3.3240,6.5760,3.3240,3.2760),vec4(3.3240,3.2760,3.5160,3.1320));
  for(int i=0;i<(iFrame<0?9999:16);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[12]=vec4[](vec4(3.5040,3.0240,3.1140,2.6190),vec4(2.9280,2.3760,2.5620,3.1740),vec4(2.4480,3.0720,2.4330,3.0780),vec4(2.6040,6.4440,2.1210,7.2150),vec4(1.8480,3.0600,1.5330,2.7450),vec4(1.2480,2.3400,1.0350,2.8170),vec4(0.6600,3.1920,0.8370,3.2700),vec4(0.9240,6.6480,0.8910,6.8520),vec4(0.6960,6.9120,1.0560,7.1910),vec4(1.3560,7.6800,2.1450,6.4440),vec4(3.0000,7.7040,3.2280,7.2930),vec4(3.5280,7.0800,3.4350,6.9900));
  const vec2 QB[12]=vec2[](vec2(3.0480,2.3760),vec2(2.4480,3.0720),vec2(2.4360,3.0840),vec2(1.6560,6.5640),vec2(1.3440,2.3400),vec2(0.6600,2.9880),vec2(0.9240,3.5280),vec2(0.6960,6.9120),vec2(1.1280,7.4520),vec2(2.9160,7.6920),vec2(3.5280,7.0800),vec2(3.3240,6.5760));
  for(int i=0;i<(iFrame<0?9999:12);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}
float sdf111(vec2 p){
  if(!all(lessThan(abs(p-vec2(2.0985,5.1120)),vec2(1.9385,3.1400)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[13]=vec4[](vec4(1.7160,2.5080,1.6920,2.4720),vec4(1.6920,2.4720,1.5480,2.4720),vec4(1.5480,2.4720,1.1160,2.7600),vec4(0.6600,3.0360,0.6600,3.2040),vec4(0.6600,3.2040,0.8280,3.3840),vec4(0.6960,6.8520,0.6960,7.0080),vec4(2.4720,7.7040,2.5080,7.7520),vec4(2.5080,7.7520,2.6160,7.7520),vec4(3.4440,6.9480,3.3240,6.6480),vec4(3.3240,6.6480,3.3120,3.6840),vec4(3.3120,3.6840,3.4800,3.4320),vec4(2.5440,3.3120,2.5440,6.8280),vec4(1.7280,6.9240,1.7160,3.3720));
  for(int i=0;i<(iFrame<0?9999:13);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[14]=vec4[](vec4(3.4800,3.4320,2.5800,3.0780),vec4(1.1160,2.7600,0.8790,2.9250),vec4(0.8280,3.3840,0.8460,3.3780),vec4(0.8640,3.5160,0.8640,3.8235),vec4(0.8640,5.1570,0.8640,6.4905),vec4(0.8640,6.6720,0.8520,6.7890),vec4(0.6960,7.0080,1.0219,7.0954),vec4(1.6065,7.3155,2.1911,7.5356),vec4(2.6160,7.7520,3.0240,7.4550),vec4(3.4320,7.2120,3.5370,7.0800),vec4(2.1000,3.0840,2.3310,3.2610),vec4(2.5440,6.8280,2.3070,6.9480),vec4(2.0880,7.1040,1.8810,7.0770),vec4(1.7160,3.3720,1.9350,3.2730));
  const vec2 QB[14]=vec2[](vec2(1.7160,2.5080),vec2(0.6600,3.0360),vec2(0.8640,3.5160),vec2(0.8640,5.1570),vec2(0.8640,6.6720),vec2(0.6960,6.8520),vec2(1.6065,7.3155),vec2(2.4720,7.7040),vec2(3.4320,7.2120),vec2(3.4440,6.9480),vec2(2.5440,3.3120),vec2(2.0880,7.1040),vec2(1.7280,6.9240),vec2(2.1000,3.0840));
  for(int i=0;i<(iFrame<0?9999:14);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}
float sdf114(vec2 p){
  if(!all(lessThan(abs(p-vec2(2.0280,5.0700)),vec2(1.8680,3.0980)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[15]=vec4[](vec4(1.7400,3.2760,1.7400,3.1080),vec4(1.2120,2.4720,1.0560,2.6280),vec4(1.0560,2.6280,1.0560,2.7240),vec4(0.7320,3.1800,0.7320,3.2880),vec4(0.7320,3.2880,0.8280,3.3720),vec4(0.8280,3.3720,0.8160,6.8160),vec4(0.8160,6.8160,0.6600,6.9360),vec4(0.6600,6.9360,1.1040,7.4760),vec4(1.1040,7.4760,1.1040,7.5840),vec4(1.1040,7.5840,1.2720,7.6560),vec4(2.5320,7.6680,2.6280,7.6680),vec4(2.6280,7.6680,3.1080,7.1400),vec4(3.1080,7.1400,3.1080,6.9360),vec4(1.6680,6.7440,1.6680,3.3840),vec4(1.6680,3.3840,1.7400,3.2760));
  for(int i=0;i<(iFrame<0?9999:15);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[19]=vec4[](vec4(1.7400,3.1080,1.4670,2.8800),vec4(1.0560,2.7240,0.9840,3.0330),vec4(1.2720,7.6560,1.4160,7.4040),vec4(1.6320,7.2240,1.9110,7.1250),vec4(2.1360,7.2240,2.3250,7.3740),vec4(3.1080,6.9360,2.8470,6.6750),vec4(2.6400,6.4320,2.3850,6.0810),vec4(2.2560,5.6040,2.2005,5.3655),vec4(2.2890,5.0280,2.3775,4.6905),vec4(2.6280,4.6320,2.9700,4.6740),vec4(2.9160,4.7160,2.3850,5.1000),vec4(2.8080,5.4480,3.2970,5.4930),vec4(3.3720,5.0880,3.3960,4.8360),vec4(3.2400,4.6560,3.2250,4.6380),vec4(3.1920,4.6200,3.0780,4.4100),vec4(2.6040,4.4160,2.1630,4.4670),vec4(2.0280,5.1480,2.0430,6.1470),vec4(2.3640,6.5160,2.1750,6.9180),vec4(1.7880,6.8880,1.6830,6.8520));
  const vec2 QB[19]=vec2[](vec2(1.2120,2.4720),vec2(0.7320,3.1800),vec2(1.6320,7.2240),vec2(2.1360,7.2240),vec2(2.5320,7.6680),vec2(2.6400,6.4320),vec2(2.2560,5.6040),vec2(2.2890,5.0280),vec2(2.6280,4.6320),vec2(2.9160,4.7160),vec2(2.8080,5.4480),vec2(3.3720,5.0880),vec2(3.2400,4.6560),vec2(3.1920,4.6200),vec2(2.6040,4.4160),vec2(2.0280,5.1480),vec2(2.3640,6.5160),vec2(1.7880,6.8880),vec2(1.6680,6.7440));
  for(int i=0;i<(iFrame<0?9999:19);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}
float sdf115(vec2 p){
  if(!all(lessThan(abs(p-vec2(2.1900,5.0925)),vec2(2.0300,3.1715)))) return 1e9;
  float d=1e9; int w=0;
  const vec4 LS[14]=vec4[](vec4(2.9640,6.1080,3.6720,5.6760),vec4(3.6720,5.6760,3.6720,5.5560),vec4(2.2920,2.4720,2.1600,2.4720),vec4(0.6720,2.4840,0.6600,2.6400),vec4(0.6600,2.6400,1.7760,4.8000),vec4(0.9000,5.2320,0.9000,5.3400),vec4(0.9000,5.3400,1.1280,5.5920),vec4(1.1280,5.5920,1.1280,6.5280),vec4(2.4240,7.7640,2.5320,7.7640),vec4(3.5640,7.5840,3.6960,7.4520),vec4(1.3920,3.3120,1.4160,3.3240),vec4(2.6400,3.0960,2.8080,3.1920),vec4(2.8080,3.1920,2.8200,5.1360),vec4(1.8600,7.0920,1.8600,5.5800));
  for(int i=0;i<(iFrame<0?9999:14);i++){vec2 a=LS[i].xy,b=LS[i].zw;d=min(d,LD(p,a,b));w+=LW(p,a,b);}
  const vec4 QA[24]=vec4[](vec4(3.6720,5.5560,3.6060,5.4330),vec4(3.5400,5.0040,3.5291,4.8019),vec4(3.5295,4.3455,3.5299,3.8891),vec4(3.5640,3.7680,3.5970,3.5940),vec4(3.7200,3.4920,3.6623,3.3878),vec4(3.1320,3.0630,2.6018,2.7382),vec4(2.1600,2.4720,1.6920,3.1522),vec4(1.3320,2.9550,0.9720,2.7577),vec4(0.8280,2.4840,0.7320,2.4210),vec4(1.7760,4.8000,1.4820,4.9800),vec4(1.1280,6.5280,1.1130,6.7500),vec4(1.0800,6.7920,1.0110,6.9060),vec4(0.9600,6.9480,1.6740,7.1850),vec4(2.5320,7.7640,2.8950,7.2870),vec4(3.6960,7.4520,3.6690,7.2690),vec4(3.3000,6.7080,3.0150,6.2550),vec4(1.9440,4.3320,1.3710,3.4080),vec4(1.4160,3.3240,2.0640,3.6780),vec4(2.8200,5.1360,2.6790,5.3190),vec4(2.5560,5.3760,2.3130,4.9440),vec4(2.1720,5.3760,2.1941,5.4041),vec4(2.5065,5.9115,2.8189,6.4189),vec4(3.1560,7.0680,2.4900,6.7830),vec4(1.8600,5.5800,1.9260,5.4240));
  const vec2 QB[24]=vec2[](vec2(3.5400,5.0040),vec2(3.5295,4.3455),vec2(3.5640,3.7680),vec2(3.7200,3.4920),vec2(3.1320,3.0630),vec2(2.2920,2.4720),vec2(1.3320,2.9550),vec2(0.8280,2.4840),vec2(0.6720,2.4840),vec2(0.9000,5.2320),vec2(1.0800,6.7920),vec2(0.9600,6.9480),vec2(2.4240,7.7640),vec2(3.5640,7.5840),vec2(3.3000,6.7080),vec2(2.9640,6.1080),vec2(1.3920,3.3120),vec2(2.6400,3.0960),vec2(2.5560,5.3760),vec2(1.9440,4.3320),vec2(2.5065,5.9115),vec2(3.1560,7.0680),vec2(1.8600,7.0920),vec2(2.1720,5.3760));
  for(int i=0;i<(iFrame<0?9999:24);i++){vec2 a=QA[i].xy,b=QA[i].zw,c=QB[i];d=min(d,QD(p,a,b,c));w+=QW(p,a,b,c);}
  return d*(w!=0?-1.:1.);}

int _GC = 0;

vec4 getGlyphRect(int c){
  _GC=c;
  switch(c){
  case 66: return vec4(0.6622,0.1802,8.1698,10.8212);
  case 100: return vec4(0.6600,2.4000,3.1170,7.8480);
  case 101: return vec4(0.6600,2.4000,2.3551,5.4120);
  case 102: return vec4(0.6584,2.4480,3.0727,7.8960);
  case 110: return vec4(0.6600,2.3400,2.8680,5.3640);
  case 111: return vec4(0.6600,2.4720,2.8276,5.2800);
  case 114: return vec4(0.6600,2.4720,2.7173,5.1960);
  case 115: return vec4(0.6600,2.4492,3.0600,5.3148);
  default: return vec4(0.);
  }
}

vec4 getGlyphMeta(int c){
  switch(c){
  case 66: return vec4(0.6622,0.9986,9.4920,10.8212);
  case 100: return vec4(0.6600,1.7520,4.4400,7.8480);
  case 101: return vec4(0.6600,4.1880,3.6720,5.4120);
  case 102: return vec4(0.6584,1.6560,4.3920,7.8960);
  case 110: return vec4(0.6600,4.2960,4.1880,5.3640);
  case 111: return vec4(0.6600,4.2480,4.1520,5.2800);
  case 114: return vec4(0.6600,4.3320,4.0320,5.1960);
  case 115: return vec4(0.6600,4.2360,4.3800,5.3148);
  default: return vec4(0.);
  }
}

float cSDF(vec2 p) {
  if(_GC==66) return sdf66(p);
  if(_GC==100) return sdf100(p);
  if(_GC==101) return sdf101(p);
  if(_GC==102) return sdf102(p);
  if(_GC==110) return sdf110(p);
  if(_GC==111) return sdf111(p);
  if(_GC==114) return sdf114(p);
  if(_GC==115) return sdf115(p);
  return 1e9;
}"""


def build_movie_glsl(image_code, notes, info):
    """Flatten the viz-3 Image pass into ONE GLSL-120 shader for offline movie
    renderers (shadertoy-render.py etc.): no buffers (key/pedal states scanned
    per pixel from float-encoded MIDI; logo SDF analytic), no texelFetch, no
    switch, iGlobalTime, cubemap wall remapped to 2D. Video only — mux audio
    from the numpy-mirror WAV afterwards."""
    import re as _re
    code = image_code

    # ---- float-encoded MIDI data (GLSL 120 has no uint) ----
    Q, shift = info['Q'], info['shift']
    blk = 1 << shift
    N = len(notes)
    ON, NG, RV = [], [], []
    for (onT, dT, n, v, gT) in notes:
        ON.append(float(onT))
        NG.append(float((n - info['note_min'] + (info['note_min'] - 21)) ) + float(min(gT, 65535))*128.0)  # (midi-21) + gate*128
        RV.append(float(min(dT, 65535))*128.0 + float(max(0, min(127, v))))
    # block index: start*128 + count
    SC, idx = [], 0
    for b in range(info['nblocks']):
        start, cnt = idx, 0
        while idx < N and (notes[idx][0] >> shift) == b:
            idx += 1; cnt += 1
        SC.append(float(start)*128.0 + float(min(cnt, 127)))
    maxcnt = int(max(sc % 128.0 for sc in SC)) if SC else 1

    def farr(name, vals):
        # vec4-packed (4 floats per element: quarters BOTH the declared item
        # count and the const-slot cost vs scalar float arrays), sharded at
        # <=1024 items (RTX-class compiler cap), <=16 vec4s per line (ancient
        # GLSL parsers — Apple GL2 — choke on multi-kilobyte single lines),
        # read through a same-name accessor. GLSL-120-safe: no bitops, no
        # dynamic vector subscript (r==0?…: chain instead).
        fmt = lambda v: (("%.1f" % v).rstrip("0").rstrip(".") + "."
                         if float(v).is_integer() else ("%.1f" % v))
        toks = [fmt(v) for v in vals] or ["0."]
        while len(toks) % 4: toks.append("0.")
        packed = ["vec4(%s)" % ",".join(toks[i:i+4]) for i in range(0, len(toks), 4)]
        chunks = [packed[i:i+MAX_ARRAY_ITEMS] for i in range(0, len(packed), MAX_ARRAY_ITEMS)]
        decls = []
        for c, ch in enumerate(chunks):
            lines = [",".join(ch[i:i+16]) for i in range(0, len(ch), 16)]
            decls.append("const vec4 %s_%d[%d] = vec4[%d](\n%s);"
                         % (name, c, len(ch), len(ch), ",\n".join(lines)))
        acc = ["float %s(int i){" % name,
               "    int q=i/4, r=i-q*4;",
               "    vec4 v=vec4(0.0);",
               "    if(q<%d) v=%s_0[q];" % (len(chunks[0]), name)]
        for c in range(1, len(chunks)):
            acc.append("    q-=%d; if(q>=0 && q<%d) v=%s_%d[q];"
                       % (len(chunks[c-1]), len(chunks[c]), name, c))
        acc.append("    return r==0?v.x:r==1?v.y:r==2?v.z:v.w;\n}")
        return "\n".join(decls) + "\n" + "\n".join(acc)

    songlen = info['end_tick'] * Q / 44100.0
    data = "\n".join([
        "// ---- movie build: float-encoded MIDI (keys/pedal), GLSL 120 ----",
        "#define iTime iGlobalTime",
        "#define texture texture2D",
        "#define MV_Q %.1f" % float(Q),
        "#define MV_END %.1f" % float(info['end_tick']),
        "#define MV_LOOKBACK %d" % info['lookback'],
        "#define MV_MAXCNT %d" % max(1, maxcnt),
        "#define MV_SONGLEN %.4f" % songlen,
        farr("mvON", ON), farr("mvNG", NG), farr("mvRV", RV), farr("mvSC", SC), ""])

    # ---- keyState/pedalState: per-pixel scan instead of Buffer A ----
    old = """float keyState(int midi){
    if(midi < 21 || midi > 108) return 0.0;
    return texelFetch(iChannel3, ivec2(midi - 21, 0), 0).r;
}
float pedalState(){ return texelFetch(iChannel3, ivec2(88, 0), 0).r; }  // damper (CC64)"""
    new = """float gKeys[89];
float keyState(int midi){
    if(midi < 21 || midi > 108) return 0.0;
    return gKeys[midi - 21];
}
float pedalState(){ return gKeys[88]; }  // damper (CC64)
void initKeys()
{
    for (int i = 0; i < (iTime < -1.0 ? 9999 : 89); i++) gKeys[i] = 0.0;
    float qSec = MV_Q/44100.0;
    float now = clamp(iTime/qSec, 0.0, MV_END);
    float b = floor(now/64.0);
    float ped = 0.0;
    for (int j = 0; j < (iTime < -1.0 ? 9999 : MV_LOOKBACK); j++)
    {
        float bi = b - float(j);
        if (bi < 0.0) break;
        float sc = mvSC(int(bi));
        int start = int(floor(sc/128.0));
        int cnt = int(sc - floor(sc/128.0)*128.0);
        for (int k = 0; k < (iTime < -1.0 ? 9999 : MV_MAXCNT); k++)
        {
            if (k >= cnt) break;
            int i2 = start + k;
            float tt = (now - mvON(i2))*qSec;
            if (tt < 0.0) continue;
            float ng = mvNG(i2);
            float gate = floor(ng/128.0);
            int note = int(ng - gate*128.0);
            gate *= qSec;
            float rv = mvRV(i2);
            float ring = floor(rv/128.0);
            float vel = rv - ring*128.0;
            ring *= qSec;
            float e = smoothstep(0.0, 0.018, tt)*(1.0 - smoothstep(gate, gate + 0.09, tt));
            gKeys[note] = max(gKeys[note], e*(0.75 + 0.25*vel/127.0));
            ped = max(ped, smoothstep(gate, gate + 0.12, tt)*(1.0 - smoothstep(ring - 0.06, ring + 0.03, tt)));
        }
    }
    gKeys[88] = ped;
}"""
    assert code.count(old) == 1
    code = code.replace(old, new, 1)

    # ---- logo: analytic distance instead of the Buffer A bake ----
    m = _re.search(r'// distance \(font units\).*?return min\(dmin, 1\.0\);.*?\n\}', PIANO_BUFB_CODE, _re.S)
    assert m, "logoTextDf not found in PIANO_BUFB_CODE"
    logofn = m.group(0)
    old = "void MapLogo (in vec3 world_pos, out Obj obj)"
    assert code.count(old) == 1
    code = code.replace(old, logofn + "\n\n" + old, 1)
    baked = _re.search(r"\t// the glyph distance field is PREBAKED.*?vec2 w = vec2\(d2\*LOGO_SCALE", code, _re.S)
    assert baked, "baked-fetch block not found"
    code = code[:baked.start()] + "\tfloat d2 = logoTextDf(uvf);\n\tvec2 w = vec2(d2*LOGO_SCALE" + code[baked.end():]

    # ---- GLSL-120isms ----
    code = code.replace("iFrame<0", "iTime < -1.0")                   # de-unroll sentinels
    # cubemap wall -> single-face emulation (GL 4.1 has cubemaps but only one
    # face is cached; proper cube-face UV from the direction avoids the tiling
    # that a flat world_pos.zy lookup produced)
    code = code.replace("texture(iChannel1, (world_pos.zyx * 0.05)).rgb",
                        "texture(iChannel1, mvCubeUV(world_pos.zyx)).rgb")
    # ceiling skydome sample -> same single-face emulation
    code = code.replace("texture(iChannel1, normalize(world_pos - vec3(0.0, 100.0, 0.0)).zyx).rgb",
                        "texture(iChannel1, mvCubeUV(normalize(world_pos - vec3(0.0, 100.0, 0.0)).zyx)).rgb")
    # key env-reflection sample -> same emulation
    code = code.replace("texture(iChannel1, ref_dir).rgb",
                        "texture(iChannel1, mvCubeUV(ref_dir)).rgb")
    _cube = ("vec2 mvCubeUV(vec3 d){\n"
             "\tvec3 a = abs(d); float ma; vec2 uv;\n"
             "\tif(a.x>=a.y && a.x>=a.z){ ma=a.x; uv=vec2(-d.z*sign(d.x), d.y); }\n"
             "\telse if(a.y>=a.z){ ma=a.y; uv=vec2(d.x, -d.z*sign(d.y)); }\n"
             "\telse { ma=a.z; uv=vec2(d.x*sign(d.z), d.y); }\n"
             "\treturn uv/max(ma,1e-5)*0.5 + 0.5;\n}\n")
    code = code.replace("vec3 place_pos = vec3(10.0,0.0,40.0);",
                        _cube + "vec3 place_pos = vec3(10.0,0.0,40.0);", 1)
    # switch -> if chains (getGlyphRect / getGlyphMeta); GLSL 120 has no switch
    code = _re.sub(r"case (\d+): return (vec4\([^;]+\));", r"if(c==\1) return \2;", code)
    code = _re.sub(r"[ \t]*switch\s*\(c\)\s*\{\n", "", code)
    code = _re.sub(r"[ \t]*default: return vec4\(0\.\);\n[ \t]*\}\n", "  return vec4(0.);\n", code)
    # GLSL 120 has no backslash line continuation: drop the makeStr/makeColorStr2D
    # macro machinery and hand-expand col1 (the only user — the nameboard decal)
    a = code.index("#define makeStr(name)")
    b = code.index("const float ATLAS_W")
    code = code[:a] + code[b:]
    old_inv = """// -- Edit text here -------------------------------------------------------
makeStr(line1)       _B _o _e _s _e _n _d _o _r _f _e _r  _end
makeColorStr2D(col1) _B _o _e _s _e _n _d _o _r _f _e _r  _endC2D"""
    assert code.count(old_inv) == 1
    code = code.replace(old_inv, """// -- nameboard text (hand-expanded makeColorStr2D for GLSL 120) -----------
vec4 col1 (vec2 u) {
	if (u.x<0.||u.y<0.||u.y>LINE_H) return vec4(0.);
	const int C[12] = int[12](66,111,101,115,101,110,100,111,114,102,101,114);
	float x = 0.0;
	for (int k = 0; k < (iTime < -1.0 ? 99 : 12); k++) {
		vec4 mt = getGlyphMeta(C[k]);
		if (u.x < x + mt.z) {
			vec4 rc = getGlyphRect(C[k]);
			vec2 b = vec2(mt.x, LINE_H - mt.y - mt.w);
			vec2 l = (u - vec2(x, 0.) - b)/vec2(rc.z*ATLAS_W, mt.w);
			float a = (all(greaterThanEqual(l, vec2(0.))) && all(lessThanEqual(l, vec2(1.))))
				? _atlasAlpha(rc.xy + l*rc.zw) : 0.;
			return vec4(1., 1., 1., a);
		}
		x += mt.z;
	}
	return vec4(0.);
}""", 1)

    # GLSL 120 has no integer % operator
    code = code.replace("d[w%7]", "d[w - (w/7)*7]")
    # cover songLen from uint defines -> baked constant (legacy: the lid now
    # keys off the plain-float SONG_END define, which is GLSL-120 safe as-is)
    old = "float songLen = float(MIDI_END_TICK)*float(MIDI_TIME_Q_SAMPLES)/MIDI_SAMPLE_RATE;"
    if old in code:
        code = code.replace(old, "float songLen = MV_SONGLEN;", 1)
    # per-pixel key scan before anything renders
    old = "\tfloat aspect = iResolution.y / iResolution.x;"
    assert code.count(old) == 1
    code = code.replace(old, "\tinitKeys();\n" + old, 1)

    assert "texelFetch(" not in code
    assert not _re.search(r"\bswitch\s*\(c", code)   # (a font COMMENT says "switch() tables")
    assert "iFrame<" not in code and "iFrame <" not in code   # comments may mention it
    return data + code


PIANO_BUFB_CODE = r"""// mid2glsl --viz 3: Buffer B — the intro logo's glyph distance field.
// Owns the FontyMon Bezier font core; bakes the signed distance once per frame
// (rows >= 4, same mapping Buffer A relays and the Image pass extrudes).
//__FONTCORE__

// distance (font units) to the "Boesendorfer" outline at font-space uv —
// the same per-char cell algebra makeStr uses, but returning DISTANCE
float logoTextDf (vec2 uvf)
{
	const int CH[12] = int[12](66,111,101,115,101,110,100,111,114,102,101,114);
	// cumulative advances: cheap x-reject so each eval runs 1-2 glyph SDFs, not 12
	const float LX0[13] = float[13](0.0,9.492,13.644,17.316,21.696,25.368,
	                                29.556,33.996,38.148,42.18,46.572,50.244,54.276);
	float dmin = 1e9;
	for (int i = 0; i < (iFrame<0?99:12); i++) {
		if (uvf.x < LX0[i] - 2.5 || uvf.x > LX0[i+1] + 2.5) continue;   // can't win the min
		vec4 mt = getGlyphMeta(CH[i]);
		vec4 rc = getGlyphRect(CH[i]);          // also selects the glyph for cSDF
		vec2 loc = vec2(uvf.x - LX0[i] - mt.x + rc.x,
		                uvf.y - (LINE_H - mt.y - mt.w) + rc.y);
		dmin = min(dmin, cSDF(loc));
	}
	return min(dmin, 1.0);   // SAFE bound: reject-all margin zones must not overestimate
	// (1e9/large values there let edge-on rays warp past the letters — proven by simulation)
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    if (fragCoord.y >= 4.0 && iTime < 34.5) {
        vec2 lm = vec2(fragCoord.x/iResolution.x, (fragCoord.y - 4.0)/(iResolution.y - 4.0));
        fragColor = vec4(logoTextDf(vec2(-2.0, -1.5) + lm*vec2(58.5, 14.0)), 0.0, 0.0, 1.0);
    } else fragColor = vec4(0.0);
}"""

# ============================================================ ShaderToy JSON
# One-click ShaderToy ▸ Import descriptor, same VERIFIED schema as MOD2GLSL's
# emit_json: bare {ver,renderpass,flags,info}, inputs use type/filepath, buffer
# linkage by magic ids (Buffer A=4dXGR8, Buffer B=XsXGR8, Image out=4dfGRr).
# A sacrificial Buffer B is placed FIRST: ShaderToy's new-shader import resets
# the first tab's code, so Buffer B absorbs it (delete the tab after import;
# Cmd+Z restores Image if the delete pasted over it).
_STJ_SMP_BUF = {"filter": "linear", "wrap": "clamp",
                "vflip": "true", "srgb": "false", "internal": "byte"}
_STJ_SACRIFICE = {
    "outputs": [{"channel": 0, "id": "XsXGR8"}],
    "inputs":  [{"channel": 3, "id": "XsXGR8",
                 "filepath": "/media/previz/buffer01.png",
                 "type": "buffer", "sampler": _STJ_SMP_BUF}],
    "code": ("// *** DELETE THIS TAB after import ***\n"
             "// After clicking ×, press Cmd+Z (Mac) / Ctrl+Z (Win) immediately!\n"
             "// ShaderToy pastes this tab's code into Image on delete —\n"
             "// Cmd+Z undoes that and restores Image's real code.\n"
             "void mainImage(out vec4 o,in vec2 u){o=vec4(0);}"),
    "name": "Buffer B", "description": "", "type": "buffer"}
# --viz 3 (piano): the user's own Image-tab texture setup (fctGD8 export) —
# wood floor (ch0), forest cubemap environment (ch1), soundboard wood (ch2);
# Buffer A (key/pedal states) rides on ch3.
_STJ_PIANO_TEX = [
    {"channel": 0, "id": "4dXGRn", "type": "texture",
     "filepath": "/media/a/10eb4fe0ac8a7dc348a2cc282ca5df1759ab8bf680117e4047728100969e7b43.jpg",
     "sampler": {"filter": "mipmap", "wrap": "repeat", "vflip": "true",
                 "srgb": "false", "internal": "byte"}},
    {"channel": 1, "id": "XsX3zn", "type": "cubemap",
     "filepath": "/media/a/94284d43be78f00eb6b298e6d78656a1b34e2b91b34940d02f1ca8b22310e8a0.png",
     "sampler": {"filter": "mipmap", "wrap": "clamp", "vflip": "false",
                 "srgb": "false", "internal": "byte"}},
    {"channel": 2, "id": "XsfGRn", "type": "texture",
     "filepath": "/media/a/1f7dca9c22f324751f2a5a59c9b181dfe3b5564a04b724c657732d0bf09c99db.jpg",
     "sampler": {"filter": "mipmap", "wrap": "repeat", "vflip": "true",
                 "srgb": "false", "internal": "byte"}},
]

_STJ_IMAGE_STUB = ("// mid2glsl: sound-only build — the music lives in the Sound tab.\n"
                   "// Rebuild with --viz 1 (LED tunnel) or --viz 2 (Soundform CD) for visuals.\n"
                   "void mainImage(out vec4 o, in vec2 u){ o = vec4(vec3(0.02), 1.0); }")

def shadertoy_json(name, sound, common=None, buffera=None, image=None, keyboard=False,
                   img_buf_channel=0, extra_image_inputs=None, bufferb=None):
    """Import JSON string. With buffera/common/image → 5 passes (sacrificial
    Buffer B, Image, Common, Buffer A, Sound); else 3 (Buffer B, Image stub,
    Sound). img_buf_channel: which Image iChannel receives Buffer A (viz 3
    uses 3, leaving 0/1/2 free for the piano's optional photo textures)."""
    import json
    buf_in = {"channel": 0, "id": "4dXGR8",
              "filepath": "/media/previz/buffer00.png",
              "type": "buffer", "sampler": _STJ_SMP_BUF}
    bufb_in = {"channel": 1, "id": "XsXGR8",
               "filepath": "/media/previz/buffer01.png",
               "type": "buffer", "sampler": _STJ_SMP_BUF}
    if bufferb is not None:
        # the REAL Buffer B is in use -> the sacrificial import tab moves to Buffer C
        sac = dict(_STJ_SACRIFICE)
        sac["name"] = "Buffer C"
        sac["outputs"] = [{"channel": 0, "id": "4sXGR8"}]
        sac["inputs"] = [{"channel": 3, "id": "4sXGR8",
                          "filepath": "/media/previz/buffer02.png",
                          "type": "buffer", "sampler": _STJ_SMP_BUF}]
        rp = [sac]
    else:
        rp = [_STJ_SACRIFICE]
    if buffera is not None:
        img_in = [dict(buf_in, channel=img_buf_channel)] + list(extra_image_inputs or [])
        if keyboard:
            img_in.append({"channel": 3, "id": "4dXGRr",
                           "filepath": "/presets/tex00.jpg",
                           "type": "keyboard", "sampler": _STJ_SMP_BUF})
        rp += [
            {"outputs": [{"channel": 0, "id": "4dfGRr"}], "inputs": img_in,
             "code": image, "name": "Image", "description": "", "type": "image"},
            {"outputs": [], "inputs": [],
             "code": common, "name": "Common", "description": "", "type": "common"},
            {"outputs": [{"channel": 0, "id": "4dXGR8"}],
             "inputs": [dict(buf_in)] + ([dict(bufb_in)] if bufferb is not None else []),
             "code": buffera, "name": "Buffer A", "description": "", "type": "buffer"},
        ]
        if bufferb is not None:
            rp.append({"outputs": [{"channel": 0, "id": "XsXGR8"}], "inputs": [],
                       "code": bufferb, "name": "Buffer B", "description": "", "type": "buffer"})
        rp.append({"outputs": [], "inputs": [],
                   "code": sound, "name": "Sound", "description": "", "type": "sound"})
    else:
        rp += [
            {"outputs": [{"channel": 0, "id": "4dfGRr"}], "inputs": [],
             "code": _STJ_IMAGE_STUB, "name": "Image", "description": "", "type": "image"},
            {"outputs": [], "inputs": [],
             "code": sound, "name": "Sound", "description": "", "type": "sound"},
        ]
    return _stj_finish(name, rp, keyboard)

def _stj_finish(name, rp, keyboard):
    """Shared tail: audit every pass exactly as ShaderToy compiles it (Common
    is prepended to each tab, so its const data counts every time), then wrap
    the renderpass list in the export-json envelope."""
    import json
    _common = next((r["code"] for r in rp if r.get("type") == "common"), "")
    for _r in rp:
        if _r.get("type") != "common":
            audit_glsl_pass(_common + _r["code"], f"{_r['name']} pass")
    shader = {
        "ver": "0.1",
        "renderpass": rp,
        "flags": {"mFlagVR": False, "mFlagWebcam": False,
                  "mFlagSoundInput": False, "mFlagSoundOutput": True,
                  "mFlagKeyboard": bool(keyboard), "mFlagMultipass": True,
                  "mFlagMusicStream": False},
        "info": {"id": "-1", "date": "0", "viewed": 0, "name": name[:64],
                 "username": "", "description": name[:64], "likes": 0,
                 "published": 0, "flags": 0, "usePreview": 0,
                 "tags": ["sound", "music", "midi"], "hasliked": 0,
                 "parentid": "", "parentname": ""},
    }
    return json.dumps(shader, separators=(',', ':'))

def shadertoy_json_island(name, sound, common, island_path):
    """--viz 5: the RtMI 'Monkey Island' 3D flythrough (mk_island3d.py output)
    as the visual. Its three passes — noise Buffer A ('4dXGR8', keyboard+self),
    scene Buffer B ('XsXGR8', brick/wood textures + noise + self camera state)
    and the Image post — are spliced in VERBATIM, so mk_island3d.py stays the
    single source of truth (camera tour, DEUN de-unroll sentinels, textures).
    mid2glsl adds Common (melody data) + Sound (synth); the sacrificial import
    tab moves to Buffer C on the unused '4sXGR8' id."""
    import json
    isl = json.load(open(island_path))
    ipass = {r["name"]: r for r in isl["renderpass"]}
    for need in ("Buffer A", "Buffer B", "Image"):
        if need not in ipass:
            raise SystemExit(f"!! {island_path}: no '{need}' pass — regenerate with mk_island3d.py")
    sac = dict(_STJ_SACRIFICE)
    sac["name"] = "Buffer C"
    sac["outputs"] = [{"channel": 0, "id": "4sXGR8"}]
    sac["inputs"] = [{"channel": 3, "id": "4sXGR8",
                      "filepath": "/media/previz/buffer02.png",
                      "type": "buffer", "sampler": _STJ_SMP_BUF}]
    rp = [sac,
          ipass["Image"],
          {"outputs": [], "inputs": [], "code": common,
           "name": "Common", "description": "", "type": "common"},
          ipass["Buffer A"], ipass["Buffer B"],
          {"outputs": [], "inputs": [], "code": sound,
           "name": "Sound", "description": "", "type": "sound"}]
    return _stj_finish(name, rp, keyboard=True)   # island camera uses Shift

def override_reverb_taps(code, taps):
    """Force REVERB_TAPS to `taps` (CLI --reverb N): drop any instrument-defs
    value, then prepend the override so the template #ifndef picks it up."""
    if taps is None:
        return code
    import re as _re
    code = _re.sub(r"#define REVERB_TAPS \d+\n", "", code)
    return f"#define REVERB_TAPS {taps}\n" + code

def build_shader(tpq, tempo, tracks, bpm=None, mode="grains", pad_sec=1.2, tempo_map=None,
                 trim_s=None, budget_vec4=None):
    if mode == "buffer": return build_buffer(tpq, tempo, tracks, bpm, pad_sec=pad_sec,
                                             tempo_map=tempo_map, trim_s=trim_s,
                                             budget_vec4=budget_vec4)[0]
    if mode == "player": return build_player(tpq, tempo, tracks, bpm, pad_sec=pad_sec,
                                             tempo_map=tempo_map, trim_s=trim_s,
                                             budget_vec4=budget_vec4)
    return (build_grains if mode == "grains" else build_macro)(tpq, tempo, tracks, bpm)

# ------------------------------------------------------------ minimal writer
def write_midi(path, tracks, tpq=480, tempo=500000):
    def vlq(n):
        out = bytearray([n & 0x7f]); n >>= 7
        while n: out.insert(0, (n & 0x7f) | 0x80); n >>= 7
        return bytes(out)
    chunks = b""
    for ti, notes in enumerate(tracks):
        ev = []
        if ti == 0: ev.append((0, b"\xff\x51\x03" + struct.pack(">I", tempo)[1:]))
        for s, dur, n, v in notes:
            ev.append((s, bytes([0x90, n, v]))); ev.append((s+dur, bytes([0x80, n, 0])))
        ev.sort(key=lambda e: e[0])
        data, last = b"", 0
        for tick, msg in ev: data += vlq(tick-last) + msg; last = tick
        data += vlq(0) + b"\xff\x2f\x00"
        chunks += b"MTrk" + struct.pack(">I", len(data)) + data
    hdr = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), tpq)
    open(path, "wb").write(hdr + chunks)

if __name__ == "__main__":
    a = sys.argv
    if len(a) < 2:
        print("usage: python mid2glsl.py in.mid [out.glsl] "
              "[--bpm N] [--mode grains|macro|buffer|player] [--inst brass|...|piano6 (default: piano6)] "
              "[--no-reverb] [--reverb N] [--viz [1|2|3|4|5|led|cd|piano|melee|island]] [--no-viz] [--gm] [--time N]  "
              "(piano viz is the DEFAULT; --reverb N sets REVERB_TAPS, e.g. 32 = denser room, default 16; "
              "--gm = multi-timbral General-MIDI engine, per-channel program -> voice, organ family voiced; "
              "--time N = playback/data limit seconds, default 180)")
        sys.exit(1)
    inp  = a[1]
    out  = a[2] if len(a) > 2 and not a[2].startswith("--") else inp.rsplit(".", 1)[0] + ".glsl"
    bpm  = float(a[a.index("--bpm")+1])  if "--bpm"  in a else None
    mode = a[a.index("--mode")+1]        if "--mode" in a else "player"
    reverb = "--no-reverb" not in a              # reverb ON by default; --no-reverb disables
    reverb_taps = int(a[a.index("--reverb")+1]) if "--reverb" in a else None   # e.g. --reverb 32
    end_s = (float(a[a.index("--time")+1]) if "--time" in a else
             float(a[a.index("--end")+1]) if "--end" in a else 180.0)  # playback limit: data trim + lid close
    # the grand-piano build IS the default: a bare `mid2glsl.py song.mid` equals
    # `--mode player --viz 3 --inst piano6`; --no-viz = the sound-only single file
    viz = "piano" if (mode == "player" and "--no-viz" not in a) else None
    if "--viz" in a:
        viz = "piano"                            # bare --viz = the grand piano
        _vi = a.index("--viz")
        if _vi + 1 < len(a):
            viz = {"1": "led", "2": "cd", "3": "piano", "4": "melee", "5": "island",
                   "island": "island",
                   "led": "led", "cd": "cd", "piano": "piano", "melee": "melee"}.get(a[_vi+1].lower(), "piano")
    inst = a[a.index("--inst")+1].lower() if "--inst" in a else "piano6"   # the Bösendorfer is the default
    if inst not in ("brass", "piano", "epiano", "organ", "piano2", "epiano2", "piano3", "piano4", "piano5", "piano6", "piano7"):
        print(f"unknown --inst {inst} (brass|piano|...|piano7) — using brass"); inst = "brass"
    if inst == "piano5":
        try:
            from piano5_pcm import PIANO5_PCM_GLSL
        except ImportError:
            print("--inst piano5 needs piano5_pcm.py (the baked Salamander PCM) — using piano4"); inst = "piano4"
    if inst == "piano6":
        try:
            from piano6_pcm import PIANO6_PCM_GLSL
        except ImportError:
            print("--inst piano6 needs piano6_pcm.py (the baked Bösendorfer PCM) — using piano4"); inst = "piano4"
    if inst == "piano7":
        try:
            from piano7_pcm import PIANO7_PCM_GLSL
        except ImportError:
            print("--inst piano7 needs piano7_pcm.py (the seam-calibrated Bösendorfer PCM) — using piano6"); inst = "piano6"
    tpq, tempo, tracks = parse_midi(inp)
    tempo_map = getattr(parse_midi, "tempo_map", None)   # full 0x51 map (rubato files)
    nn = sum(len(t) for t in tracks)
    name = inp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    title = a[a.index("--title")+1] if "--title" in a else name.replace("_", " ")
    _hdr = TAB_HEADER.replace("__TRACK__", title)   # per-build track name in the credits

    if "--gm" in a:
        # multi-timbral GM engine: per-channel program -> voice recipe. Isolated
        # from the mono-timbral PLAYER_CODE + piano viz (a standalone sound shader)
        FAM = ["Piano","ChromPerc","Organ","Guitar","Bass","Strings","Ensemble","Brass",
               "Reed","Pipe","SynthLead","SynthPad","SynthFX","Ethnic","Percussive","SFX"]
        GMN = ["Ac.Grand","Br.Grand","El.Grand","Honky","EP1","EP2","Harpsi","Clav",
               "Celesta","Glock","MusicBox","Vibra","Marimba","Xylo","TubBell","Dulcimer",
               "DrawOrgan","PercOrgan","RockOrgan","ChurchOrg","ReedOrgan","Accordion",
               "Harmonica","TangoAcc"]
        # Split the GM build so a --viz can share the MIDI note data via Common:
        #   Common = data (notes + midiOM/midiDur/midiPG) — build_buffer(gm=True), identical to build_gm
        #   Sound  = GM_PLAYER_CODE (the multi-timbral synth)
        #   Buffer A / Image = the LED/CD visualizer, its midiSpec fed from Common (no FFT tab on ShaderToy)
        gm_viz = viz if viz in ("led", "cd", "melee", "island") else None  # --gm --viz 1/2/4/5 → add the visualizer
        # cd viz re-synthesizes with the piano6 voice: its PCM blob rides in
        # Buffer A, so the melody budget must be pre-shrunk by its reserve
        data, notes, info = build_buffer(tpq, tempo, tracks, bpm, pad_sec=1.2,
                                         tempo_map=tempo_map, trim_s=end_s, gm=True,
                                         budget_vec4=VEC4_DATA_BUDGET
                                         - (pcm_reserve_vec4("piano6") if gm_viz == "cd" else 0))
        gm_sound = GM_PLAYER_CODE
        kit = a[a.index("--kit")+1].lower() if "--kit" in a else "clean"   # clean = SC-55-matched acoustic (default) | dirty
        if kit == "dirty":
            gm_sound = "#define GM_KIT_DIRTY 1\n" + gm_sound

        prn = [("128:Drums" if p == 128 else f"{p}:{GMN[p]}" if p < len(GMN) else str(p))
               for p in info['progs']]
        drums = 128 in info['progs']
        fams  = sorted(set(FAM[p>>3] for p in info['progs'] if p < 128))
        base  = out.rsplit(".", 1)[0]

        if gm_viz == "island":
            # --gm --viz 5: Monkey Island flythrough + the multi-timbral GM engine
            isl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) or ".",
                                    "rtmi_flythrough.json")
            if not os.path.exists(isl_path):
                raise SystemExit(f"!! --viz island needs {isl_path} — run mk_island3d.py first")
            common_t, sound_t = _hdr + data, _hdr + gm_sound
            stj = f"{base}_shadertoy.json"
            _stj_str = shadertoy_json_island(name, sound_t, common_t, isl_path)  # strict audit inside
            import json as _json
            _ip = {r["name"]: r for r in _json.loads(_stj_str)["renderpass"]}
            for p, s in [(f"{base}_common.glsl",  common_t), (f"{base}_sound.glsl", sound_t),
                         (f"{base}_bufferA.glsl", _ip["Buffer A"]["code"]),
                         (f"{base}_bufferB.glsl", _ip["Buffer B"]["code"]),
                         (f"{base}_image.glsl",   _ip["Image"]["code"])]:
                open(p, "w").write(s)
            open(stj, "w").write(_stj_str)
            html_out = None
            try:
                from st_player import emit_piano_player
                _here = os.path.dirname(os.path.abspath(__file__)) or "."
                html_out = emit_piano_player(
                    _json.loads(_stj_str), f"{base}.html", title=title,
                    song_dur_s=info['end_tick']*info['Q']/44100.0,
                    tex_dirs=[os.path.dirname(os.path.abspath(inp)) or ".", _here,
                              "~/shadertoy-browser/output/media/a",
                              "~/Downloads/shadertoy-render-master"],
                    fx_mode=2, reverb_taps=(reverb_taps or 32), render_cap=1152)
            except Exception as _e:
                print(f"  !! html player skipped: {_e}")
            print(f"{inp}: {len(tracks)} tracks, {nn} notes -> 5 files + json (GM multi-timbral + viz:island)")
            print(f"  programs used: {prn}")
            if html_out:
                print(f"  {html_out}   <- SINGLE-PAGE PLAYER (island + engine + textures embedded; red ● records mp4)")
            print(f"  {stj}   <- ShaderToy > Import (all tabs; then DELETE the sacrificial Buffer C tab)")
            print(f"    (Shift+drag steers the camera, idle 8s resumes the tour)")
            sys.exit(0)
        if gm_viz:
            note_max = max(n for _, _, n, _, _ in notes)
            img_ch = 0                                  # which Image channel receives Buffer A
            if gm_viz == "cd":                          # Soundform CD: waveform ring re-synth (piano6 stand-in)
                buffera = _pcm_blob("piano6") + INST_DEFS["piano6"] + inject_voices(
                              build_viz_cd_buffera(info['note_min'], note_max, reverb), "piano6")
                image   = CD_IMAGE_CODE
            elif gm_viz == "melee":                     # Melee Island scene; stars twinkle to the spectrum
                buffera = build_viz_buffera(info['note_min'], note_max)   # reuse the LED spectrum as the "music" tex
                image   = VIZ_MELEE_IMAGE_CODE
                img_ch  = 3                             # Buffer A -> Image iChannel3 (= the shader's music input)
            else:                                       # LED Band Spectro 3D: midiSpec, synth-agnostic
                buffera = build_viz_buffera(info['note_min'], note_max)
                image   = VIZ_IMAGE_CODE
            common_t, sound_t   = _hdr + data,    _hdr + gm_sound
            buffera_t, image_t  = _hdr + buffera, _hdr + image
            stj = f"{base}_shadertoy.json"
            _stj_str = shadertoy_json(name, sound_t, common=common_t,
                                      buffera=buffera_t, image=image_t,
                                      img_buf_channel=img_ch,
                                      extra_image_inputs=(_STJ_MELEE_TEX if gm_viz == "melee" else None),
                                      keyboard=(gm_viz == "cd"))   # strict audit runs in here,
            for p, s in [(f"{base}_common.glsl", common_t), (f"{base}_sound.glsl", sound_t),
                         (f"{base}_bufferA.glsl", buffera_t), (f"{base}_image.glsl", image_t)]:
                open(p, "w").write(s)                              # ...BEFORE anything hits disk
            open(stj, "w").write(_stj_str)
            open(out, "w").write(_hdr + data + gm_sound)   # combined single-file (reference only)
            print(f"{inp}: {len(tracks)} tracks, {nn} notes -> {stj} (GM multi-timbral + viz:{gm_viz})")
            print(f"  programs used: {prn}")
            print(f"  families voiced: {', '.join(fams)}{' + drums' if drums else ''}")
            print(f"  {stj}   <- ShaderToy > Import (Common/Sound/Buffer A/Image; then DELETE the sacrificial Buffer B tab)")
            sys.exit(0)

        # (no --viz) original standalone GM sound shader
        glsl = _hdr + data + gm_sound
        stj = out.rsplit(".", 1)[0] + "_shadertoy.json"
        _stj_str = shadertoy_json(name, glsl)     # strict audit before disk writes
        open(out, "w").write(glsl)
        open(stj, "w").write(_stj_str)
        print(f"{inp}: {len(tracks)} tracks, {nn} notes -> {out} + {stj} (GM multi-timbral)")
        print(f"  programs used: {prn}")
        print(f"  families voiced: {', '.join(fams)}{' + drums' if drums else ''}  (all 16 GM families mapped)")
        sys.exit(0)

    if viz and mode != "player":
        print(f"--viz needs --mode player (got {mode}) — ignoring --viz")
        viz = None
    if inst != "brass" and mode not in ("player", "buffer") and not viz:
        print(f"--inst {inst} needs --mode player (got {mode}) — using brass")
        inst = "brass"
    if viz:
        # 4-tab ShaderToy build: data → Common (shared by Sound AND Buffer A),
        # synth → Sound, visualizer → Buffer A (self-fed), display/render → Image.
        #   1/led: LED Band Spectro 3D tunnel   2/cd: Soundform CD (raymarched disc)
        _pad = 3.0 if inst == "piano6" else 1.2
        data, notes, info = build_buffer(tpq, tempo, tracks, bpm, pad_sec=_pad,
                                         tempo_map=tempo_map, trim_s=end_s,
                                         budget_vec4=VEC4_DATA_BUDGET - pcm_reserve_vec4(inst))
        _pc = inject_voices(PLAYER_CODE, inst)   # only the selected voice ships
        sound = _pc if reverb else "#define USE_REVERB 0\n" + _pc
        if viz == "island":
            # ---- --viz 5: RtMI "Monkey Island" 3D flythrough as the visual —
            # mk_island3d.py's json spliced in verbatim, this MIDI as soundtrack
            if inst != "brass":
                sound = INST_DEFS[inst] + sound
            sound = override_reverb_taps(sound, reverb_taps)
            if inst in ("piano5", "piano6", "piano7"):    # baked PCM lives in Common
                data = data + _pcm_blob(inst)
            data, sound = _hdr + data, _hdr + sound
            isl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) or ".",
                                    "rtmi_flythrough.json")
            if not os.path.exists(isl_path):
                raise SystemExit(f"!! --viz 5 needs {isl_path} — run mk_island3d.py first")
            base = out.rsplit(".", 1)[0]
            stj = f"{base}_shadertoy.json"
            _stj_str = shadertoy_json_island(name, sound, data, isl_path)   # strict audit inside
            import json as _json
            _ip = {r["name"]: r for r in _json.loads(_stj_str)["renderpass"]}
            files = [(f"{base}_common.glsl",  data),
                     (f"{base}_sound.glsl",   sound),
                     (f"{base}_bufferA.glsl", _ip["Buffer A"]["code"]),   # noise/key LUT
                     (f"{base}_bufferB.glsl", _ip["Buffer B"]["code"]),   # island scene
                     (f"{base}_image.glsl",   _ip["Image"]["code"])]      # post
            for p, s in files:                    # audit passed -> now touch disk
                open(p, "w").write(s)
            open(stj, "w").write(_stj_str)
            html_out = None
            try:
                from st_player import emit_piano_player
                _here = os.path.dirname(os.path.abspath(__file__)) or "."
                html_out = emit_piano_player(
                    _json.loads(_stj_str), f"{base}.html", title=title,
                    song_dur_s=info['end_tick']*info['Q']/44100.0,
                    tex_dirs=[os.path.dirname(os.path.abspath(inp)) or ".", _here,
                              "~/shadertoy-browser/output/media/a",
                              "~/Downloads/shadertoy-render-master"],
                    fx_mode=2, reverb_taps=(reverb_taps or 32), render_cap=1152)
            except Exception as _e:
                print(f"  !! html player skipped: {_e}")
            print(f"{inp}: {len(tracks)} tracks, {nn} notes -> {len(files)} files + json "
                  f"(player+viz:island, {inst}{', +reverb' if reverb else ', no-reverb'})")
            if html_out:
                print(f"  {html_out}   <- SINGLE-PAGE PLAYER (island + engine + textures embedded; red ● records mp4)")
            print(f"  {stj}   <- ShaderToy > Import (all tabs; then DELETE the sacrificial Buffer C tab)")
            print(f"    (Shift+drag steers the camera, idle 8s resumes the tour)")
            sys.exit(0)
        note_max = max(n for _, _, n, _, _ in notes)
        if viz == "cd":
            buffera = inject_voices(build_viz_cd_buffera(info['note_min'], note_max, reverb), inst)
            image   = CD_IMAGE_CODE
        elif viz == "melee":
            buffera = build_viz_buffera(info['note_min'], note_max)   # spectrum → the shader's iChannel3 "music"
            image   = VIZ_MELEE_IMAGE_CODE
        elif viz == "piano":
            buffera = PIANO_BUFA_CODE
            bufferb = PIANO_BUFB_CODE.replace("//__FONTCORE__", PIANO_FONT_CORE)
            image   = (f"#define SONG_END {end_s:.1f}\n"
                       + PIANO_IMAGE_CODE.replace("//__FONTCORE__", PIANO_FONT_CORE))
        else:
            buffera = build_viz_buffera(info['note_min'], note_max)
            image   = VIZ_IMAGE_CODE
        if inst != "brass":                      # voice defines in Sound AND viz buffer
            sound   = INST_DEFS[inst] + sound
            buffera = INST_DEFS[inst] + buffera
        sound   = override_reverb_taps(sound, reverb_taps)
        buffera = override_reverb_taps(buffera, reverb_taps)
        if inst in ("piano5", "piano6", "piano7"):   # baked PCM lives in Common
            data = data + _pcm_blob(inst)            # resharded to <=1024-item arrays
        data    = _hdr + data
        sound   = _hdr + sound
        buffera = _hdr + buffera
        image   = _hdr + image
        if viz == "piano":
            bufferb = _hdr + bufferb
        base = out.rsplit(".", 1)[0]
        files = [(f"{base}_common.glsl",  data),
                 (f"{base}_sound.glsl",   sound),
                 (f"{base}_bufferA.glsl", buffera),
                 (f"{base}_image.glsl",   image)]
        if viz == "piano":
            files.append((f"{base}_bufferB.glsl", bufferb))
        stj = f"{base}_shadertoy.json"
        _stj_str = shadertoy_json(name, sound, common=data,
                                  buffera=buffera, image=image,
                                  keyboard=(viz == "cd"),
                                  img_buf_channel=(3 if viz in ("piano", "melee") else 0),
                                  extra_image_inputs=(_STJ_PIANO_TEX if viz == "piano"
                                                      else _STJ_MELEE_TEX if viz == "melee" else None),
                                  bufferb=(bufferb if viz == "piano" else None))
        for p, s in files:                # strict audit passed -> now touch disk
            open(p, "w").write(s)
        if viz == "piano":
            movie = build_movie_glsl(image, notes, info)
            audit_glsl_pass(_hdr + movie, "movie glsl", strict=False)   # offline path: warn only
            open(f"{base}_movie.glsl", "w").write(_hdr + movie)
            files.append((f"{base}_movie.glsl", None))
        open(stj, "w").write(_stj_str)
        html_out = None
        if viz == "piano":
            # single-page player: shader + the ShaderToy engine + textures in
            # ONE self-contained html (MOD2GLSL's emit_player machinery)
            try:
                import json as _json, os as _os
                from st_player import emit_piano_player
                _here = _os.path.dirname(_os.path.abspath(__file__)) or "."
                html_out = emit_piano_player(
                    _json.loads(_stj_str), f"{base}.html", title=title,
                    song_dur_s=info['end_tick']*info['Q']/44100.0,
                    tex_dirs=[_os.path.dirname(_os.path.abspath(inp)) or ".", _here,
                              "~/shadertoy-browser/output/media/a",
                              "~/Downloads/shadertoy-render-master"],
                    fx_mode=2,   # soundfield (AuralSpace) on load; A key cycles
                    reverb_taps=(reverb_taps or 32))  # html room denser than the 16-tap default
                files.append((html_out, None))
            except Exception as _e:
                print(f"  !! html player skipped: {_e}")
        print(f"{inp}: {len(tracks)} tracks, {nn} notes -> {len(files)} files + json "
              f"(player+viz:{viz}, {inst}{', +reverb' if reverb else ', no-reverb'})")
        if html_out:
            print(f"  {html_out}   <- SINGLE-PAGE PLAYER (engine + textures embedded; just open it)")
        print(f"  {stj}   <- ShaderToy > Import (all tabs; then DELETE the sacrificial "
              f"{'Buffer C' if viz == 'piano' else 'Buffer B'} tab)")
        print(f"  or paste by hand:")
        print(f"    Common    <- {files[0][0]}")
        print(f"    Sound     <- {files[1][0]}")
        print(f"    Buffer A  <- {files[2][0]}   (set Buffer A's iChannel0 = Buffer A"
              + ("; iChannel1 = Buffer B)" if viz == "piano" else ")"))
        if viz == "piano":
            print(f"    Buffer B  <- {files[4][0]}   (the logo font bake; no inputs)")
        print(f"    Image     <- {files[3][0]}   (set Image's iChannel"
              f"{3 if viz == 'piano' else 0} = Buffer A)")
        if viz == "piano":
            print(f"    (the json binds the wood/cubemap/soundboard textures on iChannel0/1/2")
            print(f"     automatically — hand-pasters must set those three + iChannel3 = Buffer A;")
            print(f"     json import: DELETE the sacrificial Buffer C tab, not B!)")
        if viz == "cd":
            print(f"    Image iChannel3 = Keyboard (optional: keys 1/2 toggle the wave/FFT rings;")
            print(f"    drag mouse to rotate the disc)")
        sys.exit(0)
    glsl = build_shader(tpq, tempo, tracks, bpm, mode,
                        pad_sec=(3.0 if inst == "piano6" else 1.2), tempo_map=tempo_map,
                        trim_s=end_s,
                        budget_vec4=VEC4_DATA_BUDGET - (pcm_reserve_vec4(inst)
                                    if mode in ("player", "buffer") else 0))
    if mode in ("player", "buffer"):
        glsl = inject_voices(glsl, inst)
    if not reverb:                               # --no-reverb forces the comb reverb off
        glsl = "#define USE_REVERB 0\n" + glsl
    if inst != "brass" and mode in ("player", "buffer"):
        glsl = INST_DEFS[inst] + glsl
    glsl = override_reverb_taps(glsl, reverb_taps)
    if inst in ("piano5", "piano6", "piano7") and mode in ("player", "buffer"):
        glsl = _pcm_blob(inst) + glsl            # resharded to <=1024-item arrays
    glsl = _hdr + glsl
    audit_glsl_pass(glsl, "combined sound glsl")
    open(out, "w").write(glsl)
    extra = ""
    if mode in ("player", "grains", "macro"):    # buffer mode = data-only, not importable
        stj = out.rsplit(".", 1)[0] + "_shadertoy.json"
        open(stj, "w").write(shadertoy_json(name, glsl))
        extra = f" + {stj}"
    print(f"{inp}: {len(tracks)} tracks, {nn} notes -> {out}{extra} "
          f"({mode}, {inst}{', +reverb' if reverb else ', no-reverb'})")

# ============================================================ BAND (multi-voice)
BAND_PLAYER = r"""
// ---- multi-voice band player -------------------------------------
#define TAU 6.28318530718
float NoteToHz(int n){ return 440.0*exp2((float(n)-69.0)/12.0); }
uint  onTick(uint i){ return midiOM(i)&0xFFFFu; }
uint  meta16(uint i){ return midiOM(i)>>16; }
uint  durTick(uint i){ uint w=midiDur(i>>1u); return ((i&1u)==0u)?(w&0xFFFFu):(w>>16); }
int   metaNote(uint m){ return MIDI_NOTE_MIN+int(m&0x7Fu); }
float metaVel(uint m){ return float((m>>7)&63u)/63.0; }
uint  metaInstr(uint m){ return (m>>13)&7u; }

float nse(float x){ return fract(sin(x*45.233)*43758.5453)*2.0-1.0; }
float fbsin(float p,float fb){ float y=sin(p); y=sin(p+fb*y); y=sin(p+fb*y); return y; }

// ── master soft-knee limiter (ported 1:1 from MOD2GLSL's mod_player.py) ──
// T=0.85 knee: bit-perfect below it; the summed voices approach 1.0 smoothly
// instead of saturating from the first sample like the old always-on tanh.
vec2 softLimit(vec2 x){
    const float T=0.85, HEAD=1.0-T;
    vec2 ax=abs(x), over=max(ax-T,vec2(0.0));
    return sign(x)*(min(ax,vec2(T)) + HEAD*over/(over+HEAD));
}

float vLead(float f,float a,float vel){
    float ph=TAU*f*a;
    float I=(1.0+2.4*vel)*exp2(-a*3.5)+0.6;
    float m=I*fbsin(ph,0.5);
    float dec=0.5+0.5*exp2(-a*0.9);
    return (sin(ph+m)+sin(ph*1.004+m))*0.5*dec;
}
float vBass(float f,float a,float vel){
    float cut=3.0+11.0*exp2(-a*6.0);            // envelope-swept harmonic cutoff
    float s=0.0;
    for(int h=1;h<=10;h++){ float hh=float(h); if(f*hh>15000.0) break;
        s += (1.0/hh)*exp(-0.07*max(hh-cut,0.0))*sin(TAU*f*hh*a); }
    return tanh(s*1.6)*0.6 + sin(TAU*f*a)*0.6;  // + clean sub
}
float vKick(float a){
    float ph=TAU*(45.0*a + 4.328*(1.0-exp2(-30.0*a)));   // integrated pitch sweep
    float body=sin(ph)*exp2(-a*7.0);
    float click=nse(floor(a*45000.0))*exp2(-a*320.0)*0.5;
    return tanh((body+click)*1.7);
}
float vSnare(float a){
    float tone=sin(TAU*185.0*a)*exp2(-a*26.0)*0.4;
    float noise=nse(floor(a*44100.0))*exp2(-a*22.0);
    return tanh((tone+noise)*1.3)*0.85;
}
float vHat(float a){
    float n=nse(floor(a*90000.0)) - nse(floor(a*90000.0)-1.0);  // crude HP
    return n*exp2(-a*55.0)*0.6;
}

vec2 mainSound(int samp, float timeNow){
    float SR=MIDI_SAMPLE_RATE, qSec=float(MIDI_TIME_Q_SAMPLES)/SR;
    float tPhase=float(samp)/SR;
    float songLen=float(MIDI_END_TICK)*qSec;
    float tEvent=mod(tPhase, songLen);
    uint T=uint(floor(tEvent/qSec+0.5)); T=min(T,MIDI_END_TICK-1u);
    uint b=T>>MIDI_BLOCK_SHIFT_TICKS; if(b>=MIDI_BLOCK_COUNT) return vec2(0.0);
    uint padT=uint(1.0/qSec+0.5);
    float L=0.0,R=0.0;
    for(uint j=0u;j<(samp<0 ? 99999u : MIDI_LOOKBACK_BLOCKS);j++){  // de-unroll (see brass template): runtime-opaque bound stops ANGLE unrolling the note scan MIDI_LOOKBACK_BLOCKS×
        if(b<j) break;
        uint bi=b-j, sc=midiSC(bi), start=sc&0xFFFFu, count=sc>>16;
        for(uint k=0u;k<count;k++){
            uint si=start+k; if(si>=MIDI_SEG_COUNT) break;
            uint onT=onTick(si); if(T<onT) continue;
            uint dT=durTick(si), offT=onT+dT;
            if(T>offT+padT) continue;
            uint m=meta16(si); int nn=metaNote(m); float vel=metaVel(m); uint ins=metaInstr(m);
            float onSec=float(onT)*qSec, offSec=float(offT)*qSec;
            float ageOn=max(0.0,tEvent-onSec);
            float held=float(dT)*qSec;
            float ageOff=(tEvent>offSec)?(tEvent-offSec):0.0;
            float f=NoteToHz(nn), s=0.0, pan=0.0, gate=1.0;
            if(ins<=1u){ // melodic: gated
                gate=(tEvent<offSec)?clamp(ageOn/0.01,0.0,1.0)
                                    :exp(-ageOff/0.16)*clamp(held/0.01,0.0,1.0);
                if(gate<0.0005) continue;
            }
            if(ins==0u){ s=vLead(f,ageOn,vel)*gate*vel; pan=clamp((float(nn)-68.0)/22.0,-1.0,1.0)*0.42; }
            else if(ins==1u){ s=vBass(f,ageOn,vel)*gate*vel*0.95; }
            else if(ins==2u){ s=vKick(ageOn)*vel*1.15; }
            else if(ins==3u){ s=vSnare(ageOn)*vel*0.8; pan=0.08; }
            else { s=vHat(ageOn)*vel*0.5; pan=0.32; }
            L+=s*(0.5-pan*0.5); R+=s*(0.5+pan*0.5);
        }
    }
    vec2 lr=softLimit(vec2(L,R)*0.5);   // was tanh(x*0.5): same small-signal level, but transparent below the 0.85 knee
    return lr*smoothstep(0.0,0.4,tPhase);
}
"""

def build_band(notes, Q=256, shift=6, pad_sec=1.0, sr=44100.0):
    # notes: list of (start_s, dur_s, midi, vel127, instr 0..4)
    qSec=Q/sr; c16=lambda x:min(int(x),0xFFFF)
    ev=[]
    for (s,dur,midi,vel,instr) in notes:
        ev.append((round(s/qSec), max(1,round(min(dur,4.0)/qSec)), int(midi), int(vel), int(instr)))
    ev.sort(key=lambda e:e[0])
    N=len(ev); note_min=min(e[2] for e in ev); blk=1<<shift
    end_tick=c16(max(o+d for o,d,_,_,_ in ev)+1); nblocks=(end_tick+blk-1)//blk
    OM=[]; DUR=[]
    for onT,dT,midi,vel,instr in ev:
        vel6=max(0,min(63,round(vel/127*63)))
        meta=((midi-note_min)&0x7F)|((vel6&0x3F)<<7)|((instr&0x7)<<13)
        OM.append(c16(onT)|(meta<<16)); DUR.append(c16(dT))
    if len(DUR)%2: DUR.append(0)
    DURW=[DUR[i]|(DUR[i+1]<<16) for i in range(0,len(DUR),2)]
    SC=[]; idx=0
    for bb in range(nblocks):
        st=idx; cnt=0
        while idx<N and (ev[idx][0]>>shift)==bb: idx+=1; cnt+=1
        SC.append((st&0xFFFF)|(min(cnt,0xFFFF)<<16))
    maxdur=max(d for _,d,_,_,_ in ev); pad=round(pad_sec/qSec)
    lookback=(maxdur+pad)//blk+2
    defs="\n".join(["// paste whole file into the Sound tab",
        "#define MIDI_SAMPLE_RATE 44100.0",f"#define MIDI_TIME_Q_SAMPLES {Q}",
        f"#define MIDI_END_TICK {end_tick}u",f"#define MIDI_BLOCK_SHIFT_TICKS {shift}u",
        f"#define MIDI_BLOCK_COUNT {nblocks}u",f"#define MIDI_SEG_COUNT {N}u",
        f"#define MIDI_LOOKBACK_BLOCKS {lookback}u",f"#define MIDI_NOTE_MIN {note_min}",""])
    return (defs+"\n"+emit_sharded("midiOM",OM)+"\n\n"+emit_sharded("midiDur",DURW)
            +"\n\n"+emit_sharded("midiSC",SC)+"\n"+BAND_PLAYER)
