#!/usr/bin/env python3
# Standalone GGUF metadata dumper (stdlib only, no deps, no model load).
# Reads ONLY the header KV section — never touches tensor data — so it is
# instant even on a 28GB file. Prints every metadata key, then checks the
# arch keys the qengine inference path divides by (a missing one reads as 0
# -> SIGFPE during load). Usage: python3 dump_gguf_meta.py model.gguf
import sys, struct

GGUF_MAGIC = 0x46554747  # 'GGUF'
U8,I8,U16,I16,U32,I32,F32,BOOL,STRING,ARRAY,U64,I64,F64 = range(13)
FMT = {U8:'<B',I8:'<b',U16:'<H',I16:'<h',U32:'<I',I32:'<i',F32:'<f',BOOL:'<?',U64:'<Q',I64:'<q',F64:'<d'}
SZ  = {U8:1,I8:1,U16:2,I16:2,U32:4,I32:4,F32:4,BOOL:1,U64:8,I64:8,F64:8}

def main():
    if len(sys.argv) < 2:
        print("usage: dump_gguf_meta.py <model.gguf>"); return 1
    f = open(sys.argv[1], 'rb')
    magic, = struct.unpack('<I', f.read(4))
    if magic != GGUF_MAGIC:
        print("!! not a GGUF file (magic=0x%08x)" % magic); return 1
    version, = struct.unpack('<I', f.read(4))
    n_tensors, = struct.unpack('<Q', f.read(8))
    n_kv, = struct.unpack('<Q', f.read(8))
    print("GGUF v%d: %d tensors, %d metadata\n" % (version, n_tensors, n_kv))

    def rd_str():
        ln, = struct.unpack('<Q', f.read(8)); return f.read(ln).decode('utf-8','replace')
    def rd_val(t):
        if t in FMT: return struct.unpack(FMT[t], f.read(SZ[t]))[0]
        if t == STRING: return rd_str()
        if t == ARRAY:
            et, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
            head = []
            for i in range(n):
                v = rd_val(et)
                if i < 4: head.append(v)
            return "[array et=%d len=%d %s...]" % (et, n, head)
        raise ValueError("unknown value type %d" % t)

    meta = {}
    for _ in range(n_kv):
        k = rd_str(); t, = struct.unpack('<I', f.read(4)); v = rd_val(t)
        meta[k] = v
        s = str(v)
        print("%s = %s" % (k, s if len(s) <= 70 else s[:70] + "..."))

    arch = meta.get("general.architecture", "?")
    print("\n==== engine-critical arch keys (missing -> 0 -> SIGFPE) ====")
    crit = ["embedding_length", "block_count", "attention.head_count",
            "attention.head_count_kv", "attention.key_length",
            "attention.value_length", "full_attention_interval",
            "rope.dimension_count", "rope.freq_base", "nextn_predict_layers",
            "ssm.conv_kernel", "ssm.state_size", "ssm.group_count",
            "ssm.inner_size", "ssm.time_step_rank"]
    missing = []
    for c in crit:
        key = arch + "." + c
        if key in meta:
            print("  OK   %s = %s" % (key, meta[key]))
        else:
            print("  MISS %s" % key)
            missing.append(key)
    print("\nmissing %d critical key(s): %s" % (len(missing), missing if missing else "none"))
    return 0

sys.exit(main())
