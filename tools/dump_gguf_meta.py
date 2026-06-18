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

    # Optional tensor-table dump: `dump_gguf_meta.py model.gguf tensors`.
    # The tensor infos follow the KV section: name, n_dims(u32), dims[n_dims]
    # (u64), type(u32), offset(u64). Lets us compare tensor shapes/types
    # between two GGUFs (e.g. a metadata-light repack vs the deployed file)
    # when the arch metadata matches but a load-time divide still trips.
    want_tensors = len(sys.argv) > 2 and sys.argv[2] == "tensors"
    if want_tensors:
        print("\n==== tensor table (%d tensors) ====" % n_tensors)
        GGML_TYPE = {0:"F32",1:"F16",8:"Q8_0",2:"Q4_0",12:"Q4_K",14:"Q6_K"}
        rows = []
        for _ in range(n_tensors):
            tn = rd_str()
            nd, = struct.unpack('<I', f.read(4))
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            tt, = struct.unpack('<I', f.read(4))
            off, = struct.unpack('<Q', f.read(8))
            rows.append((tn, dims, tt))
        # Print blk.0 + blk.64 (MTP) + non-blk tensors in full; summarize the rest.
        def show(tn): return tn.startswith("blk.0.") or tn.startswith("blk.64.") or not tn.startswith("blk.")
        for tn, dims, tt in rows:
            if show(tn):
                print("  %-44s %-5s %s" % (tn, GGML_TYPE.get(tt, "t%d"%tt), dims))
        print("  ... (%d blk.1..63 tensors elided)" % sum(1 for tn,_,_ in rows if not show(tn)))
        return 0

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
