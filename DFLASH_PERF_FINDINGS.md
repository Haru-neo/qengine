# DFlash 추론 속도 연구 결과 (2026-06-05 밤)

두 질문: (1) 컨텍스트가 커져도 속도를 유지할 수 있나? (2) 30 t/s가 진짜 한계인가?

---

## 오늘 한 일 (커밋·검증 완료)

| 변경 | 효과 |
|---|---|
| **A. 드래퍼 증분 K/V 캐시** (`2532368`) | fc(905GFLOP)+wk/wv 전체 재계산 → 새 행만. O(ctx)→O(accept) |
| **B. 배치 multi-query verify** (`13d6c23`) | verify가 TQ KV를 8× → **1× read**. greedy bit-identical 검증 |
| **task5. DFlash면 MTP KV 스킵** | GPU2 1GB 해방 (16014→14964MB) |

**측정 (gen-only t/s):**
```
컨텍스트   이전(recompute)  A+B(현재)
short(20)        ~16?          30.1   ← MTP 24-27 추월
3.4K             5.5           15.6
11K              3.7           17.5   (4.7×)
```
256K 로딩·정상 작동·needle 12.9K 통과(윈도우 3회 wrap).

---

## Q2: 30 t/s가 한계인가? → **아니다. AL × 단일스트림 GEMV대역폭 천장이지 SW/HW 벽이 아님.**

short-context 30 t/s를 PROFILE_GEN으로 분해 (budget=5, AL 2.85, iter ~95ms):
- **verify가 매 iter 27B 가중치 전체(~28GB)를 1회 read** (budget 토큰 배치라 budget과 무관)
  - MLP read 측정 = 18.2GB / 37ms = **~491 GB/s** = Q8 GEMV 대역폭 천장 근처
  - HBM2 이론 800 / memcpy 실측 700 대비 낮은 건 GEMV(dequant+dp4a)가 순수 memcpy보다 비효율
- **3-GPU는 PP(파이프라인)** → 단일 스트림에선 한 번에 GPU 1개만 활성, 2/3 유휴. TP면 3×지만 **P2P 없는 CMP에선 layer마다 all-reduce가 죽음** → 단일스트림 천장 = ~1 GPU 대역폭
- 드래퍼 22.6ms/iter (iter의 23%) — 드래퍼 1.97GB 가중치 read + 16-noise GEMM

**식: t/s = AL / (verify가중치read 61ms@491GB/s + 드래퍼 22ms + 오버헤드)**

### 30을 넘는 레버 (효과 순)
1. **더 높은 AL = 분포-매칭 전용 드래퍼 학습** ← 가장 큼. 가중치는 iter당 1회 read 고정이라 AL이 직접 amortize. AL 2.85→5면 ~45-50 t/s. (→ `project_dflash_custom_drafter_training`)
2. **GEMV 커널 최적화** 491→~550-600 GB/s (~15-20%). CLAUDE.md P0(Q5_K은 과거 519 달성). 어렵고 CMP-전용.
3. **budget 튜닝 = 안 통함**: 8→5에서 AL 3.1→2.85 떨어져 오히려 느려짐. 가중치 read는 budget 무관이라 줄지도 않음.
4. 유휴 2 GPU = P2P 없어 못 씀.

**결론: 커널/엔진은 A+B 이후 거의 최적. 30 위는 거의 전적으로 드래퍼 AL(재학습) 문제.**

---

## Q1: 컨텍스트 커져도 속도 유지? → **현재는 NO(곡선 가파름). fix는 sparse verify.**

현재 dense verify 곡선:
```
ctx     gen t/s   verify-attn 비중
6K       16.5       48%
18K      14.1       73%
43K       4.7       86%
256K     ~0.6(추정)  ~95%
```
원인: verify-attn = 27B가 chain 토큰을 검증할 때 **풀컨텍스트 TQ KV를 WHT로 cooperative decode**(compute-bound), 각 chain query(t_idx)마다 [0..pos] 재디코드 → **O(ctx)**. 256K면 지배.

### ✅ 밤에 실증함: sink+window sparse verify로 곡선 평탄화 (env-gated, 기본 OFF)
`DFLASH_VERIFY_SPARSE=1`(sink 256 + window 4096) 구현·측정. tq3 커널이 sink+최근window 타일만 디코드 → verify-attn이 O(window) 상수.

| ctx | dense | **sparse** | |
|---|---|---|---|
| 6K | 16.5 | **19.0** | +15% |
| 18K | 14.1 | 13.9 | wash (반복프롬프트 dense AL 3.73을 sparse 3.08이 못 따라가 상쇄) |
| 43K | **4.7** | **13.3** | **2.8× — 곡선 평탄화** |

→ dense는 16.5→4.7 폭락, sparse는 13-19로 평탄. **"컨텍스트 커져도 속도 유지" 가능 실증.**

**품질(정직):** sink+window는 **lossy**. 측정:
- needle-at-TOP: **found** (sink=256이 보존) ✓
- needle-at-MIDDLE(~9K/18K): **NOT found** (sink/window 밖 손실) ✗
- AL도 dense 대비 하락(43K 3.30→3.01) = 출력이 dense와 달라짐
→ 채팅/코딩(최근 컨텍스트 위주)엔 적합, **임의위치 회수 필요한 장문 QA엔 부적합.** 기본 OFF 유지. window/sink는 env로 키우면 품질↑속도↓.

### 무손실 대안 = profile-guided block-sparse (prefill에 이미 있음, verify만 dense)
설계 워크플로(adversarial 검증 포함) 결론 **Approach A (TQ-aware block-sparse verify)**:
- prefill의 K_pool + block-index 빌더 + sparse FA 재사용, 선택된 ~5% 블록만 TQ→디코드
- 새 커널 `dequant_selected_tq3_blocks` + 증분 K_pool(high-water-mark) + per-query index
- **예상 곡선**: 43K 4.7→**~15-16**, 256K ~0.6→**~10-12** (곡선 평탄화)
- env-gated `DFLASH_VERIFY_SPARSE=1`, 기본 OFF

### ⚠️ 왜 오늘 밤 안 질렀나 (정직)
- 규모: **~400-500줄, 5-8일, medium-risk** (TQ WHT 디코드 + 증분 K_pool + sparse index)
- **adversarial verifier 3개 approach 전부 `quality_ok=False`**: 프로파일은 prefill용으로 만든 건데 verify에 쓰는 losslessness가 **미검증**. needle-in-middle 등 품질 검증 필요.
- L3 sticky 버그(`INHERITED_STICKY_ERR invalid argument`, prefill 첫 attn 레이어, 43K prefill 133s로 느림) root cause 미규명.
- 품질 미검증 대형 커널을 무감독으로 기본탑재하면 A+B(검증완료, MTP 추월) working state를 깰 위험. → **사용자 품질 기준 확인 후 진행 권장.**

### 흥미로운 통찰 (검토 요)
드래퍼가 이미 **window=4096로 제한**돼 최근 컨텍스트만 보고 예측함. 그래서 verify도 sink+window(~4096+sink) 정도면 드래퍼 예측을 잘 검증할 수 있음 — verify-attn을 **상수(O(window))로 만들어 곡선 완전 평탄화** 가능. 단 "전체 문서 요약" 같이 far-context가 필요한 task엔 출력 품질 손실(prefill이 sparse여도 profile-block ≠ window). **task별 품질 트레이드오프라 사용자 결정 필요.**

---

## 다음 단계 (사용자 결정)
1. **sparse verify 진행?** (롱컨텍스트 평탄화, 5-8일, 품질 검증 동반) — 곡선을 정말 평탄하게 만드는 유일한 길.
2. **드래퍼 재학습?** (short+all 천장↑, AL 2.85→5) — 30→50 t/s의 길. 이미 트레이너 코어 검증됨(`project_dflash_custom_drafter_training`).
3. L3 sticky 버그 fix (prefill 가속, 독립적).

현재 서버는 batched-on / budget=8 기본으로 정리해둠 (short 30 / 11K 17.5 t/s 사용 가능).
