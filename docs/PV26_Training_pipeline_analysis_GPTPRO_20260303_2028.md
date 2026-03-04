총평부터 말씀드리면, 이번 버전은 **확실히 좋아졌습니다.**
이제는 “일단 돌아가는 코드”가 아니라, **장기 학습을 돌려도 되는 실전형 코드**에 가깝습니다. 특히 `num_workers>0` + `pin_memory=True`, `zero_grad(set_to_none=True)`, `channels_last + AMP`, `cudnn.benchmark=True` 같은 선택은 PyTorch가 권장하는 성능 최적화 방향과 잘 맞습니다. 또한 지금처럼 `(Tensor, dict)` 형태로 배치를 반환하면 `pin_memory=True`의 기본 pinning 로직이 맵/iterable 내부의 텐서를 재귀적으로 처리할 수 있어서, 굳이 커스텀 batch 클래스를 만들지 않아도 됩니다. ([PyTorch Documentation][1])

또 하나 좋았던 점은 `torch.compile` 기본값을 `reduce-overhead`에서 `default`로 바꾼 것입니다. PyTorch 문서도 `default`를 성능과 오버헤드의 균형이 좋은 기본 모드로 설명하고, `reduce-overhead`는 CUDA graphs를 써서 Python 오버헤드를 줄이지만 메모리를 더 먹고 항상 잘 맞는 것은 아니라고 적고 있습니다. 그리고 `--compile-fullgraph` 플래그를 따로 둔 것도 아주 좋습니다. PyTorch는 성능이 기대보다 안 나오면 `fullgraph=True`로 graph break를 찾아 없애라고 권장합니다. ([PyTorch Documentation][2])

다만, **성능 우선**으로 보면 아직 손댈 곳이 분명히 있습니다. 우선순위대로 말씀드리겠습니다.

### 1) 가장 먼저 고칠 곳: criterion 안쪽의 남은 GPU sync

학습 루프 바깥의 `.item()`은 많이 치워졌지만, **진짜 hot path는 아직 `PV26Criterion` 안에 남아 있습니다.**

지금 e2e 경로에서 특히 신경 써야 할 부분은 대략 이 셋입니다.

* `_da_loss()`의 `if not bool(keep.any())`
* `_rm_loss()`의 `if not bool(keep.any())`
* `_od_loss_ultralytics()`의 `bool(valid_code.all())`, `bool(m.any())`

이런 식의 **CUDA tensor를 Python bool로 바꾸는 분기**는 CPU-GPU 동기화를 만들고, CUDA graph/compile 관점에서도 좋지 않습니다. PyTorch 튜닝 가이드도 `.item()` 같은 CPU-GPU sync를 피하라고 하고, CUDA graphs 문서도 CPU와 GPU를 동기화하는 연산이나 동적 제어 흐름은 캡처에 부적합하다고 설명합니다. ([PyTorch Documentation][1])

여기는 아래처럼 **branchless reduction**으로 바꾸는 것이 좋습니다.

```python
# _da_loss
keep_f = keep.to(dtype=per_sample.dtype)
return (per_sample * keep_f).sum() / keep_f.sum().clamp_min(1.0)

# _rm_loss
keep_f = keep.to(dtype=per_channel.dtype)
return (per_channel * keep_f).sum() / keep_f.sum().clamp_min(1.0)
```

그리고 `_od_loss_ultralytics()`에서는:

```python
# det_scope_code는 collate에서 이미 정제되었다고 보고 hot path 검사를 제거
keep_mask = has_det.ne(0)
if det_scope_code is not None:
    keep_mask = keep_mask & det_scope_code.ne(2)

keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)

src_old = det_tgt_batch_idx.to(device=device, dtype=torch.long)
new_idx = old_to_new[src_old]
m = new_idx.ge(0)

batch_idx_t = new_idx[m].to(dtype=torch.float32)
cls_t = det_tgt_cls.to(device=device, dtype=torch.float32)[m]
bboxes_t = det_tgt_bboxes.to(device=device, dtype=torch.float32)[m]
```

이렇게 하면 `m.any()` 같은 Python 분기가 사라집니다.

제 판단으로는, **지금 코드에서 학습 step time을 더 줄일 수 있는 가장 값싼 수정**이 바로 이것입니다.

### 2) validation 경로는 아직 비효율이 남아 있습니다

훈련보다 덜 중요해 보일 수 있지만, 50~150 epoch를 돌리면 validation도 누적 시간이 꽤 됩니다.

지금 `validate()`에서는 이미 한 번

```python
target_batch = _move_prepared_batch_to_device(target_batch_cpu, device=device)
```

를 했는데, 그 뒤에 다시

```python
da_mask_cpu[i].to(device=device)
rm_mask_cpu[i, c_idx].to(device=device)
```

를 하면서 **샘플 단위로 CPU→GPU 복사**를 다시 하고 있습니다. 이 부분은 그대로 손해입니다.

최소한 이렇게 바꾸는 것이 맞습니다.

```python
da_mask_dev = target_batch["da_mask"]
rm_mask_dev = target_batch["rm_mask"]

# 이후에는 da_mask_cpu / rm_mask_cpu 대신 da_mask_dev / rm_mask_dev 사용
```

그런데 여기서 한 걸음 더 가는 편이 좋습니다. 현재 `update_binary_iou()` 자체가 샘플마다 `sum().item()`을 하므로 **validation metric 계산이 샘플 단위 sync**가 됩니다. 이건 배치 단위로 묶는 것이 낫습니다.

예를 들면 DA는 이런 식입니다.

```python
da_mask = target_batch["da_mask"]
has_da = target_batch["has_da"].bool()

valid_da = (da_mask != 255) & has_da[:, None, None]
pred_da = preds.da[:, 0] > 0
tgt_da = da_mask == 1

metric_stats["da"]["inter"] += int((((pred_da & tgt_da) & valid_da).sum()).cpu())
metric_stats["da"]["union"] += int((((pred_da | tgt_da) & valid_da).sum()).cpu())
metric_stats["da"]["supervised"] += int(has_da.sum().cpu())
```

RM도 채널별로 같은 방식으로 묶으면 됩니다.
이렇게 하면 validation 쪽이 훨씬 깔끔해지고, profiling도 더 믿을 만해집니다.

### 3) train/eval collate는 이제 분리하는 편이 좋습니다

이건 모듈화이면서 동시에 성능 작업입니다.

지금 `_collate_with_images()`는 train loader에도 다음을 같이 싣고 갑니다.

* `sample_id`
* `det_yolo` 리스트
* `det_label_scope` 문자열 리스트

그런데 **train path에서는 이 중 상당수가 필요 없습니다.** e2e loss는 이미 `det_scope_code`, `det_tgt_batch_idx`, `det_tgt_cls`, `det_tgt_bboxes`를 쓰고 있으니, train용 collate는 더 가볍게 만들 수 있습니다.

권장 구조는 이렇습니다.

* `_collate_train`: 이미지 + tensor-only target
* `_collate_eval`: 이미지 + 메타데이터 포함 target

그리고 train loader는 `drop_last=True`를 권합니다. DataLoader의 `drop_last`는 마지막 작은 batch를 버리는 옵션이고, CUDA graphs는 **동일한 shape/layout**를 전제로 하므로 마지막 작은 batch는 compile/CUDA graph 관점에서 좋지 않습니다. 특히 `reduce-overhead`/CUDA graph 계열을 다시 시험할 생각이 있다면 train 쪽은 고정 batch가 유리합니다. DataLoader는 `drop_last`, `prefetch_factor`, `persistent_workers`를 이런 용도로 제공하고 있고, `prefetch_factor`는 worker당 미리 로드할 batch 수, `persistent_workers=True`는 epoch 이후에도 worker를 유지한다는 의미입니다. ([PyTorch Documentation][3])

즉, 저는 이렇게 두는 편을 권합니다.

```python
train_loader = DataLoader(
    train_ds,
    ...,
    collate_fn=_collate_train,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    ...,
    collate_fn=_collate_eval,
    drop_last=False,
)
```

### 4) 이제 진짜 병목은 train script보다 dataset 쪽입니다

이건 붙여주신 train 코드 바깥 이야기지만, 성능에는 더 중요합니다.

현재 `pv26/torch_dataset.py`를 보면 샘플 하나마다 대략 다음을 합니다.

* RGB 이미지 `Image.open`
* DA mask 읽기
* RM mask 3장 읽기
* det txt 읽어서 파싱
* PIL resize / letterbox
* NumPy 변환
* Torch tensor 변환

즉, **샘플당 파일 I/O + PIL + NumPy + Tensor 생성**이 꽤 많습니다.
A10G에서는 지금 버텨도, A100 80GB 같은 더 빠른 GPU로 가면 이쪽이 바로 다음 병목이 됩니다.

가장 ROI가 큰 방향은 새 알고리즘이 아니라 **오프라인 캐시**입니다.

* 960×544 letterbox 결과를 미리 저장
* `det_tgt_*`도 미리 저장
* train 때는 color jitter / hflip만 online
* 가능하면 `.pt` shard, LMDB, WebDataset tar 등으로 묶기

이게 가장 현실적입니다.
추가로, `_load_yolo_txt()`를 매 샘플마다 텍스트 파싱하는 것도 아까우니, det target을 manifest 옆 캐시 파일에 미리 직렬화해 두는 편이 좋습니다.

아주 나중 이야기지만, PyTorch 문서상 map-style dataset은 `__getitems__()`를 구현해서 batched sample loading을 가속하는 방법도 있습니다. 다만 제 판단으로는 **그 전에 오프라인 캐시가 우선**입니다. ([PyTorch Documentation][3])

### 5) profiling 기본값은 다시 “꺼짐”으로 두는 편이 맞습니다

지금은 profile 도구가 꽤 좋아졌지만, 여전히 regular run 기본값으로는 무겁습니다.

* `profile_every` 기본값 10
* `torch.cuda.mem_get_info()`를 매 batch에서 샘플링
* `nvidia-smi` subprocess 호출

이 셋이 합쳐지면, 프로파일링을 하지 않을 때도 코드가 profiling 친화적으로 돌아갑니다. PyTorch 튜닝 가이드도 일반 학습에서는 profiler 계열 디버깅 API를 꺼 두라고 권합니다. ([PyTorch Documentation][1])

그래서 저는:

* `--profile-every` 기본값을 `0`
* `mem_get_info()`도 `prof_every > 0`일 때만
* `nvidia-smi`는 지금처럼 profiling window에서만

이렇게 두는 편이 좋다고 봅니다.

### 6) compile 관점에서 가장 수상한 곳은 forward hook입니다

`PV26MultiHeadYOLO26`는 `self.det_model.model[4].register_forward_hook(...)`로 `p3_backbone`을 `self._feat` dict에 저장하고 있습니다.

이 패턴은 기능적으로는 괜찮지만, **compile/fullgraph 입장에서는 상당히 수상합니다.**
PyTorch는 graph break가 느려짐의 원인이 될 수 있다고 하고, `fullgraph=True`는 이런 break를 찾는 용도로 쓰라고 권합니다. ([PyTorch Documentation][4])

따라서 `--compile-fullgraph`를 넣은 것은 매우 잘하셨고, 이 옵션으로 에러가 나면 **첫 번째 용의자는 forward hook + Python dict side-effect**라고 보시면 됩니다.

중장기적으로는 hook 대신:

* det trunk wrapper가 `p3_backbone`을 직접 반환하거나
* segmentation heads용 feature를 명시적으로 뽑는 함수로 바꾸는 것

이 더 좋습니다.

### 7) TF32 설정은 나중에 한 번 정리하면 됩니다

지금의

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

는 당장 나쁘지 않지만, 최신 PyTorch 문서에서는 `allow_tf32` 계열이 **deprecated 방향**이고, 새 `fp32_precision` API 사용을 권장합니다. 다만 이것은 **지금 당장 throughput을 바꾸는 1순위 작업은 아닙니다.** 현재 버전을 계속 쓸 거면 놔둬도 되고, PyTorch 2.9+로 올릴 때 정리하면 됩니다. ([PyTorch Documentation][5])

---

### 모듈화는 꼭 해야 하나?

제 답은 이렇습니다.

**지금 당장 대대적 모듈화는 필요 없습니다.**
1201줄 자체가 문제는 아닙니다. 성능은 파일 길이보다 **hot path의 purity**와 **데이터 경로의 무게**에서 갈립니다.

다만 아래 세 군데는 나누는 편이 좋습니다.

* `collate.py`
  `_collate_train`, `_collate_eval`, `_move_prepared_batch_to_device`, `_prepare_images_for_model`

* `runtime.py`
  optimizer/scheduler/build, compile, checkpoint save/load

* `profile.py`
  `_format_cuda_mem_stats`, `_query_nvidia_smi`, `_format_cpu_stats`, profiler state

이 정도면 충분합니다.
반대로 checkpoint class, logger class, runner class까지 과하게 쪼개는 것은 **지금 시점에서는 속도 이득이 없습니다.**

---

### 제 우선순위 결론

지금 바로 손댈 순서는 이렇습니다.

1. `PV26Criterion` 안의 남은 GPU sync 제거
2. `validate()`에서 이미 올린 device mask 재사용 + batch-wise IoU 누적
3. train/eval collate 분리 + train `drop_last=True`
4. dataset 오프라인 캐시(최대 효과)
5. `profile_every` 기본 0으로 변경
6. `compile_fullgraph`로 hook 문제 확인

이 여섯 개 중 **1~3만 해도 코드 자체의 완성도가 한 단계 더 올라가고**, **4번까지 가면 A10G뿐 아니라 A100 80GB에서도 훨씬 정직하게 스케일**될 가능성이 큽니다. 지금 버전은 분명 좋아졌고, 이제부터는 “코드 정리”보다 **criterion sync 제거와 dataset 캐시**가 더 큰 돈값을 합니다.

[1]: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html "Performance Tuning Guide — PyTorch Tutorials 2.11.0+cu130 documentation"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html "torch.compile — PyTorch 2.10 documentation"
[3]: https://docs.pytorch.org/docs/stable/data.html "https://docs.pytorch.org/docs/stable/data.html"
[4]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html "https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html"
[5]: https://docs.pytorch.org/docs/stable/notes/cuda.html "https://docs.pytorch.org/docs/stable/notes/cuda.html"
