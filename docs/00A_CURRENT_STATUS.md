# 00A. Current Status

> 다음 작업자는 이 문서를 먼저 읽는다.
> 상세 실패 이력은 `00B_STATUS_HISTORY.md`, 다음 실행 gate는 `00C_NEXT_GATES.md`를 본다.

## 1. 한 줄 결론

PV26은 exhaustive OD + lane-family 통합 학습 경로와 derived fine-tune 경로가 구현되어 있고, lane-family는 exact epoch-2 two-checkpoint runtime probe 기준 `phase_objective=0.6525130665`, broader-val512 two-checkpoint runtime probe 기준 `0.6467983828`까지 확인됐다.

이 60% objective 돌파는 raw model만으로 만든 결론이 아니고, 세 task F1이 모두 0.6을 넘었다는 뜻도 아니다. core-centerline/refinement checkpoint 위에 row-scan/tangent-link vectorizer, small-fragment FP postprocess filters, lane-head transplant, flip-centerline TTA, crosswalk-mask lane competition, projection-competition stop-line runtime decode, 또는 hull-based crosswalk decode가 붙어서 만든 partial success다. Latest stop/cross lane-frozen trained composite는 broader-val512 objective `0.6421`, lane/stop-line/crosswalk F1 `0.5571 / 0.5278 / 0.6142`이고, retained lane-preserving task-balance composite는 `0.5628 / 0.5164 / 0.6187`이다. User-requested larger-range `2048` train-batch scale audit of the same lane-frozen axis regressed to broader lane/stop/cross `0.5564 / 0.5122 / 0.6162`, so it is negative. Latest lane seed-trace instance decoder larger-slice train reached exact-val128 epoch-2 lane/stop/cross `0.5488 / 0.5000 / 0.5644`, so it is also negative. Latest input-scale `672x896` dense-target train reached fixed broader-val512 lane/stop/cross `0.5519 / 0.4734 / 0.6257`, so it is negative despite crosswalk passing. Latest stop-line axis-distance field train reached fixed broader-val512 lane/stop/cross `0.5431 / 0.2629 / 0.6081`, so it is negative and far below the projection-competition stop-line reference. Latest lane center-offset field train reached fixed broader-val512 lane/stop/cross `0.5390 / 0.5164 / 0.6166`, so it is also negative: stop-line only matches the retained projection-competition runtime reference while lane regresses sharply. Latest stop-line focus-crop feeding train reached fixed broader-val512 lane/stop/cross `0.5454 / 0.4061 / 0.5910`, so crop/zoom feeding is negative: stop-line TP rises versus raw runtime but FP nearly doubles versus projection-comp. Latest upper-trunk retention-distill train reached fixed broader-val512 lane/stop/cross `0.5336 / 0.5195 / 0.6200`, so retention distill preserves crosswalk and gives only a tiny stop-line bump over projection-comp while lane collapses further. Latest two-checkpoint stop-line router audit preserves retained lane/crosswalk while routing stop-line from the stop-line-priority specialist, giving broader lane/stop/cross `0.5628 / 0.5309 / 0.6187`; however a newly trained stop-line-only upper-trunk router specialist regressed to `0.5628 / 0.4826 / 0.6187`. Latest lane-router specialist upper-trunk smoke was rejected: fixed router val4 lane regressed `0.5839 -> 0.5735`, TP/FP/FN `40 / 11 / 46 -> 39 / 11 / 47`, so it was not broadened. Latest stop-line dual-source arbitration was exact-only positive but broader-negative: exact `primary_absent_specialist` moved stop-line `32 / 28 / 28 -> 33 / 28 / 27`, but broader fell below the specialist router at `129 / 103 / 142`, F1 `0.5129` versus `0.5309`. Latest V3 stop-line isolated-neck train was internal-val positive but fixed exact-negative: training val epoch 3 stop-line reached `34 / 28 / 19`, F1 `0.5913`, but fixed router exact-val128 epoch-2 fell to `27 / 42 / 33`, F1 `0.4186`, so broader was skipped. Latest stop-line midpoint proposal head train added a dedicated midpoint proposal logit and trained it at real `3`-epoch/`512` train-batch scale, but fixed router exact-val128 fell to lane/stop/cross `0.5888 / 0.1739 / 0.5988`, stop-line `6 / 3 / 54`, so broader was skipped. Latest heterogeneous endpoint-pair router audit fixed the specialist scenario/postprocess contract plumbing, but exact-val128 endpoint-pair specialist was only stop-line `21 / 43 / 39`, F1 `0.3387`, and all dual-source modes stayed below primary projection-comp exact `0.5333`. User-requested larger-range `2048` train-batch stop-line-priority positive-sampler audit also failed fixed exact: internal epoch-2 stop-line reached `63 / 46 / 44`, F1 `0.5833`, but router exact-val128 was only `30 / 34 / 30`, F1 `0.4839`, so broader was skipped. Latest stop-line-only mask-first specialist train was also internal-val positive but fixed exact-negative: training val epoch 3 stop-line reached `34 / 29 / 19`, F1 `0.5862`, but fixed router exact-val128 fell to `28 / 42 / 32`, F1 `0.4308`, so broader was skipped. Latest learned source-router trained a small MLP over primary/specialist output statistics, but exact-val128 learned-router stop-line was only `30 / 29 / 30`, F1 `0.5042`, below primary projection-comp `32 / 28 / 28`, F1 `0.5333`; oracle routing would reach `34 / 2 / 26`, F1 `0.7083`, so source headroom exists but the current no-GT output-stat signal is not enough. Latest dense-aligned source-router reopened only the permitted materially different runtime quality signal premise by adding line-aligned dense-map/raw-image features and training on `256` batches, but exact-val128 learned-router fell to stop-line `27 / 23 / 33`, F1 `0.4909`, versus primary `32 / 28 / 28`, F1 `0.5333`; broader was skipped. Latest raw-patch geometry-repair replay trained a no-GT MLP to predict stop-line candidate repair score plus endpoint deltas, but exact held-out collapsed to stop-line `5 / 34 / 26`, F1 `0.1429`, and all-split train-threshold replay was only `25 / 34 / 35`, F1 `0.4202`, below baseline `0.4483` and projection-comp `0.5167`; broader was skipped. All of these still leave lane/stop-line below `0.60`.

Latest stop-line det-source hard-negative feeding changed the data-feeding contract, not the decoder: stage-4 train loader can include det-source-only BDD/traffic/obstacle records from the existing full canonical root, then marks only those train samples as stop-line source-empty hard negatives while validation remains lane-family-only. It trained a real heads-only CUDA `2x64` smoke run, indexed the existing `429350` records in place, and created no dataset copy. The train history confirms the larger-data exposure (`det_source_samples=1`, `lane_source_samples=3`, `crosswalk_source_samples=3`, `stop_line_source_samples=4` on logged batches), but fixed val4 epoch-2 best was smoke-negative: lane/stop/cross `0.4885 / 0.0000 / 0.5455`, lane TP/FP/FN `32 / 13 / 54`, stop-line `0 / 3 / 2`, crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because it recovered no stop-line TP and sharply regressed lane. Negative checkpoints were pruned; retained run size is about `7.9M`.

Latest task-rotation sampler trained a real heads-only checkpoint with epoch-level `task_positive_task="rotate:stopline,lane,crosswalk"` feeding instead of mixing task-positive records inside each batch. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, ran CUDA `3` epochs by `64` train batches with skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-3 best with the retained flip/cross-mask lane variant reached lane/stop/cross `0.5224 / 0.0000 / 0.5455`, lane TP/FP/FN `35 / 13 / 51`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`, below the retained fixed reference lane `0.5839`, `40 / 11 / 46`. Exact-val128 and broader-val512 were skipped because fixed val4 lost lane TP, added lane FP, and recovered no stop-line TP. Internal phase objective reached `0.6704`, but this is not success evidence. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `816K`.

Latest stop-line patch projection-merge made the existing local patch segment candidates participate inside the projection-competition runtime candidate pool instead of being bypassed by the projection-comp early return, then trained a real heads-only CUDA `2x64` smoke checkpoint. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` reached lane/stop/cross `0.5373 / 0.0000 / 0.5455`, lane TP/FP/FN `36 / 12 / 50`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because the patch candidates added no stop-line TP and lane stayed below the retained fixed reference `0.5839`, `40 / 11 / 46`. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `824K`.

Latest task-uncertainty loss balancer added trainable per-task log-variance weighting to the stage-4 criterion and optimizer, so lane/stop-line/crosswalk loss allocation is learned instead of using only fixed scalar weights or EMA normalization. It trained a real heads-only CUDA `2x64` smoke checkpoint, reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, had skipped steps `0`, and created no dataset copy. The criterion optimizer group was present and learned small log-vars (`lane=0.0026`, `stop_line=0.0027`, `crosswalk=-0.0032`), but the run is smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` reached lane/stop/cross `0.5373 / 0.0000 / 0.5455`, lane TP/FP/FN `36 / 12 / 50`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because learned task-loss allocation recovered no stop-line TP and lane stayed below the retained fixed reference `0.5839`, `40 / 11 / 46`. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `820K`. Latest lane instance-embedding auxiliary added a dense per-lane discriminative embedding target/loss to the seg-first lane head while retaining the same row-scan/tangent runtime decode, projection-comp stop-line decode, and hull crosswalk decode. It trained a real heads-only CUDA `2x64` smoke checkpoint, reused the existing dataset root in place, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` again reached lane/stop/cross `0.5373 / 0.0000 / 0.5455`, lane TP/FP/FN `36 / 12 / 50`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because the auxiliary recovered no stop-line TP and did not improve fixed smoke lane TP/FP/FN over the previous negative training baseline. Latest shared affine train augmentation added a single mild affine warp applied consistently to image, detector boxes, and lane/stop-line/crosswalk geometries, then trained a real heads-only CUDA `2x64` smoke checkpoint on the existing dataset root. It is smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` reached lane/stop/cross `0.5373 / 0.0000 / 0.5455`, lane TP/FP/FN `36 / 12 / 50`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`; exact-val128 and broader-val512 were skipped because it lost lane TP versus the retained fixed reference and recovered no stop-line TP. Latest synthetic stop-line injection added train-time synthetic stop-line pixels and labels from lane geometry without copying data, then trained a real heads-only CUDA `2x64` smoke checkpoint on the existing dataset root. It is smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` reached lane/stop/cross `0.5414 / 0.0000 / 0.5455`, lane TP/FP/FN `36 / 11 / 50`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`; exact-val128 and broader-val512 were skipped because it recovered no stop-line TP and still lost `4` lane TP versus the retained fixed reference.

Latest real stop-line copy-paste feeding replaced the previous synthetic bright-line injection with train-time real donor patches from existing stop-line-positive train samples, loaded on demand from the same canonical dataset root without copying data. It trained a real heads-only CUDA `2x64` smoke checkpoint and is also smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` reached lane/stop/cross `0.5414 / 0.0000 / 0.5455`, lane TP/FP/FN `36 / 11 / 50`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`; exact-val128 and broader-val512 were skipped because it recovered no stop-line TP and still lost `4` lane TP versus the retained fixed reference. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `824K`.

Latest stop-line positive-only empty-sample loss masking trained stop-line dense/vector losses only on source-valid samples that contain at least one stop-line, instead of letting empty `aihub_lane_seoul` source samples act as full stop-line negatives. It trained a real heads-only CUDA `2x64` smoke checkpoint on the existing canonical dataset root with skipped steps `0` and no dataset copy. It is smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` reached lane/stop/cross `0.5414 / 0.0000 / 0.4000`, lane TP/FP/FN `36 / 11 / 50`, stop-line `0 / 3 / 2`, and crosswalk `2 / 1 / 5`; exact-val128 and broader-val512 were skipped because it recovered no stop-line TP, still lost `4` lane TP versus the retained fixed reference, and regressed fixed smoke crosswalk. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `824K`.

Latest current-family vector decoder reconnected the repo's older P3-P5 query-vector lane/stop-line/crosswalk architecture as a selectable `roadmark_architecture="current_family"` and trained it at real `3`-epoch/`512` train-batch scale. It reused the existing canonical dataset root, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is strongly negative: internal val128-style epoch-2 validation had lane/stop/cross F1 all `0.0000`, TP/FP/FN lane `0 / 0 / 2390`, stop-line `0 / 0 / 60`, crosswalk `0 / 0 / 81`; best phase objective was only `0.2677`. Fixed exact/broader were skipped because the larger-slice validation emitted no matched lane-family objects. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `2.6M`.

Latest full-trunk retention-distill train reopened the shared-feature exposure premise with a new opt-in `lane_family_full_trunk` freeze policy: detector/TL heads stayed frozen, all lane-family heads stayed trainable, and the full trunk was updated at `5e-7` while lane/crosswalk were retained by teacher distill from the seed checkpoint. It trained a real CUDA `2x64` smoke on the existing canonical dataset root, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-2 best with `flip_centerline_avg_lane_cross_comp050` reached lane/stop/cross `0.5481 / 0.0000 / 0.5455`, lane TP/FP/FN `37 / 12 / 49`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because opening the full trunk lost `3` lane TP and added `1` lane FP versus the retained fixed reference `0.5839`, `40 / 11 / 46`, while recovering no stop-line TP. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `916K`.

Latest current-family sigmoid vector decoder reopened the older P3-P5 query-vector architecture with a materially different coordinate contract: lane x coordinates and stop-line/crosswalk point coordinates are sigmoid-mapped into network pixel space before loss/eval, instead of using unconstrained raw coordinates. It trained a real CUDA `2x64` smoke on the existing canonical dataset root, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-2 best reached lane/stop/cross `0.0000 / 0.0000 / 0.0000`, lane TP/FP/FN `0 / 169 / 86`, stop-line `0 / 68 / 2`, and crosswalk `0 / 0 / 7`. This fixes the previous raw-vector no-emission shape only by producing unmatched FP-heavy vectors, so larger `3x512`, exact-val128, and broader-val512 were skipped. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `7.5M`.

Latest current-family anchor query prior reopened that vector-query architecture with trainable query-specific network-geometry priors for lane row-x/visibility, horizontal stop-line segments, and rectangle-like crosswalk contours. This is a changed initialization/geometry contract, not another postprocess threshold sweep. It trained a real CUDA `2x64` smoke on the existing canonical dataset root, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is still smoke-negative: fixed val4 epoch-2 best reached lane/stop/cross `0.0000 / 0.0000 / 0.0000`, lane TP/FP/FN `0 / 131 / 86`, stop-line `0 / 27 / 2`, and crosswalk `0 / 22 / 7`. The anchor prior reduced lane/stop-line FP versus sigmoid-only but still recovered zero matched TP and introduced crosswalk FP, so exact-val128, broader-val512, and larger training were skipped. Negative checkpoints, TensorBoard, and root weights were pruned; retained run size is about `7.5M`.

Latest current-family anchor query-seed decoder made the anchor geometry feed the decoder query input itself and shared that geometry-query MLP with the training denoise path. It trained a real CUDA `2x64` smoke on the existing canonical root, indexed `429350` records, and created no dataset copy. It is strongly smoke-negative: fixed val4 epoch-2 best reached lane/stop/cross `0.0000 / 0.0000 / 0.0000`, lane TP/FP/FN `0 / 0 / 86`, stop-line `0 / 0 / 2`, and crosswalk `0 / 127 / 7`. Exact-val128 and broader-val512 were skipped because the first fixed gate recovered no matched lane-family objects. Negative checkpoints and TensorBoard were pruned; retained run size is about `7.8M`.

Latest lane seed-relative row decoder changed the conditional lane instance coordinate contract from absolute sigmoid x to seed-column anchored bounded residuals (`seed_relative`, `max_delta=160`). It trained a real CUDA `2x64` smoke on the existing canonical dataset root, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-2 best reached lane/stop/cross `0.0000 / 0.0000 / 0.5000`, lane TP/FP/FN `0 / 73 / 86`, stop-line `0 / 3 / 2`, and crosswalk `3 / 2 / 4`. The seed-relative contract reduced lane FP versus previous rescue-append (`168 -> 73`) but still recovered zero lane TP and regressed crosswalk versus retained hull behavior, so exact-val128, broader-val512, and larger training were skipped. Negative checkpoints, TensorBoard, and root weights were pruned; retained run size is about `7.8M`.

Latest lane conditional row branch-only smoke added `lane_conditional_row_only`, which trains only `conditional_seed_logits` plus `conditional_query_mlp` while keeping the trunk, dense lane maps, stop-line head, and crosswalk head frozen/eval. It trained a real CUDA `2x64` smoke on the existing canonical dataset root, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-2 best reached lane/stop/cross `0.2809 / 0.0000 / 0.5455`, lane TP/FP/FN `33 / 116 / 53`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Branch-only training prevents dense-head drift but the appended conditional rows still create severe lane FP versus the retained fixed reference `40 / 11 / 46`, so exact-val128, broader-val512, and larger training were skipped. Latest lane conditional row dense gate added an opt-in runtime quality contract that keeps appended conditional rows only when their visible anchors lie on dense centerline/support evidence. It trained a real CUDA `2x64` smoke on the same canonical dataset root, again indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is still smoke-negative: fixed val4 epoch-2 best reached lane/stop/cross `0.5507 / 0.0000 / 0.5455`, lane TP/FP/FN `38 / 14 / 48`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. The dense gate suppresses most branch-only FP (`116 -> 14`) but remains below the retained fixed lane reference (`40 / 11 / 46`) and recovers no stop-line TP, so exact-val128, broader-val512, and larger training were skipped. Negative checkpoints, TensorBoard, and root weights were pruned; retained run size is about `6.6M`.

Latest task-routed multi-teacher distill student trained a single-checkpoint student from a primary lane/crosswalk teacher plus the stop-line-priority specialist teacher. It is fixed single-checkpoint exact-negative: exact-val128 epoch-2 reached lane/stop/cross `0.5775 / 0.4754 / 0.5854`, with stop-line `29 / 33 / 31`; broader was skipped because it is below the projection-competition exact stop-line reference and crosswalk is still below `0.60`.

Latest stop-line local 2D patch segment head trained a model-side segment-emission contract at real `3`-epoch/`512` train-batch scale. It reused the existing `pv26_exhaustive_od_lane_dataset` in place and created no dataset copy. Fixed exact-val128 epoch-2 reached lane/stop/cross `0.5660 / 0.4333 / 0.5904`, with stop-line `26 / 34 / 34`; broader was skipped because stop-line is below the projection-competition exact reference and crosswalk remains below `0.60`. Retained run size after pruning duplicate checkpoints/TensorBoard/root weights is about `115M`.

Latest stop-line raw-patch verifier replay trained a small no-GT MLP verifier on existing candidate rows plus oriented raw-image patches from the existing dataset root. It is exact held-out negative: train split improved stop-line to `12 / 0 / 17`, F1 `0.5854`, but held-out collapsed from baseline `16 / 15 / 15`, F1 `0.5161`, to `4 / 3 / 27`, F1 `0.2105`; all-split train-threshold replay was `16 / 3 / 44`, F1 `0.4051`. Broader-val512 was skipped because the held-out exact gate failed. No dataset copy was created.

Latest stop-line raw-patch geometry-repair replay extended that verifier into a learned endpoint-repair probe. It reused the existing dataset root, trained on `514` exact candidate rows with `261` repair targets, and created no model artifact. It is exact held-out negative: train split overfit to `20 / 0 / 9`, F1 `0.8163`, but held-out fell from baseline `16 / 15 / 15`, F1 `0.5161`, to `5 / 34 / 26`, F1 `0.1429`; all-split replay was `25 / 34 / 35`, F1 `0.4202`. Broader-val512 was skipped.

Latest stop-line raw-patch CNN verifier replay replaced the flattened raw-patch MLP with a small learned CNN over oriented 2-channel raw/gradient patches plus numeric candidate features. It reused the existing dataset root, trained on `514` exact candidate rows with `148` oracle-positive rows, and created no checkpoint artifact. It is exact held-out negative: train split improved to `12 / 0 / 17`, F1 `0.5854`, but held-out fell from baseline `16 / 15 / 15`, F1 `0.5161`, to `7 / 8 / 24`, F1 `0.3043`; all-split train-threshold replay was `19 / 8 / 41`, F1 `0.4368`, still below baseline `0.4483` and projection-comp `0.5167`. Broader-val512 and larger training were skipped.

Latest stop-line raster source-router reopened the source-headroom premise with a spatial quality signal: a tiny CNN sees raw image, primary/specialist/union stop-line rasters, and primary/specialist dense stop-line mask/center/selector maps. It trained/evaluated using the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and created no dataset copy. Smoke train64/val4 recovered no stop-line TP. Exact train64/val128 was negative at learned-router stop-line `25 / 23 / 35`, F1 `0.4630`; the user-requested larger train256/val128 run improved only to `29 / 26 / 31`, F1 `0.5043`, still below primary projection-comp `32 / 28 / 28`, F1 `0.5333`. Oracle routing on the same exact slice remains `34 / 2 / 26`, F1 `0.7083`, so source headroom still exists, but this no-GT raster quality signal does not learn it. Broader-val512 was skipped because the larger exact gate failed. Retained raster-router exports are CSV/summary-only and about `48K-52K` each.

Latest task-loss EMA balancer trained a real stage-4 heads-only checkpoint with per-task EMA loss normalization over lane/stop-line/crosswalk. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran CUDA smoke plus `3` epochs at `512` train batches and `128` val batches. It is negative: fixed exact-val128 epoch-2 reached lane/stop/cross `0.5577 / 0.4640 / 0.6024`, and broader-val512 reached `0.5388 / 0.5077 / 0.6199`. Crosswalk stays pass, but lane regresses below retained `0.5628` and stop-line stays below both projection-comp broader `0.5164` and the two-checkpoint router `0.5309`.

Latest lane-only seg-first specialist made the existing lane-only head architecture selectable as `roadmark_architecture="lane_only_row_classifier"` and trained a real lane-only specialist smoke checkpoint. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran CUDA `1` epoch with `8` train batches and `4` val batches. It is smoke-negative: fixed router val4 reference lane/stop/cross was `0.5839 / 0.0000 / 0.5455`, TP/FP/FN lane `40 / 11 / 46`; routing the trained lane-only checkpoint changed lane to `0.5735`, TP/FP/FN `39 / 11 / 47`, while stop-line/crosswalk stayed unchanged. Exact-val128 and broader-val512 were skipped because the smoke gate already lost one lane TP at fixed FP. The negative checkpoint was pruned after export; retained run size is about `1.3M`.

Latest co-occurrence hard-positive sampler added `task_positive_task="cooccur:lane,stopline,crosswalk"` so training batches can draw records where lane, stop-line, and crosswalk are all positive in the same scene. Train split has `13,125` such samples and the CUDA smoke reused the existing dataset root without copying data. It is smoke-negative: fixed val4 reference lane/stop/cross was `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`; the co-occurrence checkpoint fell to lane `0.5294`, TP/FP/FN `36 / 14 / 50`, while stop-line/crosswalk stayed `0.0000 / 0.5455`. Exact-val128 and broader-val512 were skipped. The negative checkpoint/TensorBoard/root weights were pruned; retained run size is about `740K`.

Latest lane row-native primary audit made `lane_head_mode="row_native"` selectable through the training config and trained the existing row-classification lane head as the primary lane contract instead of the seg-first centerline/tangent head. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, first ran `64` train batches, then continued at larger `512` train batches for `2` epochs. It is negative: fixed val4 after the larger run was lane/stop/cross `0.0000 / 0.0000 / 0.5455`, with lane TP/FP/FN `0 / 0 / 86`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because the lane head emitted no matched lanes even after the larger training slice. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `784K`.

Latest cross-stitch task routing added an opt-in task-feature cross-stitch mixer on top of task-specific P2/P3/P4 adapters and trained it with PCGrad over the `lane_family_adapters` optimizer group. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran a real CUDA `64` train-batch smoke with skipped steps `0`. It is smoke-negative: fixed val4 epoch-2 was lane/stop/cross `0.4885 / 0.0000 / 0.5455`, with lane TP/FP/FN `32 / 13 / 54`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because lane regressed sharply and stop-line did not move. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `7.7M`.

Latest lane bidirectional seed-trace added an opt-in interior-seed trace decoder that follows learned seed logits both upward and downward from centerline-supported points, then appends non-duplicate traces after the retained row-scan/tangent lane decode. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran a real CUDA `64` train-batch smoke with skipped steps `0`. It is smoke-negative: fixed val4 epoch-2 was lane/stop/cross `0.4923 / 0.0000 / 0.4000`, with lane TP/FP/FN `32 / 12 / 54`, stop-line `0 / 3 / 2`, and crosswalk `2 / 1 / 5`. Re-evaluating the same checkpoint through the existing projection-comp runtime decode produced the same fixed val4 counts, so the loss comes from the trained dense behavior, not only the appended trace decoder. Exact-val128 and broader-val512 were skipped. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `22M`.

Latest stop-line priority retention-distill heads-only smoke added lane/cross teacher-cache retention to the stop-line-priority sampler path, using the retained merged checkpoint as the lane/cross teacher and no stop-line distill. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran a real CUDA `64` train-batch smoke with skipped steps `0`. It is smoke-negative: fixed val4 epoch-2 was lane/stop/cross `0.5373 / 0.0000 / 0.5455`, with lane TP/FP/FN `36 / 12 / 50`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because lane still regressed versus the retained fixed val4 reference and stop-line did not move. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `7.3M`.

Latest stop-line source-union projection-comp smoke added a runtime proposal source that preserves center/selector source-local candidates before projection competition, plus rowx-band selector supervision. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran a real CUDA `64` train-batch smoke with skipped steps `0`. It is smoke-negative: fixed val4 epoch-2 was lane/stop/cross `0.4885 / 0.0000 / 0.5455`, with lane TP/FP/FN `32 / 13 / 54`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because lane regressed sharply and stop-line recovered no TP. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `7.3M`.

Latest lane conditional seed-branch-only trace smoke added a freeze policy that trains only `LaneSegFirstHead.conditional_seed_logits`, then found that `requires_grad=False` alone still lets frozen BatchNorm buffers drift. The first CUDA `64` train-batch smoke reused the existing dataset root, improved the untrained seed-trace val4 from lane `0.3543` to `0.5324`, but failed to beat the row-scan baseline and its row-scan ablation regressed `38 / 14 / 48 -> 37 / 15 / 49`, exposing frozen-BN drift. The trainer now forces trunk/head eval mode for `lane_conditional_seed_only`. A rerun with that fix still failed fixed val4: lane/stop/cross `0.3409 / 0.0000 / 0.5455`, lane TP/FP/FN `30 / 60 / 56`, stop-line `0 / 3 / 2`, crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run sizes are about `24M` for the diagnostic pre-fix run and `6.1M` for the fixed rerun.

Latest stop-line priority static-trunk heads-only smoke generalized the frozen-BN lesson to ordinary lane-family heads-only fine-tuning by adding `lane_family_heads_static_trunk`: lane/stop/cross heads stay trainable, but the frozen detector/trunk runs in eval mode during train steps. The CUDA `64` train-batch smoke reused the existing dataset root and had skipped steps `0`, but fixed val4 epoch-2 was lane/stop/cross `0.4962 / 0.0000 / 0.5455`, with lane TP/FP/FN `33 / 14 / 53`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because static trunk mode did not prevent lane collapse and recovered no stop-line TP. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `7.3M`.

Latest lane conditional row rescue-append smoke kept the retained row-scan/tangent vectorizer output and appended conditional row decoder candidates with distance dedupe, instead of replacing row-scan outright. The CUDA `64` train-batch smoke reused the existing dataset root and had skipped steps `0`, but fixed val4 epoch-2 was lane/stop/cross `0.2049 / 0.0000 / 0.3636`, with lane TP/FP/FN `29 / 168 / 57`, stop-line `0 / 3 / 2`, and crosswalk `2 / 2 / 5`. Exact-val128 and broader-val512 were skipped because appending the conditional rows caused severe lane FP blow-up and crosswalk regression. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `7.3M`.

Latest stop-line static-only specialist smoke added `lane_family_stopline_static_trunk`, which trains only stop-line modules while keeping the frozen trunk and non-stop heads in eval mode. The CUDA `64` train-batch smoke reused the existing dataset root and had skipped steps `0`, but fixed val4 epoch-2 stayed at lane/stop/cross `0.5507 / 0.0000 / 0.5455`, with lane TP/FP/FN `38 / 14 / 48`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128 and broader-val512 were skipped because the static stopline-only training recovered no stop-line TP. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `6.7M`.

Latest stop-line endpoint-HAF consensus smoke added a runtime decoder that emits endpoint-pair candidates only when HAF support votes agree on the same segment. It trained the existing endpoint/HAF dense heads together, reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran a real CUDA `64` train-batch smoke with skipped steps `0`. It is smoke-negative: fixed val4 epoch-2 was lane/stop/cross `0.5038 / 0.0000 / 0.5455`, with lane TP/FP/FN `33 / 12 / 53`, stop-line `0 / 3 / 2`, and crosswalk `3 / 1 / 4`. Exact-val128, broader-val512, and larger-range training were skipped because the new candidate contract recovered no stop-line TP. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `7.6M`.

Latest lane static-only specialist smoke added `lane_family_lane_static_trunk`, which trains only the current seg-first lane head while forcing the frozen trunk and non-lane heads into eval mode. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran a real CUDA `2`-epoch by `64` train-batch smoke with skipped steps `0`. It is smoke-negative: fixed router val4 best lane variant was only `0.5263`, TP/FP/FN `35 / 12 / 51`, and the fixed flip/cross-mask variant was `0.5191`, TP/FP/FN `34 / 11 / 52`; stop-line stayed `0 / 3 / 2`, F1 `0.0000`, and crosswalk stayed `3 / 1 / 4`, F1 `0.5455`. Exact-val128, broader-val512, and larger-range training were skipped because the lane specialist regressed well below the retained fixed router lane reference. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `1.1M`.

Latest stop-line focus-crop + task-conflict negative smoke added an opt-in stop-line hard-negative loss on lane/crosswalk dense masks and combined it with the existing stop-line-centered crop/zoom feeding. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran a real CUDA `2`-epoch by `64` train-batch smoke with skipped steps `0`. It is smoke-negative: fixed router val4 with retained lane/crosswalk and the trained stop-line specialist was lane/stop/cross `0.5839 / 0.0000 / 0.5455`, with stop-line TP/FP/FN `0 / 4 / 2`; the retained primary stop-line route on the same slice was `0 / 3 / 2`. Exact-val128, broader-val512, and larger-range training were skipped because the new FP-control did not recover any stop-line TP and added one FP on smoke. Negative checkpoints, TensorBoard, and root `yolo26s.pt` were pruned; retained run size is about `1.2M`.

Latest stop/cross static-trunk lane-frozen smoke added `lane_family_stop_cross_static_trunk` to keep the frozen detector trunk and lane head in eval mode while training only stop-line/crosswalk heads. It reused the existing dataset root, indexed `429350` records, and ran CUDA `2x64` train-batch smoke with skipped steps `0`. Static mode did protect lane on fixed val4, but the stop-line specialist recovered no smoke TP and exact-val128 was negative: retained primary stop-line was `30 / 30 / 30`, F1 `0.5000`, while the static specialist was `30 / 32 / 30`, F1 `0.4918`. Broader-val512 and larger training were skipped. Negative checkpoints, TensorBoard, and root weights were pruned; retained run size is about `2.4M`.

Latest stop-line context segment-set smoke added an opt-in interacting seed-query segment decoder on top of the stop-line dense head. It reused the existing dataset root, indexed `429350` records, and ran CUDA `2x64` train-batch smoke with skipped steps `0`; no dataset copy was created. Fixed val4 failed to recover any stop-line TP (`0 / 3 / 2`, F1 `0.0000`), and exact-val128 confirmed the branch as negative: lane/stop/cross `0.5447 / 0.3898 / 0.5868`, with stop-line TP/FP/FN `23 / 35 / 37`. This is below baseline exact `0.4483` and projection-competition exact `0.5167`, so broader-val512 and larger-range training were skipped. Negative checkpoints, TensorBoard, and root weights were pruned after export.

Latest stop-line distance-heatmap target changed the stop-line dense center/selector supervision from midpoint-centered positives to a full-segment Gaussian distance heatmap. It reused the existing `pv26_exhaustive_od_lane_dataset` in place, indexed `429350` records, and ran both CUDA `2x64` smoke and a larger `3x512` train-batch scale audit. The `2x64` fixed exact-val128 result was lane/stop/cross `0.5427 / 0.3919 / 0.5868`, stop-line TP/FP/FN `29 / 59 / 31`. The larger `3x512` run raised internal phase objective to `0.6357`, but fixed exact-val128 was still worse for stop-line: lane/stop/cross `0.5602 / 0.3522 / 0.6071`, stop-line TP/FP/FN `28 / 71 / 32`. Broader-val512 was skipped because exact stayed below both baseline exact `0.4483` and projection-competition exact `0.5167`. Negative checkpoints and TensorBoard outputs were pruned; retained run sizes are about `16M` for the smoke and `11M` for the scale audit.

Latest lane feature-ROI bounded-residual repair probe reopened the earlier learned repair family with a TP-preserving confidence contract: repair labels require a candidate to be outside the lane metric threshold but near GT, already matched candidates are do-not-repair negatives, the MLP predicts bounded residuals around the original polyline, negatives get identity geometry loss, and runtime selection uses quality plus mean-move gates. It reused the existing dataset root in place and ran CUDA val4 smoke plus a larger train64 candidate-collection slice. The final train64 smoke still did not move the metric: baseline and repaired lane/stop/cross stayed `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`, with only `2` selected repairs from `664` train examples. Exact-val128 and broader-val512 were skipped because the fixed smoke gate had no positive TP/FP/FN movement. Negative repair weights and root `yolo26s.pt` were pruned; retained replay artifacts are CSV/summary only, and the probe now saves repair weights only with `--save-repair-model`.

Latest lane area-ROI verifier stress audit trained a no-GT MLP over raw row-scan/tangent lane candidates that the default bbox/area filters drop, then added chunked eval-only replay so broader validation can be split without copying the dataset or holding all raw batches in memory. It reused the existing canonical dataset root in place, trained on `384` train batches, saved a `432K` verifier model, and evaluated exact-val128 plus a four-chunk val512 aggregate. Exact-val128 improved lane `0.5888 -> 0.5956`, TP/FP/FN `1202 / 491 / 1188 -> 1249 / 555 / 1141`, but still missed `0.60` and added FP faster than TP. The chunked val512 aggregate moved lane `0.5749 -> 0.5821`, TP/FP/FN `4634 / 1983 / 4871 -> 4816 / 2227 / 4689`; stop-line stayed `0.5252`, `125 / 100 / 126`, and crosswalk stayed `0.5969`, `228 / 136 / 172`. This is learned runtime replay evidence, but not a production gate: broader TP gain `+182` came with FP `+244`, and crosswalk is also below `0.60` on this stress slice. Retained artifacts are CSV/summary plus the small verifier model; no dataset copy or large checkpoint artifact was created.

Latest lane area-ROI alignment-context verifier added a no-GT nearest-retained-lane geometry context to that same dropped-candidate MLP: nearest retained lane distance, center delta, y-overlap, angle error, length ratios, and retained-lane count. It reused the existing canonical dataset root in place, indexed `429350` records, trained on `64` verifier train batches, and wrote only a `36K` CSV/summary artifact. Fixed val4 smoke is negative: baseline lane `0.5839`, TP/FP/FN `40 / 11 / 46`, moved to `0.5816`, `41 / 14 / 45`; selected candidates `4`, selected oracle-positive `1`. Stop-line and crosswalk stayed `0.0000 / 0.5455`. Exact-val128 and broader-val512 were skipped because the smoke gate lost lane F1 and FP grew faster than TP.

Latest lane area-ROI side-contrast verifier added a no-GT dense ridge-vs-side-band signal to the dropped-candidate verifier: for each raw lane candidate it samples center/support values on the candidate and side offsets, plus tangent alignment. It reused the existing canonical dataset root in place, indexed `429350` records, and created no dataset copy. Smoke train64/val4 was weak-positive (`0.5839 -> 0.5874`, TP/FP/FN `40 / 11 / 46 -> 42 / 15 / 44`). Larger train256 exact-val128 improved lane `0.5888 -> 0.5950`, TP/FP/FN `1202 / 491 / 1188 -> 1243 / 545 / 1147`, while stop-line/crosswalk stayed `0.5333 / 0.5988`. This is still below `0.60` and slightly below the prior plain train384 area-ROI exact `0.5956`; broader-val512 was skipped because FP still grew faster than TP and the exact gate did not improve the retained frontier. Retained side-contrast exports are CSV/summary-only and about `36K` smoke / `152K` exact.

Latest lane area-ROI replace-nearest integration changed the verifier replay contract from append to fixed-count replacement: verifier-positive raw dropped candidates replace their nearest retained lane within a bounded distance, so prediction count does not grow. It reused the existing canonical dataset root in place, indexed `429350` records, trained a real train64 verifier smoke, and then replayed the prior saved train384 verifier under the same replacement contract. Both fixed val4 gates were negative. The train64 replacement run selected `1` candidate with `0` oracle-positive and regressed lane `0.5839 -> 0.5693`, TP/FP/FN `40 / 11 / 46 -> 39 / 12 / 47`. The train384 replay selected `1` non-oracle-positive candidate and left lane flat at `0.5839`, TP/FP/FN `40 / 11 / 46`. Exact-val128 and broader-val512 were skipped. The failed train64 verifier checkpoint was pruned; retained artifacts are only CSV/summary, about `36K` and `16K`, and no dataset copy was created.

Latest stop-line train-split raw-patch CNN verifier moved beyond the prior validation-half replay by training candidate selection on canonical train batches and replaying one train-selected threshold on exact validation. It reused the existing canonical dataset root in place, trained on `256` train batches, and evaluated exact-val128 after a val4 smoke. The train split overfit strongly: stop-line `0.6000 -> 0.7408`, TP/FP/FN `351 / 168 / 300 -> 383 / 0 / 268`. Validation moved the wrong way: exact-val128 baseline/projection-comp `0.5333`, TP/FP/FN `32 / 28 / 28`, fell to `0.4160`, `26 / 39 / 34`. Broader-val512 was skipped. Retained artifacts are CSV/summary only (`4.9M` smoke, `22M` exact); no dataset copy or checkpoint artifact was created.

Latest raw-image Hough stop-line candidate-generation probe generated Canny/Hough line candidates from the existing raw images, scored them with dense stop-line maps, trained a small MLP verifier on `64` canonical train batches, and replayed it on fixed val4. It reused the existing `429350` record dataset root in place and wrote only CSV/summary artifacts. This was smoke-negative: train baseline-plus-Hough moved stop-line `87 / 28 / 64 -> 93 / 144 / 58`, F1 `0.6541 -> 0.4794`, and val4 had no oracle-positive Hough candidates while FP increased `3 -> 8`. Exact-val128 and broader-val512 were skipped. Artifact size is about `3.2M`; no dataset copy or checkpoint artifact was created.

Latest raw-image LSD stop-line candidate-generation probe changed the raw candidate generator from Canny/Hough to OpenCV LSD line segments while keeping the same dense-map feature scoring and train-split MLP verifier replay. It reused the existing canonical dataset root, indexed `429350` records, and created no checkpoint or dataset copy. It is negative at both fixed gates. Val4 had `128` LSD candidates with `0` oracle positives, and raw-LSD replay only increased stop-line FP `3 -> 6`. Exact-val128 did contain `33` oracle-positive LSD candidates out of `4096`, but baseline-plus-raw-LSD kept TP fixed at `32` while FP jumped `28 -> 108`, dropping stop-line F1 `0.5333 -> 0.3200`. Broader-val512 was skipped. Retained artifact sizes are about `3.7M` smoke and `11M` exact.

Latest dense-support raw stop-line candidate-generation probe changed the raw candidate contract again: instead of free Canny/Hough or LSD lines, it restricted raw brightness/edge evidence by predicted stop-line mask/proposal support, fitted connected components with PCA, and trained the same train-split MLP verifier. It reused the existing canonical dataset root directly, indexed `429350` records, and created no checkpoint or dataset copy. The branch is exact-negative. Val4 had only `5` validation candidates with `0` oracle positives. Exact-val128 had `126` validation candidates with only `4` oracle positives; raw-support-PCA-only emitted `0 / 16 / 60`, and baseline-plus-raw-support-PCA kept TP fixed at `32` while FP rose `28 -> 41`, dropping stop-line F1 `0.5333 -> 0.4812`. Broader-val512 was skipped. Retained artifact sizes are about `256K` smoke and `368K` exact.

Latest row-native lane quality/dynamic assignment train exposed the existing row-native objectness-quality knobs through config/CLI and trained a real heads-only CUDA `2x64` smoke checkpoint with `lane_head_mode=row_native`, `lane_assignment_mode=dynamic_match`, and `lane_objectness_target_mode=quality_ramp`. It reused the existing canonical dataset root directly, indexed `429350` records, had skipped steps `0`, and created no dataset copy. It is smoke-negative: fixed val4 epoch-2 best reached lane/stop/cross `0.0000 / 0.0000 / 0.5000`, lane TP/FP/FN `0 / 0 / 86`, stop-line `0 / 3 / 2`, and crosswalk `3 / 2 / 4`. Exact-val128, broader-val512, and larger training were skipped because the first fixed gate recovered zero lane TP. Negative checkpoints and TensorBoard were pruned; retained run size is about `8.2M`.

Active goal:

- broader validation에서 lane / stop-line / crosswalk F1이 모두 `>= 0.60`인 checkpoint + postprocess/preprocess/runtime contract를 만든다.
- exact epoch-2 subset이나 `phase_objective` 단독 통과는 중간 신호일 뿐 최종 성공으로 보지 않는다.
- 실험은 branch/worktree 단위로 분리하고, 한 worktree는 한 축만 바꾼다.

## 2. 현재 기준 artifact

Run:

`runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412`

남긴 핵심 파일:

- checkpoint: `phase_4/checkpoints/best.pt`
- retained lane-preserving composite metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_task_mask_context_val512_epoch2/metrics.csv`
- latest stop/cross lane-frozen composite metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_cross_priority_lane_frozen_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_105725/analysis_exports/full_runtime_task_balance_val512_epoch2/metrics.csv`
- latest stop/cross lane-frozen scale2048 audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_cross_priority_lane_frozen_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_114452/analysis_exports/full_runtime_task_balance_val512_epoch2/metrics.csv`
- latest lane seed-trace instance decoder exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_trace_instance_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_131325/analysis_exports/lane_seed_trace_exact_val128_epoch2/metrics.csv`
- latest input-scale672 dense-target broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_projection_comp_runtime_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_133550/analysis_exports/input_scale672_broader_val512_epoch2/metrics.csv`
- latest stop-line axis-distance field broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_axis_distance_field_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_142901/analysis_exports/axis_distance_broader_val512_best/metrics.csv`
- latest lane center-offset field broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_center_offset_field_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_151708/analysis_exports/broader_val512_epoch2/metrics.csv`
- latest stop-line focus-crop feeding broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_focus_crop_feeding_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_155424/analysis_exports/broader_val512_epoch2/metrics.csv`
- latest retention-distill upper-trunk broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_retention_distill_upper_trunk_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_162533/analysis_exports/broader_val512_epoch2/metrics.csv`
- latest two-checkpoint stop-line-priority router broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/router_stopline_priority_broader_val512_epoch2/metrics.csv`
- latest stop-line-only upper-trunk router specialist broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_router_specialist_upper_trunk_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_171508/analysis_exports/router_broader_val512_epoch2/metrics.csv`
- latest lane-router specialist smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_router_specialist_smoke_val4_epoch2/metrics.csv`
- latest stop-line dual-source arbitration broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_dual_source_arbitration_broader_val512_epoch2/metrics.csv`
- latest V3 stop-line isolated-neck exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_v3_isolated_neck_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_183820/analysis_exports/router_exact_val128_epoch2/metrics.csv`
- latest stop-line midpoint proposal exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_midpoint_projection_comp_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_191031/analysis_exports/router_exact_val128_epoch2/metrics.csv`
- latest heterogeneous endpoint-pair router exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/heterogeneous_endpoint_pair_router_exact_val128_epoch2/metrics.csv`
- latest stop-line-priority positive-sampler scale2048 router exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_194249/analysis_exports/router_exact_val128_epoch2/metrics.csv`
- latest stop-line-only mask-first specialist exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_only_mask_first_specialist_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_205917/analysis_exports/router_exact_val128_epoch2/metrics.csv`
- latest task-routed multi-teacher distill student exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_task_routed_distill_student_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_212815/analysis_exports/single_checkpoint_exact_val128_epoch2/metrics.csv`
- latest stop-line local 2D patch segment head exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_local_patch_segment_head_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_221343/analysis_exports/patch_segment_exact_val128_epoch2/metrics.csv`
- latest stop-line raw-patch verifier exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_raw_patch_verifier_exact_val128_epoch2/raw_patch_verifier_variants.csv`
- latest stop-line raw-patch geometry-repair exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_raw_patch_geometry_repair_exact_val128_epoch2/raw_patch_geometry_repair_variants.csv`
- latest stop-line raw-patch CNN verifier exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_raw_patch_cnn_verifier_exact_val128_epoch2/raw_patch_cnn_verifier_variants.csv`
- latest stop-line det-source hard-negative feeding fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_det_negative_feeding_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_113604/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest task-loss EMA balancer exact audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_task_loss_ema_balancer_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_231130/analysis_exports/exact_val128_epoch2/metrics.csv`
- latest task-loss EMA balancer broader audit metrics: `runs/pv26_exhaustive_od_lane_train/lane60_task_loss_ema_balancer_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_231130/analysis_exports/broader_val512_epoch2/metrics.csv`
- latest lane-only seg-first reference router smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_only_segfirst_reference_router_smoke_val4_epoch2/metrics.csv`
- latest lane-only seg-first specialist router smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_only_segfirst_specialist_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_235106/analysis_exports/lane_router_smoke_val4_epoch2/metrics.csv`
- latest co-occurrence hard-positive sampler smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_cooccur_lane_stop_cross_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_000717/analysis_exports/cooccur_sampler_smoke_val4_epoch2/metrics.csv`
- latest task-rotation sampler fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_task_rotation_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_064823/analysis_exports/fixed_val4_epoch3_best/metrics.csv`
- latest stop-line patch projection-merge fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_patch_projection_merge_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_070405/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest task-uncertainty loss balancer fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_task_uncertainty_loss_balancer_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_072355/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest lane instance-embedding auxiliary fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_instance_embedding_aux_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_074321/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest shared affine train augmentation fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_shared_affine_augmentation_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_075616/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest synthetic stop-line injection fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_synthetic_stopline_injection_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_081241/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest real stop-line copy-paste feeding fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_real_stopline_copy_paste_feeding_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_083032/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest stop-line positive-only empty-sample loss fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_positive_only_loss_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_084536/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest current-family vector decoder val128-style training history: `runs/pv26_exhaustive_od_lane_train/lane60_current_family_vector_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_090008/phase_4/history/epochs.jsonl`
- latest full-trunk retention-distill fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_retention_distill_full_trunk_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_092447/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest current-family sigmoid vector decoder fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_current_family_sigmoid_vector_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_094354/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest current-family anchor query prior fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_current_family_anchor_query_prior_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_095757/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest current-family anchor denoise query auxiliary fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_current_family_anchor_denoise_vector_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_105921/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest current-family anchor query-seed decoder fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_current_family_anchor_query_seed_vector_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_163252/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest lane seed-relative row decoder fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_relative_row_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_101221/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest lane conditional row branch-only fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_conditional_row_branch_only_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_102541/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest lane conditional row dense-gate fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_conditional_row_dense_gate_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_104105/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest lane row-native primary scale-smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_row_native_primary_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_002505/analysis_exports/row_native_primary_scale_smoke_val4_epoch2/metrics.csv`
- latest lane row-native quality/dynamic assignment fixed val4 metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_row_native_quality_dynamic_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_173120/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest cross-stitch task routing smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_cross_stitch_task_routing_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_010213/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest lane bidirectional seed-trace smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_bidirectional_seed_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_011614/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest stop-line priority retention-distill heads-only smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_retention_distill_heads_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_013211/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest stop-line source-union projection-comp smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_source_union_projection_comp_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_014626/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest lane conditional seed-branch-only trace diagnostic smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_020030/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest lane conditional seed-branch-only trace fixed-BN smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_021142/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest stop-line priority static-trunk heads-only smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_static_trunk_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_022254/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest lane conditional row rescue-append smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_conditional_row_rescue_append_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_023911/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest stop-line static-only specialist smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_static_only_specialist_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_025025/analysis_exports/fixed_val4_epoch2/metrics.csv`
- latest stop-line endpoint-HAF consensus smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_endpoint_haf_consensus_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_030122/analysis_exports/fixed_val4_epoch2_best/metrics.csv`
- latest lane static-only specialist router smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_static_only_specialist_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_033045/analysis_exports/router_val4_epoch2_best/metrics.csv`
- latest stop-line focus-crop conflict-negative router smoke metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_focus_crop_conflict_negative_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_034652/analysis_exports/router_val4_epoch2_best/metrics.csv`
- latest stop/cross static-trunk lane-frozen exact metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_cross_priority_static_trunk_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_040136/analysis_exports/router_exact_val128_epoch2_best/metrics.csv`
- latest stop-line context segment-set exact metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_context_segment_set_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_042145/analysis_exports/context_segment_eval_val128_epoch2/metrics.csv`
- latest stop-line distance-heatmap target smoke exact metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_distance_heatmap_target_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_043625/analysis_exports/distance_heatmap_val128_epoch2/metrics.csv`
- latest stop-line distance-heatmap target scale exact metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_distance_heatmap_target_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_044418/analysis_exports/distance_heatmap_val128_epoch2/metrics.csv`
- latest lane feature-ROI bounded-residual repair train64 smoke summary: `runs/pv26_exhaustive_od_lane_train/lane_feature_roi_repair_train64_smoke_val4_20260530_02/summary.json`
- latest lane area-ROI verifier smoke summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train64_smoke_val4_20260530/summary.json`
- latest lane area-ROI verifier exact summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train256_exact_val128_20260530/summary.json`
- latest lane area-ROI verifier train384 exact summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train384_exact_val128_20260530/summary.json`
- latest lane area-ROI verifier train384 val512 aggregate summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train384_broader_val512_chunked_aggregate_20260530/summary.json`
- latest lane area-ROI verifier saved model: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train384_model_val4_20260530/verifier.pt`
- latest lane area-ROI alignment-context smoke summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_alignment_context_train64_smoke_val4_20260530/summary.json`
- latest lane area-ROI side-contrast smoke summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_side_contrast_train64_smoke_val4_20260530/summary.json`
- latest lane area-ROI side-contrast exact summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_side_contrast_train256_exact_val128_20260530/summary.json`
- latest lane area-ROI replace-nearest train64 smoke summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_replace_nearest_train64_smoke_val4_20260530/summary.json`
- latest lane area-ROI replace-nearest train384 replay smoke summary: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_replace_nearest_train384_replay_smoke_val4_20260530/summary.json`
- latest stop-line trainset raw-patch CNN verifier smoke summary: `runs/pv26_exhaustive_od_lane_train/stopline_trainset_patch_verifier_train64_smoke_val4_20260530_02/summary.json`
- latest stop-line trainset raw-patch CNN verifier exact summary: `runs/pv26_exhaustive_od_lane_train/stopline_trainset_patch_verifier_train256_exact_val128_20260530/summary.json`
- latest stop-line raw-Hough candidate-generation smoke summary: `runs/pv26_exhaustive_od_lane_train/stopline_raw_hough_candidates_train64_smoke_val4_20260530/summary.json`
- latest stop-line raw-LSD candidate-generation smoke summary: `runs/pv26_exhaustive_od_lane_train/stopline_raw_lsd_candidates_train64_smoke_val4_20260530/summary.json`
- latest stop-line raw-LSD candidate-generation exact summary: `runs/pv26_exhaustive_od_lane_train/stopline_raw_lsd_candidates_train64_exact_val128_20260530/summary.json`
- latest stop-line raw-support-PCA candidate-generation smoke summary: `runs/pv26_exhaustive_od_lane_train/stopline_raw_support_pca_candidates_train64_smoke_val4_20260530/summary.json`
- latest stop-line raw-support-PCA candidate-generation exact summary: `runs/pv26_exhaustive_od_lane_train/stopline_raw_support_pca_candidates_train64_exact_val128_20260530/summary.json`
- latest stop-line learned source-router exact metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_source_router_exact_val128_epoch2/metrics.csv`
- latest stop-line dense-aligned source-router exact metrics: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_dense_quality_router_exact_val128_epoch2/metrics.csv`
- latest stop-line dense-aligned source-router exact summary: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_dense_quality_router_exact_val128_epoch2/summary.json`
- latest stop-line raster source-router smoke summary: `runs/pv26_exhaustive_od_lane_train/stopline_raster_source_router_train64_smoke_val4_20260530/summary.json`
- latest stop-line raster source-router exact train64 summary: `runs/pv26_exhaustive_od_lane_train/stopline_raster_source_router_train64_exact_val128_20260530/summary.json`
- latest stop-line raster source-router exact train256 summary: `runs/pv26_exhaustive_od_lane_train/stopline_raster_source_router_train256_exact_val128_20260530/summary.json`
- previous stop-line-exposure composite metrics: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_101745/analysis_exports/full_runtime_task_balance_val512_epoch2/metrics.csv`
- retained exact stop-line lane-extent probe: `analysis_exports/stopline_lane_extent_readout_val128_epoch2/variants.csv`

Note: older exact-eval and visual-check exports were pruned from active `runs` during artifact cleanup. The numeric results below remain as historical evidence, but those old export directories are no longer current retained artifacts.

Final exact epoch-2 result:

| Metric | Value |
| --- | ---: |
| objective | `0.6088677363246272` |
| lane F1 | `0.526695898890895` |
| stop-line F1 | `0.4482758620689655` |
| crosswalk F1 | `0.5853658536585366` |
| support lane/stop/cross | `2390 / 60 / 81` |

Current useful broader runtime evidence 기준 gap:

- lane: `0.5628 -> 0.6000`, `+0.0372` 필요.
- stop-line: `0.5309 -> 0.6000`, `+0.0691` 필요 on the current two-checkpoint stop-line-priority router, which preserves retained lane/crosswalk by routing only stop-line outputs from the specialist checkpoint.
- crosswalk: `0.6219` on the stop-line-priority trained composite and `0.6187` on the retained lane-preserving composite, both broader-val512 pass.
- 따라서 F1 0.6+ 목표의 병목은 stop-line과 lane의 동시 향상이다. The best stop-line value is now a two-checkpoint runtime contract, not a single raw checkpoint default. Crosswalk는 hull decode로 broader pass를 만들었지만, 이 자체는 opt-in postprocess partial-positive이고 lane/stop-line 실패를 가리지 않는다.

Gate 4 exact crosswalk threshold candidate:

- artifact: `analysis_exports/crosswalk_postprocess_thresholds_val128_epoch2/summary.json`
- top objective variant: `lane_obj_0.35__cross_mask_0.40__cross_area_32`
- objective: `0.6115270642619861`
- lane / stop-line / crosswalk F1: `0.5326 / 0.4483 / 0.6027`
- crosswalk TP/FP/FN: `44 / 21 / 37`
- broader-val512 crosswalk-only replay: baseline lane/stop/cross F1 `0.5101 / 0.4083 / 0.5854`, candidate `0.5101 / 0.4083 / 0.5845`.
- 판단: exact val128에서는 crosswalk gap을 닫았지만 broader-val512에서 유지되지 않았다. `cross_mask=0.40`, `cross_area=32` threshold tightening은 Gate 4 success가 아니다.

Gate 4 broader crosswalk shape sweep:

- artifact: `analysis_exports/crosswalk_broader_shape_sweep_val512_epoch2/summary.json`
- variants: `64` crosswalk-only object/mask/component-area/polygon-area/aspect/top-k variants.
- best crosswalk F1 variant: `cross_aspect_2.0`, lane/stop/cross F1 `0.5101 / 0.4083 / 0.5960`, cross TP/FP/FN `239 / 168 / 156`.
- best objective variant: `cross_mask_0.70`, objective `0.5970434363`, lane/stop/cross F1 `0.5101 / 0.4083 / 0.5887`.
- 판단: broader-val512에서 crosswalk F1 `>=0.60` variant가 `0/64`다. 단순 crosswalk threshold/shape postprocess는 Gate 4 success path가 아니다.

Current crosswalk hull decode replay:

- artifact exact: `analysis_exports/crosswalk_hull_decode_val128_epoch2/summary.json`
- artifact broader: `analysis_exports/crosswalk_hull_decode_val512_epoch2/summary.json`
- opt-in postprocess: `crosswalk_polygon_mode=hull`
- exact val128 hull result: objective `0.6246557098`, lane/stop/cross F1 `0.5633 / 0.4483 / 0.5988`.
- broader-val512 hull result: objective `0.6127256636`, lane/stop/cross F1 `0.5407 / 0.4083 / 0.6187`, cross TP/FP/FN `232 / 123 / 163`.
- 판단: rectangle/aspect threshold sweep은 실패했지만, minimum-area rectangle 대신 convex hull을 쓰는 representation-aware decode는 broader-val512 crosswalk gap을 닫았다. 하지만 lane과 stop-line은 여전히 목표 미달이라 all-task success/default가 아니다.

Broader-val512 replay result:

- artifact: `analysis_exports/broader_val512_final_geometry_filters_epoch2/summary.json`
- objective: `0.5943438312141003`
- lane / stop-line / crosswalk F1: `0.5101 / 0.4083 / 0.5854`
- support lane / stop / cross: `9477 / 271 / 395`
- 판단: exact epoch-2의 geometry-filter gain은 더 넓은 slice에서도 완전히 사라지지는 않았지만, objective 0.6과 task별 F1 0.6 목표에는 미달이다.

Current best broader lane replay:

- artifact: `analysis_exports/lane_row_scan_tangent_link_val512_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_link`
- objective: `0.6027206157496527`
- lane / stop-line / crosswalk F1: `0.5407 / 0.4083 / 0.5854`
- support lane / stop / cross: `9477 / 271 / 395`
- 판단: row-scan보다 lane continuity와 objective는 올랐지만 stop-line과 crosswalk가 목표 미달이라 success/default가 아니다.

Current best broader runtime/postprocess composite by objective:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_task_mask_context_val512_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_link`
- checkpoint composition: original `best.pt` as base, lane head from segment-MIL lane-head-only `best_lane.pt`, stop-line and crosswalk heads from original `best.pt`.
- runtime/postprocess: average only `lane_seg_centerline_logits` from the normal image and horizontal-flip image, then suppress lane centerline probability by `0.50 * crosswalk_mask_probability`; keep stop-line/crosswalk outputs from the normal pass.
- evaluator-only overrides: stop-line `mask=0.80`, stop-line `min_instance_score=0.94`, stop-line `presence=0.0`, crosswalk `polygon_mode=hull`.
- objective: `0.6230558330631257`
- lane / stop-line / crosswalk F1: `0.5628 / 0.4235 / 0.6187`
- TP/FP/FN lane: `4532 / 2097 / 4945`
- TP/FP/FN stop-line: `101 / 105 / 170`
- TP/FP/FN crosswalk: `232 / 123 / 163`
- support lane / stop / cross: `9477 / 271 / 395`
- 판단: crosswalk-mask lane competition recovers a real but small broader lane gain over the previous flip-centerline composite (`0.5577 -> 0.5628`) while preserving stop-line `0.4235` and hull crosswalk `0.6187`. This is a new objective best but still not all-task success because lane and stop-line remain below `0.60`.

Latest lane composite replay tooling status:

- Active `develop` restores the minimal replay path for this composite: `row_scan_tangent` lane vectorization, evaluator-only postprocess overrides, and `tools/probe_pv26_lane_flip_tta.py`.
- The restored flip probe intentionally exposes only `baseline`, `flip_centerline_avg`, and fixed `flip_centerline_avg_lane_cross_comp050`; it is not a new sweep surface for closed TTA/task-mask/vectorizer variants.
- Focused tests pass, and a one-batch CUDA smoke on the retained merged checkpoint writes `metrics.csv` / `summary.json`. This is reproducibility status only, not F1 progress.

Current retained lane-preserving broader task-balance runtime composite:

- branch/worktree: `exp/lane-family-f1/stopline-projcomp-runtime-contract`.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/full_runtime_task_balance_exact_val128_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/full_runtime_task_balance_val512_epoch2/summary.json`
- runtime contract: run `tools/probe_pv26_lane_flip_tta.py` with fixed `flip_centerline_avg_lane_cross_comp050`, `crosswalk_polygon_mode=hull`, and `--lane60-experiment stopline_projection_comp_runtime`. This keeps the retained lane flip/crosswalk-mask path and uses the opt-in projection-competition stop-line runtime decoder instead of CSV replay.
- exact-val128 lane / stop-line / crosswalk F1: `0.5888 / 0.5333 / 0.5988`.
- exact TP/FP/FN lane: `1202 / 491 / 1188`; stop-line: `32 / 28 / 28`; crosswalk: `50 / 36 / 31`.
- broader-val512 lane / stop-line / crosswalk F1: `0.5628 / 0.5164 / 0.6187`.
- broader TP/FP/FN lane: `4532 / 2097 / 4945`; stop-line: `126 / 91 / 145`; crosswalk: `232 / 123 / 163`.
- lane-family mean/min F1: `0.5659 / 0.5164`.
- 판단: this is no longer artifact-only CSV recombination; the known lane-preserving task-balance lower bound now has an opt-in runtime/evaluator contract. It still fails all-task `0.60`: lane needs `+0.0372` and stop-line needs `+0.0836`, and it is not a single raw checkpoint default.

Latest two-checkpoint stop-line router audit:

- branch/worktree: `exp/lane-family-f1/runtime-stopline-specialist-router`.
- implementation: `tools/probe_pv26_lane_flip_tta.py` can now load a primary checkpoint and a separate `--stop-line-checkpoint`; the probe replaces only `stop_line` and `stop_line_*` outputs from the specialist pass, while lane/crosswalk stay on the retained primary checkpoint. This is a real runtime/evaluator contract, not CSV recombination, but still not a single raw checkpoint default.
- primary checkpoint: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/merged_lane_head.pt`.
- stop-line-priority specialist checkpoint: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_101745/phase_4/checkpoints/best.pt`.
- stop-line-priority router exact-val128 epoch-2: lane/stop/cross `0.5888 / 0.4918 / 0.5988`, TP/FP/FN lane `1202 / 491 / 1188`, stop-line `30 / 32 / 30`, crosswalk `50 / 36 / 31`.
- stop-line-priority router broader-val512 epoch-2: lane/stop/cross `0.5628 / 0.5309 / 0.6187`, TP/FP/FN lane `4532 / 2097 / 4945`, stop-line `133 / 97 / 138`, crosswalk `232 / 123 / 163`, lane-family mean/min `0.5708 / 0.5309`.
- judgment: this is the current best broader all-task runtime lower bound because it keeps retained lane/crosswalk while improving stop-line over projection-comp runtime (`0.5164 -> 0.5309`). It still fails the actual target: lane needs `+0.0372`, stop-line needs `+0.0691`, and the runtime has to carry two checkpoints unless a single-checkpoint contract catches up.

Latest stop-line-only upper-trunk router specialist:

- branch/worktree: `exp/lane-family-f1/runtime-stopline-specialist-router`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_router_specialist_upper_trunk_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_171508`.
- changed training axis: train a stop-line specialist for the two-checkpoint router, with freeze policy `lane_family_plus_upper_trunk`, trunk LR `2e-6`, head LR `2e-4`, loss weights det/TL/lane/crosswalk `0`, stop-line `4.0`, task-positive sampling `stopline`, projection-competition runtime decoder, and hull crosswalk retained for eval.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. After eval, the smoke run, duplicate task-best/last checkpoints, TensorBoard, and temporary root `yolo26s.pt` were removed; retained main run size is about `122M`.
- training: real CUDA smoke `1` epoch, `8` train batches, `4` val batches; real CUDA main `3` epochs, `512` train batches, `128` val batches, skipped steps `0`. Internal best phase-objective was epoch `3`, `0.6543`; this is not final evidence because final judgement uses fixed router eval.
- fixed router exact-val128 epoch-2: lane/stop/cross `0.5888 / 0.4427 / 0.5988`, TP/FP/FN lane `1202 / 491 / 1188`, stop-line `29 / 42 / 31`, crosswalk `50 / 36 / 31`.
- fixed router broader-val512 epoch-2: lane/stop/cross `0.5628 / 0.4826 / 0.6187`, TP/FP/FN lane `4532 / 2097 / 4945`, stop-line `125 / 122 / 146`, crosswalk `232 / 123 / 163`.
- judgment: the router mechanism is useful, but this new stop-line-only upper-trunk specialist is negative. It preserves lane/crosswalk by construction, but stop-line is worse than both the stop-line-priority router (`0.5309`) and projection-comp retained runtime (`0.5164`). Do not repeat this as trunk LR, head LR, stop-line loss weight, epoch-count, or stopline-only sampler tuning without a new candidate/geometry signal.

Latest lane-router specialist upper-trunk smoke:

- branch/worktree: `exp/lane-family-f1/lane-router-specialist-upper-trunk`.
- implementation: `tools/probe_pv26_lane_flip_tta.py` can now also load a separate `--lane-checkpoint`; `_merge_lane_outputs()` replaces only `lane` and `lane_*` outputs from the lane-specialist pass. The stop-line router can still replace only `stop_line` and `stop_line_*`, so the probe can test lane from one checkpoint, stop-line from another, and retained crosswalk from the primary checkpoint.
- training preset: `lane_router_specialist_upper_trunk` in `tools/run_pv26_lane60_probe.py`, with freeze policy `lane_family_plus_upper_trunk`, trunk LR `2e-6`, head LR `2e-4`, loss weights det/TL/stop-line/crosswalk `0`, lane `4.0`, and task-positive sampling `lane`.
- route baseline smoke: primary checkpoint also used as `--lane-checkpoint`, stop-line-priority checkpoint used as `--stop-line-checkpoint`, fixed `flip_centerline_avg_lane_cross_comp050`, val4 epoch2. Lane/stop/cross F1 was `0.5839 / 0.0000 / 0.5455`, TP/FP/FN lane `40 / 11 / 46`, stop-line `0 / 3 / 2`, crosswalk `3 / 1 / 4`. This proves the lane route plumbing preserves the existing lane smoke baseline.
- specialist smoke train: real CUDA `1` epoch, `8` train batches, `4` val batches, batch size `4`, skipped steps `0`, using the existing `/home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset` root directly with no dataset copy.
- trained-specialist fixed router smoke: lane/stop/cross F1 was `0.5735 / 0.0000 / 0.5455`, TP/FP/FN lane `39 / 11 / 47`, stop-line `0 / 3 / 2`, crosswalk `3 / 1 / 4`.
- storage: the transient smoke training run and temporary root `yolo26s.pt` were pruned. Only the small smoke metric exports were retained under the primary source-run `analysis_exports`.
- judgment: the lane route extension is useful plumbing, but this lane-only upper-trunk specialist is smoke-negative. It loses one lane TP at fixed FP on the router gate, so main training, exact-val128, and broader-val512 were skipped. Do not repeat this as trunk LR, head LR, lane loss weight, epoch-count, or lane-only sampler tuning without a materially different lane instance/geometry signal.

Latest stop-line dual-source arbitration:

- branch/worktree: `exp/lane-family-f1/stopline-dual-source-arbitration`.
- implementation: `tools/probe_pv26_lane_flip_tta.py` can now evaluate final stop-line source modes via `--stop-line-source-modes`. The added modes combine postprocessed stop-line predictions from the retained primary checkpoint and the stop-line-priority specialist checkpoint after lane/crosswalk have already been preserved.
- smoke val4: all source modes were identical on stop-line (`0 / 3 / 2`, F1 `0.0000`) because the slice has only `2` stop-line GT rows; lane/crosswalk stayed `0.5839 / 0.5455`.
- exact-val128 epoch-2: `primary_absent_specialist` was the best mode and improved stop-line over the primary projection-comp source from `32 / 28 / 28`, F1 `0.5333`, to `33 / 28 / 27`, F1 `0.5455`. Other modes were lower: `union_dedupe` `32 / 31 / 28`, F1 `0.5203`; `agreement` `30 / 29 / 30`, F1 `0.5042`; specialist `30 / 32 / 30`, F1 `0.4918`.
- broader-val512 epoch-2: the exact gain did not hold. Specialist-only remains the best broader source at `133 / 97 / 138`, F1 `0.5309`; primary is `126 / 91 / 145`, F1 `0.5164`; `primary_absent_specialist` is worse at `129 / 103 / 142`, F1 `0.5129`.
- storage: only small smoke/exact/broader metric exports were retained under the primary source-run `analysis_exports`; the temporary root `yolo26s.pt` download was removed.
- judgment: dual-source arbitration is broader-negative and should not replace the specialist router. Do not repeat this as source-mode, absent fallback, union distance, agreement distance, or score-order tuning unless paired with a new no-GT verifier that first proves broad TP/FP/FN movement.

Latest V3 stop-line isolated-neck train:

- branch/worktree: `exp/lane-family-f1/stopline-v3-isolated-neck`.
- implementation: `PV26Heads` now has opt-in `roadmark_architecture="v3_stopline_isolated"` and freeze policy `lane_family_stopline_only`. The V3 path adds gated P2/P3 residual isolators only in front of the stop-line head; lane/crosswalk heads are untouched by the stop-line specialist route.
- data/storage contract: real CUDA smoke and main train reused `/home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The main run indexed `429350` records with split `326709 / 82641 / 20000`. After evaluation, the smoke run, duplicate task-best/last checkpoints, TensorBoard, and root `yolo26s.pt` were removed; retained main run size is about `107M`.
- main train: `3` epochs, `512` train batches, `128` val batches, batch size `4`, skipped steps `0`, stop-line loss only. Internal training validation improved by epoch 3 to lane/stop/cross `0.5527 / 0.5913 / 0.5882`, stop-line TP/FP/FN `34 / 28 / 19`; this is not final evidence because the fixed router exact gate uses validation epoch `2`.
- fixed router exact-val128 epoch-2: lane/stop/cross `0.5888 / 0.4186 / 0.5988`, TP/FP/FN lane `1202 / 491 / 1188`, stop-line `27 / 42 / 33`, crosswalk `50 / 36 / 31`.
- judgment: the V3 stop-line isolated neck is trainable and can overfit/fit the internal val slice, but it fails the fixed exact gate and is below projection-competition exact (`0.5333`, TP/FP/FN `32 / 28 / 28`). Broader-val512 was skipped for storage/time discipline. Do not repeat as V3 neck LR, epoch-count, stopline-only sampler, gate-init, or same freeze-policy tuning unless paired with a new candidate/geometry signal that first improves fixed exact TP/FP/FN.

Latest stop-line midpoint proposal head train:

- branch/worktree: `exp/lane-family-f1/stopline-midpoint-proposal-head`.
- implementation: `StopLineDenseLocalHead` now emits opt-in `stop_line_midpoint_logits`; `PV26MultiTaskLoss` can supervise it with `stopline_midpoint_aux_weight`, and projection-competition decode can use `stop_line_projection_comp_proposal_source="midpoint"`. Lane and crosswalk runtime paths remain the retained primary checkpoint with `flip_centerline_avg_lane_cross_comp050` and `crosswalk_polygon_mode=hull`.
- data/storage contract: real CUDA smoke and main train reused `/home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The main run indexed `429350` records with split `326709 / 82641 / 20000`. After evaluation, the smoke run, duplicate task-best/last checkpoints, TensorBoard, and root `yolo26s.pt` were removed; retained main run size is about `97M`.
- main train: `3` epochs, `512` train batches, `128` val batches, batch size `4`, skipped steps `0`, stop-line loss only. Internal training validation never made the new midpoint proposal useful for stop-line: epoch stop-line F1 was `0.1875`, `0.0923`, `0.1695`; best phase-objective was epoch `1`, `0.6025`.
- fixed router exact-val128 epoch-2: lane/stop/cross `0.5888 / 0.1739 / 0.5988`, TP/FP/FN lane `1202 / 491 / 1188`, stop-line `6 / 3 / 54`, crosswalk `50 / 36 / 31`.
- judgment: the dedicated midpoint proposal head is trainable and integrated as a runtime proposal source, but it collapses stop-line recall and is far below projection-competition exact (`0.5333`, TP/FP/FN `32 / 28 / 28`). Broader-val512 was skipped because the exact gate failed. Do not repeat as midpoint aux-weight, proposal-source, head-LR, epoch-count, or same stopline-only sampler tuning without a different candidate/geometry contract.

Latest heterogeneous endpoint-pair specialist router audit:

- branch/worktree: `exp/lane-family-f1/heterogeneous-stopline-router`.
- implementation: `tools/probe_pv26_lane_flip_tta.py` now accepts `--stop-line-lane60-experiment`, builds a separate scenario/train/postprocess contract for the stop-line specialist when needed, and postprocesses specialist outputs with the specialist config rather than the primary config.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/heterogeneous_endpoint_pair_router_exact_val128_epoch2/summary.json`.
- exact-val128 endpoint-pair metric-verifier specialist result: lane/stop/cross `0.5888 / 0.3387 / 0.5988`, stop-line TP/FP/FN `21 / 43 / 39`.
- source-mode exact results stayed below primary projection-comp: primary `32 / 28 / 28`, F1 `0.5333`; primary-absent-specialist `32 / 30 / 28`, F1 `0.5246`; union-dedupe `34 / 45 / 26`, F1 `0.4892`; agreement `22 / 21 / 38`, F1 `0.4272`.
- judgment: the router tooling is useful for heterogeneous stop-line contracts, but endpoint-pair specialists remain exact-negative. Broader-val512 was skipped. Do not repeat endpoint-pair routing as source-mode, dedupe, agreement, or primary-absent fallback tuning without a new endpoint-quality/candidate-coverage signal.

Latest stop-line-priority positive-sampler larger-range router audit:

- branch/worktree: `exp/lane-family-f1/heterogeneous-stopline-router`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_194249`.
- changed execution scale only: same `stopline_priority_positive_sampler` axis, but user-requested `3` epochs, `2048` train batches per epoch, `256` val batches, batch size `4`, CUDA.
- data/storage contract: training reused `/home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. After fixed exact eval, duplicate task-best/last checkpoints, TensorBoard, and root `yolo26s.pt` were removed; retained run size is about `113M`.
- internal best phase-objective epoch was epoch `2`: objective `0.6528`, lane/stop/cross F1 `0.5398 / 0.5833 / 0.6386`, stop-line TP/FP/FN `63 / 46 / 44`. This is not final evidence because it is the training validation slice, not the fixed router exact gate.
- fixed router exact-val128 epoch-2: lane/stop/cross `0.5888 / 0.4839 / 0.5988`, TP/FP/FN lane `1202 / 491 / 1188`, stop-line `30 / 34 / 30`, crosswalk `50 / 36 / 31`.
- judgment: larger train-batch exposure did not improve the stop-line-priority specialist. It is below primary projection-comp exact (`32 / 28 / 28`, F1 `0.5333`) and below the prior 512-batch specialist exact (`30 / 32 / 30`, F1 `0.4918`). Broader-val512 was skipped. Do not continue this as train-batch, val-batch, epoch-count, or same sampler scaling.

Latest stop-line-only mask-first specialist train:

- branch/worktree: `exp/lane-family-f1/stopline-only-mask-first-specialist`.
- implementation: `stopline_only_mask_first` is now a selectable `roadmark_architecture`, backed by `PV26StopLineOnlyHeads`; the `stopline_only_mask_first_specialist` probe trains only the stop-line specialist head with lane/crosswalk losses disabled.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_only_mask_first_specialist_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_205917`.
- data/storage contract: training reused `/home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. The smoke run, duplicate task-best/last checkpoints, TensorBoard, and root `yolo26s.pt` were removed; retained run size is about `90M`.
- main train: real CUDA `3` epochs, `512` train batches per epoch, `128` val batches, batch size `4`, skipped steps `0`.

| Epoch | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 | Stop TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | `0.2794` | `0.0000` | `0.3303` | `0.0000` | `18 / 36 / 37` |
| 2 | `0.3111` | `0.0000` | `0.4531` | `0.0000` | `29 / 39 / 31` |
| 3 | `0.3251` | `0.0000` | `0.5862` | `0.0000` | `34 / 29 / 19` |

- fixed router exact-val128 epoch-2: lane/stop/cross `0.5888 / 0.4308 / 0.5988`, TP/FP/FN lane `1202 / 491 / 1188`, stop-line `28 / 42 / 32`, crosswalk `50 / 36 / 31`.
- judgment: removing lane/crosswalk branches from the specialist can make the internal training validation slice look strong, but the fixed runtime router exact gate is below primary projection-comp exact (`32 / 28 / 28`, F1 `0.5333`) and below the stop-line-priority specialist exact. Broader-val512 was skipped. Do not continue this as architecture-name plumbing, stopline-only freeze policy, head-LR, epoch-count, or stopline-only sampler tuning without a new candidate/geometry signal.

Latest task-routed multi-teacher distill student:

- branch/worktree: `exp/lane-family-f1/task-routed-distill-student`.
- implementation: `PV26TaskRoutedDistillTeacher` now builds the normal distill cache from a default teacher and replaces task-prefixed cache keys from optional task-specific teachers. The probe preset trains one single checkpoint with the retained primary checkpoint as the default lane/crosswalk teacher and the stop-line-priority specialist as the stop-line teacher.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_task_routed_distill_student_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_212815`.
- data/storage contract: real CUDA smoke and main train reused `/home/kai/yolopv26/seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The main run indexed `429350` records with split `326709 / 82641 / 20000`. After fixed exact eval, the smoke run, duplicate task-best/last checkpoints, TensorBoard, and root `yolo26s.pt` were removed; retained main run size is about `111M`.
- main train: real CUDA `3` epochs, `512` train batches per epoch, `128` val batches, batch size `4`, skipped steps `0`.

| Epoch | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 | Lane TP/FP/FN | Stop TP/FP/FN | Cross TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `0.6286` | `0.5476` | `0.3619` | `0.7037` | `1087 / 548 / 1248` | `19 / 31 / 36` | `57 / 22 / 26` |
| 2 | `0.6341` | `0.5503` | `0.4793` | `0.6026` | `1092 / 487 / 1298` | `29 / 32 / 31` | `47 / 28 / 34` |
| 3 | `0.6575` | `0.5494` | `0.6034` | `0.6599` | `1040 / 463 / 1243` | `35 / 28 / 18` | `65 / 26 / 41` |

- fixed single-checkpoint exact-val128 epoch-2: lane/stop/cross `0.5775 / 0.4754 / 0.5854`, TP/FP/FN lane `1140 / 418 / 1250`, stop-line `29 / 33 / 31`, crosswalk `48 / 35 / 33`.
- judgment: task-routed distillation is trainable and improves exact lane versus several recent trained checkpoints, but it does not transfer the stop-line specialist into a single checkpoint and it loses crosswalk below `0.60`. It is below primary projection-comp exact stop-line (`32 / 28 / 28`, F1 `0.5333`) and below the current two-checkpoint broader router lower bound, so broader-val512 was skipped. Do not continue this as teacher-map, distill-weight, head-LR, epoch-count, same task-positive sampler, or same teacher-checkpoint tuning without a new task-specific routing/geometry signal.

Latest stop/cross lane-frozen trained runtime composite:

- branch/worktree: `exp/lane-family-f1/stopline-cross-lane-frozen`.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_cross_priority_lane_frozen_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_105725/analysis_exports/full_runtime_task_balance_exact_val128_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_cross_priority_lane_frozen_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_105725/analysis_exports/full_runtime_task_balance_val512_epoch2/summary.json`
- changed training axis: add `lane_family_stop_cross_heads_only`, which freezes trunk/detector/TL/lane and trains only stop-line + crosswalk heads. Final eval keeps fixed `flip_centerline_avg_lane_cross_comp050`, projection-competition stop-line runtime decode, and `crosswalk_polygon_mode=hull`.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly, no dataset copy; main run indexed `429350` records with split `326709 / 82641 / 20000`. After evaluation, smoke checkpoints were removed and the main run keeps only `best.pt`, summaries/history, and exact/broader metric exports (`101M`).
- main training history best by phase objective was epoch `3`: objective `0.6731892262`, lane/stop/cross F1 `0.5517 / 0.6923 / 0.6368`, TP/FP/FN lane `1076 / 542 / 1207`, stop-line `36 / 15 / 17`, crosswalk `64 / 31 / 42`. This is not final evidence because it is the training validation slice, not the fixed epoch-2 eval.
- fixed full runtime task-balance exact-val128 epoch-2 eval: objective `0.6451854982`, lane/stop/cross F1 `0.5800 / 0.4655 / 0.5963`, TP/FP/FN lane `1176 / 489 / 1214`, stop-line `27 / 29 / 33`, crosswalk `48 / 32 / 33`.
- fixed full runtime task-balance broader-val512 epoch-2 eval: objective `0.6421286661`, lane/stop/cross F1 `0.5571 / 0.5278 / 0.6142`, TP/FP/FN lane `4464 / 2086 / 5013`, stop-line `128 / 86 / 143`, crosswalk `234 / 133 / 161`.
- lane-family mean/min F1: `0.5664 / 0.5278`.
- 판단: freezing lane out of the optimizer partially recovers lane versus the stop-line-priority run (`0.5463 -> 0.5571`) while preserving a small stop-line gain over projection-comp (`0.5164 -> 0.5278`). It still fails all-task `0.60`, exact-val128 rejects stop-line (`0.4655`), and crosswalk remains only slightly above broader threshold. Do not repeat this as freeze-policy, stop/cross sampler order, epoch-count, head-LR, or loss-weight tuning without a new candidate/geometry or instance-retention signal.
- stop-line-head-only transplant check: transplanting only this run's stop-line head into the retained lane/cross checkpoint restored retained lane/cross behavior but lost the stop-line gain. Exact-val128 lane/stop/cross was `0.5888 / 0.4800 / 0.5988`; broader-val512 was `0.5628 / 0.4894 / 0.6187`, stop-line TP/FP/FN `127 / 121 / 144`. The temporary merged checkpoint was pruned; only metric exports remain under `runs/pv26_exhaustive_od_lane_train/lane60_stopcross_lanefrozen_stop_head_merge_20260529/analysis_exports`. 판단: the lane-frozen stop-line gain is not a reusable stop-line-head-only improvement; do not continue this as another task-head recombination.

Latest stop/cross lane-frozen larger-range scale audit:

- branch/worktree: `exp/lane-family-f1/stopcross-lanefrozen-scale2048`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_cross_priority_lane_frozen_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_114452`.
- changed execution scale only: same `lane_family_stop_cross_heads_only` axis and same fixed final runtime eval, but train with `3` epochs, `2048` train batches per epoch, `256` val batches, batch size `4`, CUDA.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. After evaluation, duplicate task-best/last checkpoints, TensorBoard, and temporary `yolo26s.pt` were removed; retained run size is about `104M`.
- internal best phase-objective epoch was epoch `2`: objective `0.6611931888`, lane/stop/cross F1 `0.5475 / 0.6049 / 0.6300`, TP/FP/FN lane `2166 / 1128 / 2452`, stop-line `62 / 36 / 45`, crosswalk `126 / 58 / 90`. This is not final evidence because it is the training validation slice, not the fixed epoch-2 eval.
- fixed full runtime task-balance exact-val128 epoch-2 eval: objective `0.6460998095`, lane/stop/cross F1 `0.5775 / 0.4793 / 0.5839`, TP/FP/FN lane `1174 / 502 / 1216`, stop-line `29 / 32 / 31`, crosswalk `47 / 33 / 34`.
- fixed full runtime task-balance broader-val512 epoch-2 eval: objective `0.6413086591`, lane/stop/cross F1 `0.5564 / 0.5122 / 0.6162`, TP/FP/FN lane `4457 / 2087 / 5020`, stop-line `126 / 95 / 145`, crosswalk `232 / 126 / 163`.
- 판단: larger train-batch exposure did not improve the lane-frozen axis. It regresses stop-line below the retained projection-comp reference (`0.5164 -> 0.5122`) and below the 512-batch lane-frozen run (`0.5278 -> 0.5122`), while lane also stays below the retained lane composite. Do not continue this as a train-batch, epoch-count, val-batch, or same freeze/sampler scaling sweep.

Latest lane seed-trace instance decoder audit:

- branch/worktree: `exp/lane-family-f1/lane-seed-trace-instance`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_trace_instance_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_131325`.
- changed axis: add bottom-anchor seed-only auxiliary supervision and an opt-in `row_scan_tangent_seed_trace` vectorizer mode. Runtime keeps row-scan/tangent lanes first, then appends learned seed-trace candidates only when they are not within the lane matching threshold of an existing prediction.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. Intermediate smoke runs, duplicate task-best/last checkpoints, TensorBoard output, and temporary `yolo26*.pt` downloads were pruned; retained final run size is about `117M`.
- smoke validation exposed a contract bug where combined seed-trace mode returned before row-scan. That pre-fix run was pruned. After fixing preservation and duplicate suppression, exact-val128 epoch-2 on the 32-batch checkpoint was still lane/stop/cross `0.4130 / 0.5299 / 0.5799`, with lane TP/FP/FN `1051 / 1649 / 1339`.
- larger-slice train: `1` epoch, `512` train batches, `128` val batches, batch size `4`, CUDA, skipped steps `0`.
- larger-slice training validation result: objective `0.6202068390`, lane/stop/cross F1 `0.5297 / 0.3299 / 0.6832`, TP/FP/FN lane `1048 / 574 / 1287`, stop-line `16 / 26 / 39`, crosswalk `55 / 23 / 28`.
- fixed exact-val128 epoch-2 eval: objective `0.6328142036`, lane/stop/cross F1 `0.5488 / 0.5000 / 0.5644`, TP/FP/FN lane `1096 / 508 / 1294`, stop-line `29 / 27 / 31`, crosswalk `46 / 36 / 35`.
- 판단: larger training improves the seed-trace lane result versus the 32-batch checkpoint but remains below the retained lane-preserving exact and broader references, and it does not preserve crosswalk or stop-line to the required level. Because exact-val128 misses all three `0.60` task gates, broader-val512 was skipped. Do not continue this as seed threshold, max seeds, aux weight, head-LR, freeze-policy, or longer-run tuning without a new TP-preserving instance quality signal.

Latest lane bidirectional seed-trace smoke:

- branch/worktree: `exp/lane-family-f1/lane-bidirectional-seed-trace`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_lane_bidirectional_seed_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_011614`.
- changed axis: add an opt-in interior-seed trace vectorizer mode that traces from learned centerline-core seed logits upward and downward, then appends non-duplicate traces after the retained row-scan/tangent output. This was meant to test whether interior centerline-supported seeds recover truncated or missed lane instances without copying data or changing stop-line/crosswalk contracts.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. Negative checkpoints, TensorBoard output, and temporary root `yolo26s.pt` were pruned; retained run size is about `22M`.
- CUDA smoke train: `1` epoch, `64` train batches, `4` internal val batches, batch size `4`, skipped steps `0`.
- internal val4 task-best F1: lane `0.5271`, stop-line `0.0000`, crosswalk `0.7692`. This is not final evidence because it is the training validation slice, not the fixed epoch-2 eval.
- fixed val4 epoch-2 eval: objective `0.6073871382`, lane/stop/cross `0.4923 / 0.0000 / 0.4000`, TP/FP/FN lane `32 / 12 / 54`, stop-line `0 / 3 / 2`, crosswalk `2 / 1 / 5`.
- existing-decode ablation: re-running the same checkpoint with `lane60_experiment=stopline_projection_comp_runtime` produced the same fixed val4 counts. The regression is therefore not only the appended bidirectional trace branch; the short seed-supervised training already damaged retained dense behavior.
- 판단: fixed smoke loses lane and crosswalk versus the retained fixed val4 reference and recovers no stop-line TP, so exact-val128 and broader-val512 were skipped. Do not continue this as seed-threshold, max-seeds, seed-aux-weight, head-LR, epoch-count, or same interior-trace tuning without a TP-preserving seed quality / instance-existence signal.

Latest stop-line priority retention-distill heads-only smoke:

- branch/worktree: `exp/lane-family-f1/stopline-priority-retention-distill`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_retention_distill_heads_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_013211`.
- changed axis: keep the projection-competition runtime contract and stop-line-priority sampler order, but add live teacher-cache retention on lane/crosswalk only from the retained merged checkpoint. The stop-line distill weight is `0.0`, so this tests whether lane/cross retention can protect the stop-line exposure branch rather than self-distilling stop-line.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. Negative checkpoints, TensorBoard output, and temporary root `yolo26s.pt` were pruned; retained run size is about `7.3M`.
- CUDA smoke train: `1` epoch, `64` train batches, `4` internal val batches, batch size `4`, skipped steps `0`.
- internal val4 task-best F1: lane `0.5522`, stop-line `0.0000`, crosswalk `0.8000`. This is not final evidence because it is the training validation slice, not the fixed epoch-2 eval.
- fixed val4 epoch-2 eval: objective `0.6357164870`, lane/stop/cross `0.5373 / 0.0000 / 0.5455`, TP/FP/FN lane `36 / 12 / 50`, stop-line `0 / 3 / 2`, crosswalk `3 / 1 / 4`.
- 판단: lane/cross teacher retention did not protect lane enough and did not move stop-line on the fixed smoke gate. Exact-val128 and broader-val512 were skipped. Do not continue this as distill-weight, head-LR, loss-weight, epoch-count, or same stop-line-priority sampler tuning without a materially different stop-line candidate/geometry or lane-retention contract.

Latest stop-line source-union projection-comp smoke:

- branch/worktree: `exp/lane-family-f1/stopline-source-union-proposals`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_source_union_projection_comp_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_014626`.
- changed axis: add `stop_line_projection_comp_proposal_source="center_selector_union"` so projection competition receives source-local center and selector proposal peaks before top-k collapse, and train the selector map with `stopline_selector_target_mode="rowx_band"` / `stopline_selector_aux_weight=0.75`. This tested whether selector-local proposal support can recover stop-line candidates that center/global-max proposals miss.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. Negative checkpoints, TensorBoard output, and temporary root `yolo26s.pt` were pruned; retained run size is about `7.3M`.
- CUDA smoke train: `1` epoch, `64` train batches, `4` internal val batches, batch size `4`, skipped steps `0`.
- fixed val4 epoch-2 eval: objective `0.6211438491`, lane/stop/cross `0.4885 / 0.0000 / 0.5455`, TP/FP/FN lane `32 / 13 / 54`, stop-line `0 / 3 / 2`, crosswalk `3 / 1 / 4`.
- 판단: source-local center/selector proposal union plus rowx-band selector supervision did not recover any stop-line TP and damaged lane. Exact-val128 and broader-val512 were skipped. Do not continue this as proposal-source, top-k, min-gap, selector-target, selector-aux-weight, head-LR, or epoch-count tuning unless a new candidate-quality/geometry signal first moves fixed smoke TP/FP/FN.

Latest lane conditional seed-branch-only trace smoke:

- branch/worktree: `exp/lane-family-f1/lane-seed-branch-only-trace`.
- diagnostic pre-fix run: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_020030`.
- fixed-BN rerun: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_021142`.
- changed axis: add `lane_conditional_seed_only` freeze policy, train only `LaneSegFirstHead.conditional_seed_logits`, zero all dense lane loss terms, and evaluate with `row_scan_tangent_seed_trace`, projection-competition stop-line runtime, and `crosswalk_polygon_mode="hull"`.
- implementation detail: `trainer.apply_freeze_policy_train_modes()` now forces trunk/head eval mode for `lane_conditional_seed_only`, preventing frozen BatchNorm running-stat mutation while still training the seed conv.
- data/storage contract: both smoke runs reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. Both indexed `429350` records with split `326709 / 82641 / 20000`, trained real CUDA `1` epoch, `64` train batches, `4` internal val batches, batch size `4`, skipped steps `0`.
- diagnostic pre-fix fixed val4:

| Source | Lane F1 | Lane TP/FP/FN | Stop-line F1 | Stop TP/FP/FN | Crosswalk F1 | Cross TP/FP/FN |
| --- | ---: | --- | ---: | --- | ---: | --- |
| base row-scan | `0.5507` | `38 / 14 / 48` | `0.0000` | `0 / 3 / 2` | `0.5455` | `3 / 1 / 4` |
| base seed-trace | `0.3543` | `31 / 58 / 55` | `0.0000` | `0 / 4 / 2` | `0.5455` | `3 / 1 / 4` |
| trained seed-trace, pre-fix | `0.5324` | `37 / 16 / 49` | `0.0000` | `0 / 3 / 2` | `0.5455` | `3 / 1 / 4` |
| trained row-scan ablation, pre-fix | `0.5362` | `37 / 15 / 49` | `0.0000` | `0 / 3 / 2` | `0.5455` | `3 / 1 / 4` |

- pre-fix interpretation: seed-only training reduced the untrained seed-trace FP blow-up, but did not beat base row-scan. The row-scan ablation changed despite seed-trace being disabled, proving that frozen BatchNorm buffers can damage retained dense behavior when only parameters are frozen.
- fixed-BN rerun fixed val4: objective `0.5515308931`, lane/stop/cross `0.3409 / 0.0000 / 0.5455`, TP/FP/FN lane `30 / 60 / 56`, stop-line `0 / 3 / 2`, crosswalk `3 / 1 / 4`.
- artifacts:
  - pre-fix fixed smoke: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_020030/analysis_exports/fixed_val4_epoch2/metrics.csv`
  - pre-fix row-scan ablation: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_020030/analysis_exports/fixed_val4_epoch2_row_scan_ablation/metrics.csv`
  - base seed-trace reference: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_020030/analysis_exports/base_seed_fixed_val4_epoch2/metrics.csv`
  - base row-scan reference: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_020030/analysis_exports/base_row_scan_fixed_val4_epoch2/metrics.csv`
  - fixed-BN rerun smoke: `runs/pv26_exhaustive_od_lane_train/lane60_lane_seed_branch_only_trace_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260530_021142/analysis_exports/fixed_val4_epoch2/metrics.csv`
- storage: negative checkpoints, TensorBoard output, and temporary root `yolo26s.pt` were pruned. Retained pre-fix diagnostic run size is about `24M`; retained fixed-BN rerun size is about `6.1M`.
- 판단: seed-branch-only bottom-anchor trace is trainable but smoke-negative and FP-heavy. The reusable result is the freeze-mode fix; do not continue this as seed target, threshold, max-seeds, head-LR, epoch-count, or freeze-policy tuning without a new TP-preserving instance quality/existence signal.

Latest input-scale672 dense-target audit:

- branch/worktree: `exp/lane-family-f1/input-scale672-dense-target`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_projection_comp_runtime_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_133550`.
- changed axis: make `NETWORK_HW` opt-in through `PV26_NETWORK_HW`, compute `ROADMARK_DENSE_OUTPUT_HW` from that input size, and train/evaluate with `PV26_NETWORK_HW=672x896`. This is not the closed single-scale TTA family; it changes the online letterbox and dense target/output grid from `608x800` / `152x200` to `672x896` / `168x224`.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. Smoke run `...133341` was pruned. The main run retains only `phase_4/checkpoints/best.pt`, summaries/history, and exact/broader metric exports, about `112M`. Temporary `yolo26s.pt` was removed.
- smoke: real CUDA `1` epoch, `16` train batches, `4` val batches, batch size `4`, skipped steps `0`.
- main train: real CUDA `3` epochs, `512` train batches, `128` val batches, batch size `4`, skipped steps `0`, best phase-objective epoch `3`.
- training history:

| Epoch | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 | Lane TP/FP/FN | Stop TP/FP/FN | Cross TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `0.6265115065` | `0.5509` | `0.3214` | `0.6951` | `1082 / 511 / 1253` | `18 / 39 / 37` | `57 / 24 / 26` |
| 2 | `0.6480047870` | `0.5635` | `0.5692` | `0.5465` | `1122 / 470 / 1268` | `37 / 33 / 23` | `47 / 44 / 34` |
| 3 | `0.6524074556` | `0.5520` | `0.5333` | `0.6492` | `1046 / 461 / 1237` | `32 / 35 / 21` | `62 / 23 / 44` |

- fixed exact-val128 epoch-2 eval: objective `0.6483222172`, lane/stop/cross F1 `0.5662 / 0.5426 / 0.5763`, TP/FP/FN lane `1119 / 444 / 1271`, stop-line `35 / 34 / 25`, crosswalk `51 / 45 / 30`.
- fixed broader-val512 epoch-2 eval: objective `0.6378675580`, lane/stop/cross F1 `0.5519 / 0.4734 / 0.6257`, TP/FP/FN lane `4322 / 1862 / 5155`, stop-line `129 / 145 / 142`, crosswalk `239 / 130 / 156`.
- 판단: larger dense target/input training is executable and crosswalk remains broader-pass, but it regresses the retained task-balance lane/stop-line reference (`0.5628 / 0.5164 / 0.6187` -> `0.5519 / 0.4734 / 0.6257`). Exact stop-line improves over the retained exact projection-comp row, but exact crosswalk fails and broader stop-line falls sharply. Do not continue this as `672x896` longer-run, input-size ladder, dense-output-size, head-LR, or same projection-comp-runtime training sweep without a new candidate/geometry or retention signal.

Latest stop-line axis-distance field audit:

- branch/worktree: `exp/lane-family-f1/stopline-axis-distance-field`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_axis_distance_field_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_142901`.
- changed axis: add dense stop-line axis-distance targets/heads and an opt-in consensus decoder that predicts canonical axis direction, start/end distances along that axis, and normal recenter offset from support pixels. This is distinct from XY HAF endpoint voting and from seeded segment-set queries.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. Smoke run, duplicate task-best/last checkpoints, TensorBoard output, and temporary `yolo26s.pt` were pruned; retained final run size is about `111M`.
- main train: real CUDA `3` epochs, `512` train batches, `128` val batches, batch size `4`, skipped steps `0`, best phase-objective epoch `1`.
- fixed exact-val128 eval: objective `0.5807787881`, lane/stop/cross F1 `0.5643 / 0.2128 / 0.5644`, TP/FP/FN lane `1104 / 419 / 1286`, stop-line `15 / 66 / 45`, crosswalk `46 / 36 / 35`.
- fixed broader-val512 eval: objective `0.5809544355`, lane/stop/cross F1 `0.5431 / 0.2629 / 0.6081`, TP/FP/FN lane `4220 / 1842 / 5257`, stop-line `69 / 185 / 202`, crosswalk `225 / 120 / 170`.
- 판단: the decomposed axis-distance field trains and decodes, but it fails the exact gate and broadens badly. Stop-line is far below both current runtime `0.4235` and projection-competition `0.5164`, while lane also regresses. Do not continue this as axis-distance aux-weight, valid-threshold, min-vote, support-score, covariance, head-LR, or longer-run tuning without a materially different verification/candidate contract.

Latest stop-line focus-crop feeding audit:

- branch/worktree: `exp/lane-family-f1/stopline-focus-crop-feeding-v2`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_focus_crop_feeding_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_155424`.
- changed axis: train-time stop-line-centered crop/zoom feeding on the existing letterboxed tensor, with validation/eval left on the normal image contract. This is a data-feeding/preprocess axis, not a dataset copy and not a runtime TTA/postprocess sweep.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. Smoke run, duplicate task-best/last checkpoints, TensorBoard output, and temporary `yolo26s.pt` were pruned; retained final run size is about `119M`.
- main train: real CUDA `2` epochs, `512` train batches, `128` val batches, batch size `4`, skipped steps `0`, best phase-objective epoch `2`.
- exact-val128 epoch-2 training eval: objective `0.6163024956`, lane/stop/cross F1 `0.5592 / 0.3597 / 0.5591`, TP/FP/FN lane `1127 / 514 / 1263`, stop-line `25 / 54 / 35`, crosswalk `52 / 53 / 29`.
- fixed broader-val512 epoch-2 eval: objective `0.6125058329`, lane/stop/cross F1 `0.5454 / 0.4061 / 0.5910`, TP/FP/FN lane `4350 / 2124 / 5127`, stop-line `119 / 196 / 152`, crosswalk `229 / 151 / 166`.
- 판단: focus crop/zoom feeding is trainable, but it is not a stop-line breakthrough. It recovers stop-line TP versus the raw objective-best runtime (`101 -> 119`) but increases FP far more (`105 -> 196`) and falls below both raw runtime stop-line F1 `0.4235` and projection-competition `0.5164`. Lane and crosswalk also regress. Do not continue this as crop probability, crop scale/jitter, head-LR, epoch-count, or sampler-order tuning without a new FP-control/candidate-geometry signal.

Latest upper-trunk retention-distill audit:

- branch/worktree: `exp/lane-family-f1/retention-distill-stopline-train`.
- run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_retention_distill_upper_trunk_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_162533`.
- changed axis: open upper trunk plus lane-family heads, train stop-line with projection-competition runtime contract, and add source-checkpoint retention distillation only on lane/crosswalk (`distill_loss_weights={lane:0.20, stop_line:0.0, crosswalk:0.20}`). This is distinct from the closed same-checkpoint stop-line self-distill.
- implementation: live distill teacher cache now includes seg-first lane dense maps (`lane_seg_centerline/support/offset/tangent/color/type`) so lane retention can target the actual current lane head rather than only legacy row-head logits.
- data/storage contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The run indexed `429350` records with split `326709 / 82641 / 20000`. Smoke run, duplicate task-best/last checkpoints, TensorBoard output, and temporary root `yolo26s.pt`/`yolo26n.pt` downloads were pruned; retained final run size is about `150M`.
- main train: real CUDA `2` epochs, `512` train batches, `128` val batches, batch size `4`, skipped steps `0`, best phase-objective epoch `2`.
- fixed exact-val128 epoch-2 eval: objective `0.6390391066`, lane/stop/cross F1 `0.5587 / 0.4793 / 0.6038`, TP/FP/FN lane `1109 / 471 / 1281`, stop-line `29 / 32 / 31`, crosswalk `48 / 30 / 33`.
- fixed broader-val512 epoch-2 eval: objective `0.6348507903`, lane/stop/cross F1 `0.5336 / 0.5195 / 0.6200`, TP/FP/FN lane `4208 / 2086 / 5269`, stop-line `133 / 108 / 138`, crosswalk `226 / 108 / 169`.
- 판단: retention distill is trainable and keeps crosswalk broader-pass, but it does not preserve lane. Stop-line broader F1 barely rises over retained projection-competition (`0.5164 -> 0.5195`) by adding `+7 TP` and `+17 FP`, while lane falls sharply (`0.5628 -> 0.5336`, TP `4532 -> 4208`). Do not continue this as distill weight, trunk LR, head LR, epoch-count, or sampler-order tuning without a new lane-preserving shared-feature/candidate-geometry contract.

Previous stop-line-exposure trained broader task-balance runtime composite:

- branch/worktree: `exp/lane-family-f1/stopline-priority-positive-sampler`.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_101745/analysis_exports/full_runtime_task_balance_exact_val128_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_priority_positive_sampler_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_101745/analysis_exports/full_runtime_task_balance_val512_epoch2/summary.json`
- changed training axis: same projection-competition runtime decoder and fixed lane/cross runtime composite, but stage-4 task-positive sampler order is `multi:stopline,lane,crosswalk`, giving stop-line `2` positive slots per batch.
- exact-val128 lane / stop-line / crosswalk F1: `0.5639 / 0.4918 / 0.5839`.
- exact TP/FP/FN lane: `1097 / 404 / 1293`; stop-line: `30 / 32 / 30`; crosswalk: `47 / 33 / 34`.
- broader-val512 lane / stop-line / crosswalk F1: `0.5463 / 0.5309 / 0.6219`.
- broader TP/FP/FN lane: `4218 / 1746 / 5259`; stop-line: `133 / 97 / 138`; crosswalk: `227 / 108 / 168`.
- lane-family mean/min F1: `0.5664 / 0.5309`.
- 판단: this is the current broader objective/min-F1 positive among runtime composites and it shows stop-line exposure can recover some stop-line TP. It is still not a solution because the gain comes with a large lane regression versus the retained lane-preserving task-balance composite (`0.5628 -> 0.5463`, TP `4532 -> 4218`). Do not repeat it as sampler/epoch/LR tuning without a new lane-retention or geometry signal.

Latest lane conditional bottom-anchor quality smoke:

- branch/worktree: `exp/lane-family-f1/lane-conditional-bottom-anchor-quality`.
- changed axis: keep the retained lane/stop/cross runtime settings, but change the conditional lane-row auxiliary from full-centerline seed supervision to a sparse bottom-anchor seed target, metric-quality objectness targets, and stronger row-x loss.
- real CUDA smoke: `1` epoch, `32` train batches, `4` val batches, batch size `4`, seed checkpoint `merged_lane_head.pt`.
- storage/data contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The failed smoke checkpoints were pruned after evaluation, leaving only summaries/history.
- smoke val4 result: objective `0.2844807747`, lane/stop/cross F1 `0.0000 / 0.0000 / 0.6667`, lane TP/FP/FN `0 / 59 / 79`, stop-line `0 / 1 / 2`, crosswalk `4 / 3 / 1`, skipped steps `0`.
- 판단: this seed/objectness contract is not worth broadening. It suppresses or misaligns lane instances badly enough that lane recall collapses on the rejection gate; do not continue it as a seed-target, objectness-target, row-x-weight, threshold, head-LR, or longer-run sweep.

Latest stop-line priority positive-sampler train/eval:

- branch/worktree: `exp/lane-family-f1/stopline-priority-positive-sampler`.
- changed axis: keep the projection-competition runtime decoder and retained lane/cross settings, but change the stage-4 task-positive sampler from `multi:lane,stopline,crosswalk` to `multi:stopline,lane,crosswalk`. With batch size `4` and fraction `1.0`, this gives stop-line `2` positive slots per batch instead of `1`.
- real CUDA smoke: `1` epoch, `32` train batches, `4` val batches, batch size `4`, seed checkpoint `merged_lane_head.pt`.
- main train: `3` epochs, `512` train batches, `128` val batches, batch size `4`, seed checkpoint `merged_lane_head.pt`.
- storage/data contract: training reused `seg_dataset/pv26_exhaustive_od_lane_dataset` directly; no dataset copy was created. The main run indexed the existing `429350` canonical records with train/val/test split `326709 / 82641 / 20000`. Failed/duplicate checkpoints and TensorBoard were pruned after evaluation; the retained main run keeps only `best.pt`, summaries, history, and exact/broader metric exports and is about `111M`.
- smoke val4 result: objective `0.6340610868`, lane/stop/cross F1 `0.5271 / 0.0000 / 0.8000`, lane TP/FP/FN `34 / 16 / 45`, stop-line `0 / 1 / 2`, crosswalk `4 / 1 / 1`, skipped steps `0`.
- main training history best by phase objective was epoch `3`: objective `0.6609838519`, lane/stop/cross F1 `0.5440 / 0.6195 / 0.6396`, TP/FP/FN lane `1014 / 431 / 1269`, stop-line `35 / 25 / 18`, crosswalk `63 / 28 / 43`. This is not final evidence because that validation slice has stop-line support `53` and is not the fixed epoch-2 protocol.
- fixed full runtime task-balance exact-val128 epoch-2 eval: objective `0.6446295168`, lane/stop/cross F1 `0.5639 / 0.4918 / 0.5839`, TP/FP/FN lane `1097 / 404 / 1293`, stop-line `30 / 32 / 30`, crosswalk `47 / 33 / 34`.
- fixed full runtime task-balance broader-val512 epoch-2 eval: objective `0.6419406290`, lane/stop/cross F1 `0.5463 / 0.5309 / 0.6219`, TP/FP/FN lane `4218 / 1746 / 5259`, stop-line `133 / 97 / 138`, crosswalk `227 / 108 / 168`.
- 판단: broader stop-line improves over the retained task-balance reference (`0.5164 -> 0.5309`, TP `126 -> 133`), but lane regresses badly (`0.5628 -> 0.5463`, TP `4532 -> 4218`). This is not all-task success and should not be repeated as sampler order, positive fraction, epoch count, or head-LR tuning. A future branch needs a new candidate/geometry or lane-retention signal, not only more stop-line-positive exposure.

Latest stop-line projection-competition runtime contract train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-projcomp-runtime-contract`.
- changed axis: convert the fixed projection-competition stop-line replay (`proj_comp_length_s090_top2_second_frag5`) into an opt-in `postprocess_pv26_batch` runtime decoder, without adding a new model head or copying any dataset files.
- runtime contract: select spaced stop-line support cells, build mask-extent candidates, union projection-compatible fragments, split large along-axis gaps, and cap to `2` predictions with the retained second-fragment guard. Lane remains normal `row_scan_tangent` in this evaluator path; crosswalk remains `crosswalk_polygon_mode=hull`.
- pre-training exact-val128 runtime eval on `merged_lane_head.pt`: objective `0.6482167675`, lane/stop/cross F1 `0.5660 / 0.5333 / 0.5988`, TP/FP/FN lane `1162 / 554 / 1228`, stop-line `32 / 28 / 28`, crosswalk `50 / 36 / 31`.
- pre-training broader-val512 runtime eval on `merged_lane_head.pt`: objective `0.6400621883`, lane/stop/cross F1 `0.5480 / 0.5164 / 0.6187`, TP/FP/FN lane `4457 / 2333 / 5020`, stop-line `126 / 91 / 145`, crosswalk `232 / 123 / 163`.
- runtime 판단: the runtime decoder reproduces the retained projection-competition stop-line broader reference exactly (`126 / 91 / 145`, F1 `0.5164`). It promotes the stop-line replay from artifact-only evidence to a real opt-in runtime contract, but it is not yet the full task-balance composite because this evaluator path does not include the current flip-centerline/crosswalk-mask lane composite (`0.5628`).
- actual train run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_projection_comp_runtime_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_085604`.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly; no dataset copy was created. The run used train/val/test split `326709 / 82641 / 20000` from `429350` indexed records.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA, seed checkpoint `merged_lane_head.pt`.
- internal epoch-3 validation reached stop-line F1 `0.6415`, TP/FP/FN `34 / 19 / 19`, but that slice has stop-line support `53` and is not the fixed epoch-2 protocol.
- fixed exact-val128 eval of trained `best.pt`: objective `0.6419134349`, lane/stop/cross F1 `0.5635 / 0.5085 / 0.5854`, TP/FP/FN lane `1114 / 450 / 1276`, stop-line `30 / 28 / 30`, crosswalk `48 / 35 / 33`.
- fixed broader-val512 eval of trained `best.pt`: objective `0.6392521545`, lane/stop/cross F1 `0.5413 / 0.5217 / 0.6144`, TP/FP/FN lane `4231 / 1926 / 5246`, stop-line `126 / 86 / 145`, crosswalk `231 / 126 / 164`.
- train 판단: additional heads-only training with the projection-comp runtime metric slightly reduces broader stop-line FP (`91 -> 86`) while keeping TP fixed, but exact stop-line falls below the pre-training runtime eval (`0.5333 -> 0.5085`) and broader lane regresses (`0.5480 -> 0.5413`). This is not an all-task breakthrough and should not be repeated as a head-LR/epoch/loss-weight training sweep.
- trained stop-line-head transplant check: merging only the trained stop-line head into the retained lane/cross checkpoint required explicit `--allow-source-extra-keys` because the trained head has newer auxiliary keys. Full task-balance runtime exact-val128 fell to lane/stop/cross `0.5888 / 0.5042 / 0.5988`, stop-line `30 / 29 / 30`; broader-val512 fell to `0.5628 / 0.5113 / 0.6187`, stop-line `124 / 90 / 147`. The merged checkpoint was pruned after eval; only metric exports remain under `runs/pv26_exhaustive_od_lane_train/lane60_stopproj_runtime_stop_head_merge_20260529/analysis_exports`.

Latest projection/candidate tooling status:

- Projection readout scripts/tests are restored in active `develop`.
- Candidate-pool manifest regeneration is restored: `tools/probe_pv26_stopline_candidate_pool.py` now supports `--dataset-root` for detached worktrees, `--proposal-min-gap 4`, and emits the `sample_id` / GT geometry JSON / candidate geometry JSON fields required by the projection replay tools.
- A 1-batch CLI smoke on the retained merged checkpoint produced zero candidates but still wrote the required `candidate_features.csv` header, and projection-competition replay consumed that generated CSV/summary.
- Current `develop` also regenerated a non-empty exact-val128 manifest: `1170` candidate rows, `442` oracle-positive rows, baseline stop-line F1 `0.4483` with TP/FP/FN `26 / 30 / 34`. Projection-competition replay consumed the regenerated manifest and reached exact-val128 stop-line F1 `0.5167`, TP/FP/FN `31 / 29 / 29`.
- Candidate-pool generation now also supports `--projection-competition-replay`, which writes the fixed projection-competition variants and sample rows in the same run. This is a convenience/reproducibility path for the current stop-line reference, not a new stop-line improvement.
- A broader-val512 inline replay from current `develop` reproduced the fixed stop-line projection-competition reference: `5065` candidate rows, `1660` oracle-positive rows, best row `proj_comp_length_s090_top2_second_frag5`, stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- The broader inline replay only re-verifies the stop-line reference. Its lane/crosswalk fields come from the retained checkpoint's non-composite row, so it is not a regenerated all-task task-balance composite.
- 판단: this is reproducibility/tooling status only, not a metric improvement and not a reason to repeat projection readout sweeps.

Latest stop-line live distill result:

- live teacher-cache distill plumbing is implemented and passed a real one-batch CUDA smoke, but the first metric run is negative.
- single-axis follow-up `stop_line` self-distill from the same frozen merged checkpoint used phase 4 only, `2` epochs, `128` train batches, and `distill_loss_weights={lane:0, stop_line:0.25, crosswalk:0}`.
- standard exact-val128 replay rejects the resulting best checkpoint: source lane/stop/cross F1 `0.5445 / 0.4483 / 0.5854`, distill best `0.4835 / 0.3774 / 0.5036`.
- 판단: do not broaden this to val512, and do not repeat same-checkpoint stop-line-only teacher-cache self-distill as a weight/epoch/batch-size/EMA sweep. Future distill use would need a different teacher target or a changed stop-line decode/assignment contract.

Latest stop-line HAF consensus train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-haf-consensus-segment`.
- changed axis: add a learned stop-line HAF endpoint/valid field and opt-in consensus decoder; keep lane row-scan/tangent and `crosswalk_polygon_mode=hull` retained.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly; no dataset copy was created. Redundant task-best checkpoint copies were pruned from the HAF runs, leaving the retained `best.pt` checkpoint and eval summaries.
- 3-epoch train with `512` train batches and `128` val batches completed without non-finite/skipped steps, but the best checkpoint is negative.
- exact-val128 epoch-2 eval with HAF disabled: lane/stop/cross F1 `0.5580 / 0.3036 / 0.5868`, stop-line TP/FP/FN `17 / 35 / 43`.
- exact-val128 epoch-2 eval with HAF enabled at valid threshold `0.95`: lane/stop/cross F1 `0.5580 / 0.1698 / 0.5868`, stop-line TP/FP/FN `18 / 134 / 42`.
- broader-val512 epoch-2 eval with HAF disabled: lane/stop/cross F1 `0.5383 / 0.3237 / 0.6220`, stop-line TP/FP/FN `78 / 133 / 193`.
- 판단: the learned HAF target/head/loss path is train-stable, but the simple HAF valid/consensus decoder fails FP control and the checkpoint regresses against the retained broader runtime composite. Do not run longer same-axis HAF aux/head-LR/threshold/min-vote sweeps without a materially different valid-quality/verifier contract.

Latest stop-line HAF quality-hardneg artifact:

- retained run artifact: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_haf_quality_hardneg_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_005721`.
- changed axis captured by artifact config: add HAF quality supervision plus a hard-negative quality weight while keeping lane row-scan/tangent and `crosswalk_polygon_mode=hull`.
- reproducibility caveat: this is a retained run artifact from local state, not a currently maintained code surface in `develop`; restoring the exact quality head would be required before rerunning it.
- storage contract: the retained run is `130M` and keeps only `phase_4/checkpoints/best.pt` plus history and exact/broader eval exports.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset root reused `seg_dataset/pv26_exhaustive_od_lane_dataset`; no dataset copy was created.
- training history stop-line F1 stayed FP-heavy: epoch1 `0.0584` with TP/FP/FN `9 / 244 / 46`, epoch2 `0.1233` with `18 / 214 / 42`, epoch3 `0.1544` with `23 / 222 / 30`.
- exact-val128 epoch-2 HAF-quality eval: lane/stop/cross F1 `0.5596 / 0.1277 / 0.5868`, stop-line TP/FP/FN `18 / 204 / 42`.
- exact-val128 epoch-2 HAF disabled eval: lane/stop/cross F1 `0.5596 / 0.3091 / 0.5868`, stop-line TP/FP/FN `17 / 33 / 43`.
- broader-val512 epoch-2 HAF-quality eval: lane/stop/cross F1 `0.5396 / 0.1360 / 0.6180`, stop-line TP/FP/FN `83 / 867 / 188`.
- 판단: quality/hard-negative HAF did not fix the core FP-control failure; it made HAF-enabled broader stop-line much worse than both retained runtime stop-line `0.4235` and projection-competition `0.5164`. Do not repeat this as HAF quality threshold, quality hard-negative weight, aux weight, head-LR, or longer-run tuning. Reopen HAF only with a genuinely different emit/verification contract that first proves FP control on exact val128.

Latest stop-line seeded segment-set train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-seeded-segment-set`.
- changed axis: add an opt-in dense-seeded stop-line segment-set branch, Hungarian endpoint loss, and runtime decode/fallback union; keep lane row-scan/tangent and `crosswalk_polygon_mode=hull` retained.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly; no dataset copy was created. The main run checkpoint folder was pruned from `745M` to `213M`, retaining only `best.pt` and `best_stop_line.pt`.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_seeded_segment_set_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260528_223739`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- task-best exact-val128 epoch-2 eval: lane/stop/cross F1 `0.5550 / 0.4737 / 0.5783`, stop-line TP/FP/FN `27 / 27 / 33`.
- phase-objective-best exact-val128 segment-set union eval: lane/stop/cross F1 `0.5586 / 0.4522 / 0.5988`, stop-line TP/FP/FN `26 / 29 / 34`. Segment-set disabled gave the same row, so the learned segment decode did not add net TP under the final dedupe/fallback contract.
- broader-val512 task-best eval: lane/stop/cross F1 `0.5419 / 0.4033 / 0.6122`, stop-line TP/FP/FN `97 / 113 / 174`.
- 판단: the simple dense top-K seed + one-shot endpoint MLP segment-set is trainable, but it does not beat exact projection-competition `0.5167` and broader stop-line regresses below both the retained runtime composite `0.4235` and projection-competition reference `0.5164`. Do not run longer same-axis seeded segment-set sweeps without a materially stronger seed/verifier/objectness contract and no-oracle recovery evidence.

Latest stop-line segment-aligned verifier train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-segment-aligned-verifier`.
- changed axis: keep the seeded segment-set branch, then add a segment-aligned verifier that samples dense stop-line features along each predicted segment with `grid_sample` and can replace the segment score; keep lane row-scan/tangent and `crosswalk_polygon_mode=hull`.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly; no dataset copy was created. The main run was pruned from `781M` to `137M`, retaining `phase_4/checkpoints/best.pt`, history, summaries, and eval exports only.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_segment_aligned_verifier_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_013725`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- best training epoch `2`: lane/stop/cross F1 `0.5588 / 0.3680 / 0.5783`, stop-line TP/FP/FN `23 / 42 / 37`.
- exact-val128 checkpoint eval, verifier score primary: lane/stop/cross F1 `0.5588 / 0.3680 / 0.5783`, stop-line TP/FP/FN `23 / 42 / 37`.
- exact-val128 checkpoint eval, segment-set disabled: lane/stop/cross F1 `0.5588 / 0.4182 / 0.5783`, stop-line TP/FP/FN `23 / 27 / 37`.
- exact-val128 checkpoint eval, segment-set enabled but verifier score weight `0.0`: same as disabled, lane/stop/cross F1 `0.5588 / 0.4182 / 0.5783`, stop-line TP/FP/FN `23 / 27 / 37`.
- broader-val512 checkpoint eval, best exact variant with segment-set disabled: lane/stop/cross F1 `0.5418 / 0.3932 / 0.6148`, stop-line TP/FP/FN `93 / 109 / 178`.
- 판단: segment-aligned verifier plumbing is trainable, but this simple verifier does not add no-GT stop-line recovery. The base segment branch emits no net metric gain, and making verifier score primary increases exact FP. Do not continue this as verifier-score-weight, segment threshold, max-segment, or longer-run sweep without a materially different candidate generator or verifier target.

Latest stop-line GT-denoised segment-set train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-segment-denoise-seeds`.
- changed axis: keep the dense-seeded segment-set branch, then add training-only GT midpoint/jitter denoise queries through the same segment MLP plus `stopline_segment_denoise_aux_weight`; keep lane row-scan/tangent and `crosswalk_polygon_mode=hull`.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly; no dataset copy was created. The smoke run, redundant task-best/last checkpoints, TensorBoard files, and temporary YOLO weight downloads were pruned; retained main run is `117M`.
- retained main run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_segment_denoise_seeded_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_051816`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- best training epoch `3`: lane/stop/cross F1 `0.5467 / 0.4510 / 0.6327`, stop-line TP/FP/FN `23 / 26 / 30`.
- exact-val128 epoch-2 eval for `best.pt`: lane/stop/cross F1 `0.5593 / 0.4074 / 0.5976`, stop-line TP/FP/FN `22 / 26 / 38`.
- 판단: GT-denoise supervision is train-stable, but this contract does not transfer into no-GT runtime recovery. It is below projection-competition exact `0.5167` (`31 / 29 / 29`) and below the simple seeded segment-set task-best `0.4737`. Do not continue this as denoise aux weight, jitter amount, verifier-score-weight, segment threshold, head-LR, or longer-run tuning without a new runtime seed/objectness/candidate-coverage contract.

Latest stop-line metric-quality segment verifier train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-segment-metric-quality-verifier`.
- changed axis: keep the dense-seeded segment-set branch, but train the segment-aligned verifier against endpoint-distance metric quality instead of matched-query objectness; keep lane row-scan/tangent and `crosswalk_polygon_mode=hull`.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly; no dataset copy was created. The smoke run, redundant task-best/last checkpoints, TensorBoard files, and temporary YOLO weight downloads were pruned; retained main run is `117M`.
- retained main run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_segment_metric_quality_verifier_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_054808`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `3`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- best exact-val128 epoch-3 eval: lane/stop/cross F1 `0.5502 / 0.4660 / 0.6294`, stop-line TP/FP/FN `24 / 26 / 29`.
- training history:
  - epoch1 lane/stop/cross F1 `0.5363 / 0.2157 / 0.6750`, stop-line TP/FP/FN `11 / 36 / 44`.
  - epoch2 lane/stop/cross F1 `0.5581 / 0.4425 / 0.5749`, stop-line TP/FP/FN `25 / 28 / 35`.
  - epoch3 lane/stop/cross F1 `0.5502 / 0.4660 / 0.6294`, stop-line TP/FP/FN `24 / 26 / 29`.
- 판단: metric-quality verifier target reduces FP relative to the matched-query verifier, but it still does not recover enough TP and remains below projection-competition exact `0.5167` (`31 / 29 / 29`) and simple seeded segment-set task-best `0.4737` (`27 / 27 / 33`). Do not continue this as quality-tau, verifier-score-weight, segment-threshold, max-segment, head-LR, or longer-run tuning without a runtime candidate generator that first recovers no-oracle positives.

Latest lane conditional row instance decoder train/eval result:

- branch/worktree: `exp/lane-family-f1/lane-conditional-row-instance-decoder`.
- changed axis: add an opt-in lane conditional row instance decoder on top of `LaneSegFirstHead`; dense centerline/support/tangent maps remain, but `lane_conditional_row_enabled=true` decodes top-K learned seed rows instead of the row-scan/tangent vectorizer. Stop-line and crosswalk contracts were retained, with `crosswalk_polygon_mode=hull`.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly; no dataset copy was created. The smoke run was deleted, redundant task-best checkpoints and TensorBoard event files were pruned, and the main run was reduced to `138M`, retaining only `phase_4/checkpoints/best.pt`, history, summaries, and eval exports.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_lane_conditional_row_instance_decoder_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_022231`.
- main scale: `5` epochs, `1024` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- best training epoch `5`: lane/stop/cross F1 `0.0102 / 0.4715 / 0.6818`, with lane TP/FP/FN `37 / 4841 / 2346`.
- exact-val128 checkpoint eval, conditional row enabled: lane/stop/cross F1 `0.0111 / 0.4865 / 0.6135`, lane TP/FP/FN `40 / 4794 / 2350`.
- exact-val128 checkpoint eval, conditional row disabled: lane/stop/cross F1 `0.5575 / 0.4865 / 0.6135`, lane TP/FP/FN `1096 / 446 / 1294`.
- broader-val512 checkpoint eval, conditional row enabled: lane/stop/cross F1 `0.0112 / 0.4069 / 0.6319`, lane TP/FP/FN `161 / 19210 / 9316`.
- broader-val512 checkpoint eval, conditional row disabled: lane/stop/cross F1 `0.5336 / 0.4069 / 0.6319`, lane TP/FP/FN `4165 / 1970 / 5312`.
- 판단: the conditional row head/loss/decode path is train-stable, but the simple top-K seed + one-shot MLP row decoder collapses into massive lane FP and is not a production lane instance contract. Disabling the decoder recovers the older dense row-scan path, but the trained checkpoint still regresses below the retained broader runtime composite `0.5628 / 0.4235 / 0.6187` on lane and stop-line. Do not continue this exact branch as seed top-K, objectness threshold, aux-weight, head-LR, or longer-run tuning.

Latest upper-trunk PCGrad rebalance train/eval result:

- branch/worktree: `exp/lane-family-f1/upper-trunk-pcgrad-rebalance`.
- changed axis: keep the retained row-scan/tangent lane contract and hull crosswalk decode, but open `lane_family_plus_upper_trunk` and enable the existing PCGrad-style multitask conflict path on lane/stop-line/crosswalk gradients.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly. No dataset copy was created. The smoke run was deleted, redundant task-best checkpoints and TensorBoard files were pruned, and the main run was reduced to `129M`, retaining `phase_4/checkpoints/best.pt`, history, summaries, and eval exports.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_upper_trunk_pcgrad_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_033232`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- PCGrad diagnostics: `18` windows / `1536` enabled steps, mean conflict rates lane-vs-stop `0.5183`, lane-vs-crosswalk `0.5022`, stop-vs-crosswalk `0.4839`. This confirms real gradient conflict in the opened upper trunk, but it did not translate into broader task F1 gains.
- best training epoch `3`: lane/stop/cross F1 `0.5468 / 0.4717 / 0.6462`, TP/FP/FN lane `1026 / 444 / 1257`, stop-line `25 / 28 / 28`, crosswalk `63 / 26 / 43`.
- exact-val128 checkpoint eval: lane/stop/cross F1 `0.5619 / 0.4505 / 0.5732`, TP/FP/FN lane `1110 / 451 / 1280`, stop-line `25 / 26 / 35`, crosswalk `47 / 36 / 34`.
- broader-val512 checkpoint eval: lane/stop/cross F1 `0.5412 / 0.3992 / 0.6168`, TP/FP/FN lane `4224 / 1910 / 5253`, stop-line `95 / 110 / 176`, crosswalk `231 / 123 / 164`.
- 판단: upper-trunk PCGrad exposes and projects conflicting task gradients, but the current retained runtime composite is still better on broad lane/stop-line (`0.5628 / 0.4235 / 0.6187`). Do not continue this exact branch as trunk-LR, PCGrad task-list, epoch-count, or loss-weight tuning. If training exposure is reopened, it needs a materially different adapter/head-level balancing contract, not only upper-trunk PCGrad on the same row-scan/tangent and stop-line readout.

Latest head-level PCGrad smoke result:

- branch/worktree: `exp/lane-family-f1/head-level-pcgrad-rebalance`.
- changed axis: extend the PCGrad plumbing so the selected optimizer parameter groups can be `trunk`, `heads`, or both, then test the stage-4 `lane_family_heads_only` premise with `param_groups=["heads"]`.
- storage contract: the smoke training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly. No dataset copy was created. The negative smoke run and temporary YOLO download were deleted after metric/diagnostic extraction.
- smoke run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_head_pcgrad_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_040823`, now deleted.
- smoke scale: `1` epoch, `8` train batches, `4` val batches, batch size `2`, CUDA.
- smoke val4 metrics: objective `0.6197765539`, lane/stop/cross F1 `0.4750 / 0.0000 / 0.6667`, TP/FP/FN lane `19 / 13 / 29`, stop-line `0 / 1 / 1`, crosswalk `1 / 1 / 0`.
- PCGrad diagnostics: `8` enabled steps, `param_groups=["heads"]`, mean target parameter count `162`, but all pairwise dots were exactly `0.0`, and conflict/projection rates were empty.
- 판단: current heads-only lane-family training has task-specific head parameters with no shared head/adapter surface for PCGrad to balance. This branch proves the plumbing can target head groups, but it is not a candidate for broader training by itself. Do not repeat as a PCGrad task-list or `param_groups=["heads"]` sweep; training-exposure work needs actual shared zero-gated adapters/routing or a different emit contract.

Latest shared-adapter PCGrad train/eval result:

- branch/worktree: `exp/lane-family-f1/shared-adapter-pcgrad-rebalance`.
- changed axis: add opt-in zero-init shared P2/P3/P4 lane-family feature adapters, split them into a dedicated optimizer group `lane_family_adapters`, and apply PCGrad only to that shared adapter group.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly. No dataset copy was created. The smoke run was deleted, redundant task-best/last checkpoints and TensorBoard files were pruned, and the main run was reduced to `125M`, retaining `phase_4/checkpoints/best.pt`, history, summaries, and exact/broader eval exports.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_shared_adapter_pcgrad_from_lane60_lane_head_transplant_original_stop_pca_20260512_default_20260529_042113`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- PCGrad diagnostics: `1536` enabled steps over `lane_family_adapters`, mean target parameter count `24`. Unlike heads-only PCGrad, adapter conflict/projection was real; per-window lane-vs-stop conflict was generally around `0.42` to `0.62`, and projection rates were non-empty for all tasks.
- best training epoch `2`: lane/stop/cross F1 `0.5644 / 0.4561 / 0.5952`, TP/FP/FN lane `1126 / 474 / 1264`, stop-line `26 / 28 / 34`, crosswalk `50 / 37 / 31`.
- exact-val128 checkpoint eval: lane/stop/cross F1 `0.5644 / 0.4561 / 0.5952`, TP/FP/FN lane `1126 / 474 / 1264`, stop-line `26 / 28 / 34`, crosswalk `50 / 37 / 31`.
- broader-val512 checkpoint eval: lane/stop/cross F1 `0.5433 / 0.4033 / 0.6061`, TP/FP/FN lane `4286 / 2014 / 5191`, stop-line `97 / 113 / 174`, crosswalk `227 / 127 / 168`.
- 판단: the shared adapter surface is a real training-exposure intervention, not a no-op, but it still regresses against the retained broader runtime composite `0.5628 / 0.4235 / 0.6187` on all three tasks. Do not continue this exact branch as adapter LR/gate-init/depth/PCGrad-task-list/epoch-count tuning. Reopen only if the shared surface is paired with a stronger stop-line/lane emit contract or per-task adapter routing that first moves TP/FP/FN.

Latest task-specific adapter routing train/eval result:

- branch/worktree: `exp/lane-family-f1/task-specific-adapter-routing`.
- changed axis: add opt-in zero-init task-specific P2/P3/P4 lane-family feature adapters, with separate routed features for lane, stop-line, and crosswalk. This deliberately disables PCGrad and tests whether task-specific adapter exposure itself improves the current row-scan/tangent lane plus hull crosswalk contract.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly. No dataset copy was created. The smoke run was deleted, duplicate task-best/last checkpoints, TensorBoard files, and temporary YOLO weights were pruned, and the retained main run is `128M`.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_task_adapter_routing_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260529_062400`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- best training epoch `3`: objective `0.6243652225`, lane/stop/cross F1 `0.5517 / 0.4381 / 0.6300`, TP/FP/FN lane `1037 / 439 / 1246`, stop-line `23 / 29 / 30`, crosswalk `63 / 31 / 43`.
- exact-val128 checkpoint eval: objective `0.6204810631`, lane/stop/cross F1 `0.5627 / 0.4505 / 0.5854`, TP/FP/FN lane `1113 / 453 / 1277`, stop-line `25 / 26 / 35`, crosswalk `48 / 35 / 33`.
- broader-val512 checkpoint eval: objective `0.6109086167`, lane/stop/cross F1 `0.5414 / 0.4142 / 0.6119`, TP/FP/FN lane `4236 / 1936 / 5241`, stop-line `99 / 108 / 172`, crosswalk `231 / 129 / 164`.
- 판단: per-task routing is trainable and storage-clean, but it regresses below the retained broader runtime composite `0.5628 / 0.4235 / 0.6187` on lane and stop-line, and exact stop-line remains below projection-comp exact `0.5167`. Do not continue this exact branch as task-adapter LR, gate-init, depth, loss-weight, or epoch-count tuning. Reopen only with a changed lane/stop-line emit contract that first moves TP/FP/FN.

Latest stop-line axis-profile segment head train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-axis-profile-segment-head`.
- changed axis: keep the retained row-scan/tangent lane contract and hull crosswalk decode, but add a model-side stop-line axis-profile segment branch. The branch samples feature profiles along each predicted stop-line seed axis and predicts no-GT center shift, normal shift, half-length, segment confidence, and verifier score. This is not the older read-only axis-profile postprocess replay; it is a trainable emit contract.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly. No dataset copy was created. The smoke run was deleted, duplicate task-best/last checkpoints and TensorBoard files were pruned, temporary YOLO weights were removed, and the retained main run is `119M`.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_axis_profile_segment_head_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260529_070336`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- best training epoch `3`: objective `0.6311550114`, lane/stop/cross F1 `0.5514 / 0.4571 / 0.6566`, TP/FP/FN lane `1033 / 431 / 1250`, stop-line `24 / 28 / 29`, crosswalk `65 / 27 / 41`.
- exact-val128 checkpoint eval: objective `0.6185894834`, lane/stop/cross F1 `0.5568 / 0.4425 / 0.5890`, TP/FP/FN lane `1095 / 448 / 1295`, stop-line `25 / 28 / 35`, crosswalk `48 / 34 / 33`.
- 판단: the axis-profile segment head is trainable and directly targets the along-axis midpoint/extent failure, but exact stop-line remains below projection-comp exact `0.5167`, simple seeded segment-set task-best `0.4737`, and metric-quality verifier exact `0.4660`. Broader val512 was intentionally skipped because the exact gate failed. Do not continue this exact branch as radius/sample-count, aux/verifier weight, segment threshold, max-segment, head-LR, or epoch-count tuning. Reopen only with changed candidate coverage or a different no-GT midpoint/extent signal that first moves exact TP/FP/FN.

Latest stop-line dual-endpoint pair head train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-dual-endpoint-pair-head`.
- changed axis: add a model-side stop-line left/right endpoint heatmap and endpoint-offset contract, then pair top endpoint candidates at runtime using dense support along the proposed segment. This changes candidate generation, not only center/selector thresholding.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly. No dataset copy was created. The smoke run was deleted, duplicate task-best/last checkpoints and TensorBoard files were pruned, temporary YOLO weights were removed, and the retained main run is `117M`.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_dual_endpoint_pair_head_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260529_074205`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`.
- best training epoch `3`: objective `0.5891505804`, lane/stop/cross F1 `0.5503 / 0.2066 / 0.6332`, TP/FP/FN lane `1033 / 438 / 1250`, stop-line `22 / 138 / 31`, crosswalk `63 / 30 / 43`.
- exact-val128 checkpoint eval: objective `0.5857434590`, lane/stop/cross F1 `0.5619 / 0.2281 / 0.5854`, TP/FP/FN lane `1112 / 456 / 1278`, stop-line `26 / 142 / 34`, crosswalk `48 / 35 / 33`.
- 판단: the dual-endpoint target/head learns enough to produce stop-line TP, but the pair decoder has severe FP-control failure. Exact stop-line TP equals baseline exact TP `26`, while FP increases from baseline `30` to `142`; it is far below projection-comp exact `0.5167`, `31 / 29 / 29`. Broader val512 was intentionally skipped because the exact gate failed. Do not continue this exact branch as endpoint radius, pair score threshold, top-K, max-segment, aux weight, head-LR, or epoch-count tuning. Reopen only with a materially different endpoint verifier/quality contract that first suppresses FP on exact val128.

Latest stop-line endpoint-pair metric-verifier train/eval result:

- branch/worktree: `exp/lane-family-f1/stopline-endpoint-pair-metric-verifier`.
- changed axis: keep the left/right endpoint heatmap and endpoint-offset head, but disable the raw FP-heavy endpoint-pair runtime decoder. Instead, the head builds a small endpoint-pair segment set from top left/right endpoint cells, samples segment-aligned dense features, and trains a metric-quality verifier for those candidate segments. This is a changed emit/verification contract, not a pair-threshold-only replay.
- storage contract: training reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root directly. No dataset copy was created. The smoke run was deleted, duplicate task-best/last checkpoints, TensorBoard files, and temporary YOLO weights were pruned, and the retained main run is `118M`.
- main run: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_endpoint_pair_metric_verifier_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260529_081525`.
- main scale: `3` epochs, `512` train batches, `128` val batches, batch size `4`, validation epoch `2`, CUDA. Dataset split reported by the run: train `326709`, val `82641`, test `20000`; the evaluator scanned `429350` canonical records from the same dataset root.
- best training epoch `3`: objective `0.6189972068`, lane/stop/cross F1 `0.5501 / 0.4248 / 0.6300`, TP/FP/FN lane `1034 / 442 / 1249`, stop-line `24 / 36 / 29`, crosswalk `63 / 31 / 43`.
- exact-val128 checkpoint eval: objective `0.6095422894`, lane/stop/cross F1 `0.5616 / 0.3387 / 0.5818`, TP/FP/FN lane `1112 / 458 / 1278`, stop-line `21 / 43 / 39`, crosswalk `48 / 36 / 33`.
- 판단: the endpoint-pair metric verifier fixes the raw endpoint-pair FP explosion directionally (`142` exact FP down to `43`), but it loses too much TP (`26 -> 21`) and remains below baseline exact `0.4483`, simple seeded segment-set task-best `0.4737`, metric-quality verifier `0.4660`, and projection-comp exact `0.5167`. Broader val512 was intentionally skipped because the exact gate failed. Do not continue this exact branch as endpoint side top-K, verifier-score weight, metric-quality tau, segment threshold, max-segment, head-LR, or epoch-count tuning. Reopen endpoint-pair only with a materially different candidate-coverage or consensus signal that improves exact TP/FP/FN first.

Latest lane feature-ROI bounded-residual repair probe result:

- branch/worktree: `exp/lane-family-f1/lane-feature-roi-repair`.
- artifacts:
  - final train64 smoke: `runs/pv26_exhaustive_od_lane_train/lane_feature_roi_repair_train64_smoke_val4_20260530_02/summary.json`.
  - earlier 2026-05-29 replay: `runs/pv26_exhaustive_od_lane_train/lane_feature_roi_repair_replay_20260529/smoke_val4_epoch2/summary.json`.
- changed axis: no-GT lane repair replay with dense lane feature/centerline/support/tangent samples along decoded candidate polylines, but now trained as a bounded-residual replacement head with do-not-repair negatives, identity geometry loss, and mean-move FP control.
- final train64 smoke baseline lane/stop/cross F1: `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`.
- final train64 smoke repaired lane/stop/cross F1: `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`; selected repair count `2`, train examples `664` (`161` repair-positive / `503` negative).
- 판단: the TP-preserving repair contract no longer destroys already matched lanes, but it also fails to convert any near-unmatched candidates into TP on fixed val4. Exact-val128 and broader-val512 are intentionally skipped. Do not repeat this family as repair-label threshold, feature MLP size, point-loss weight, mean-move gate, or train-batch scaling unless a new confidence/instance signal first moves fixed smoke TP/FP/FN.

Latest lane area-ROI verifier probe result:

- branch/worktree: `exp/lane-family-f1/lane-area-roi-verifier`.
- artifacts:
  - fixed smoke: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train64_smoke_val4_20260530/summary.json`.
  - exact-val128: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train256_exact_val128_20260530/summary.json`.
  - train384 exact-val128: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train384_exact_val128_20260530/summary.json`.
  - train384 saved verifier: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train384_model_val4_20260530/verifier.pt`.
  - train384 four-chunk val512 aggregate: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_verifier_train384_broader_val512_chunked_aggregate_20260530/summary.json`.
  - alignment-context train64 smoke: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_alignment_context_train64_smoke_val4_20260530/summary.json`.
  - replace-nearest train64 smoke: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_replace_nearest_train64_smoke_val4_20260530/summary.json`.
  - replace-nearest saved train384 replay smoke: `runs/pv26_exhaustive_od_lane_train/lane_area_roi_replace_nearest_train384_replay_smoke_val4_20260530/summary.json`.
- changed axis: train a small no-GT MLP verifier over line-ROI dense features sampled from raw seg-first lane candidates dropped by default bbox/area filters. Candidate positives require a nearby GT lane that the baseline output did not already match; runtime selection uses only verifier probability, duplicate distance, and a fixed append cap.
- storage/runtime contract: the probe reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root and created no dataset copy. The stress-audit update added chunked validation replay plus save/load for the small verifier model so val512 can be evaluated under runtime limits without holding all raw batches in memory. Retained output sizes are `36K` smoke, `152K` train256 exact, `216K` train384 exact, `580K` saved-model val4, and about `472K` across the four val512 chunks plus `8K` aggregate.
- smoke result: baseline lane/stop/cross `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`; verifier append result `0.5915 / 0.0000 / 0.5455`, lane TP/FP/FN `42 / 14 / 44`, selected candidates `5`, selected oracle-positive `2`.
- exact-val128 result: baseline lane/stop/cross `0.5888 / 0.5333 / 0.5988`, lane TP/FP/FN `1202 / 491 / 1188`; verifier append result `0.5931 / 0.5333 / 0.5988`, lane TP/FP/FN `1253 / 582 / 1137`, selected candidates `142`, selected oracle-positive `58`.
- train384 exact-val128 result: baseline lane/stop/cross `0.5888 / 0.5333 / 0.5988`, lane TP/FP/FN `1202 / 491 / 1188`; verifier append result `0.5956 / 0.5333 / 0.5988`, lane TP/FP/FN `1249 / 555 / 1141`, selected candidates `111`, selected oracle-positive `48`.
- train384 chunked val512 aggregate: baseline lane/stop/cross `0.5749 / 0.5252 / 0.5969`, lane TP/FP/FN `4634 / 1983 / 4871`; verifier append result `0.5821 / 0.5252 / 0.5969`, lane TP/FP/FN `4816 / 2227 / 4689`, selected candidates `426`, selected oracle-positive `192`.
- alignment-context train64 smoke: baseline lane/stop/cross `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`; verifier append result `0.5816 / 0.0000 / 0.5455`, lane TP/FP/FN `41 / 14 / 45`, selected candidates `4`, selected oracle-positive `1`. This added nearest-retained-lane distance/center/overlap/angle/length context and raised feature dim to `368`, but it lost F1 on the fixed smoke gate.
- replace-nearest train64 smoke: baseline lane/stop/cross `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`; fixed-count replacement result `0.5693 / 0.0000 / 0.5455`, lane TP/FP/FN `39 / 12 / 47`, selected candidates `1`, selected oracle-positive `0`. The run trained on `440` examples (`66` positive / `374` negative).
- replace-nearest saved train384 replay smoke: baseline and repaired lane/stop/cross both `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`; selected candidates `1`, selected oracle-positive `0`.
- 판단: the learned verifier finds real recall signal in raw dropped candidates, but FP-control remains too weak even after larger train exposure, broader chunked evaluation, nearest-retained-lane alignment-context features, dense side-contrast features, and fixed-count nearest-lane replacement. Exact TP gain `+47` came with FP `+64`, val512 TP gain `+182` came with FP `+244`, alignment-context smoke TP `+1` came with FP `+3`, and replace-nearest selected no oracle-positive candidates on fixed smoke. Lane remains below `0.60`, stop-line is unchanged, and crosswalk is below `0.60` on the stress slice. Do not repeat this as quality-threshold, hidden-dim, train-batch, max-append/max-replace, duplicate-distance, bbox-area, candidate-distance, nearest-retained-lane context, side-contrast, or replace-nearest distance tuning without a materially new instance-quality/alignment signal that first improves fixed TP/FP/FN.

Latest stop-line trainset raw-patch CNN verifier probe result:

- branch/worktree: `exp/lane-family-f1/stopline-trainset-patch-verifier`.
- artifacts:
  - fixed smoke: `runs/pv26_exhaustive_od_lane_train/stopline_trainset_patch_verifier_train64_smoke_val4_20260530_02/summary.json`.
  - exact-val128: `runs/pv26_exhaustive_od_lane_train/stopline_trainset_patch_verifier_train256_exact_val128_20260530/summary.json`.
- changed axis: train a 2-channel raw/gradient patch CNN verifier on stop-line projection-comp candidate rows collected from canonical train batches, then replay one train-selected task threshold on validation candidate rows. Runtime selection uses no GT; GT is used only for candidate labels and audit metrics.
- storage contract: the probe reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root, wrote only CSV/summary artifacts, and created no checkpoint or dataset copy. Retained output sizes are about `4.9M` for smoke and `22M` for exact.
- smoke result: train selected threshold improved train stop-line `0.6617 -> 0.7398`, TP/FP/FN `88 / 27 / 63 -> 91 / 4 / 60`; val4 had no oracle-positive candidate rows, so stop-line stayed `0 / 2 / 2` after one FP was removed.
- exact-val128 result:
  - train baseline stop-line `0.6000`, TP/FP/FN `351 / 168 / 300`;
  - train threshold replay stop-line `0.7408`, TP/FP/FN `383 / 0 / 268`;
  - val baseline/projection-comp stop-line `0.5333`, TP/FP/FN `32 / 28 / 28`;
  - val train-threshold replay stop-line `0.4160`, TP/FP/FN `26 / 39 / 34`.
- 판단: train-split exposure does not fix raw-patch verifier generalization. The learned selector overfits train candidates, loses `6` exact validation TP, adds `11` FP, and increases FN by `6`, so broader-val512 is intentionally skipped. Do not repeat this as train-batch, epoch, LR, top-K, threshold-grid, patch-size, CNN-depth, or same train-threshold replay tuning without a materially different candidate-generation or verification contract.

Latest stop-line raw-Hough candidate-generation probe result:

- branch/worktree: `exp/lane-family-f1/stopline-raw-hough-candidates`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_raw_hough_candidates_train64_smoke_val4_20260530/summary.json`.
- changed axis: use raw-image Canny/Hough line segments as new stop-line candidates, then score each segment using dense stop-line mask/center/selector support and train a small no-GT MLP verifier from canonical train candidates. This is candidate-generation, not another projection-comp threshold replay and not the older raw-axis stripe midpoint readout.
- storage contract: the probe reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root, wrote only CSV/summary artifacts, and created no checkpoint or dataset copy. Retained output size is about `3.2M`.
- smoke setup: `64` train batches, `4` validation batches, validation epoch `2`, Hough top-k `8`, max raw-Hough candidates `16`, verifier epochs `40`.
- smoke train candidate stats: train candidates `2048`, train oracle-positive `117`.
- smoke val candidate stats: val candidates `128`, val oracle-positive `0`.
- smoke train result:
  - baseline stop-line `0.6541`, TP/FP/FN `87 / 28 / 64`;
  - raw-Hough-only `0.0997`, TP/FP/FN `18 / 192 / 133`;
  - baseline-plus-raw-Hough `0.4794`, TP/FP/FN `93 / 144 / 58`.
- smoke val4 result:
  - baseline stop-line `0.0000`, TP/FP/FN `0 / 3 / 2`;
  - raw-Hough-only `0.0000`, TP/FP/FN `0 / 8 / 2`;
  - baseline-plus-raw-Hough `0.0000`, TP/FP/FN `0 / 8 / 2`.
- 판단: raw-Hough candidate generation creates some train TP, but FP grows much faster and the fixed validation smoke has no oracle-positive Hough candidates while adding FP. Exact-val128 and broader-val512 are intentionally skipped. Do not repeat this as Canny threshold, Hough threshold, minLineLength, maxLineGap, Hough top-k, MLP epoch/LR, dense-score weight, or train-threshold tuning. Reopen raw-image candidate generation only with a materially different candidate-quality/geometry contract that first improves fixed smoke TP/FP/FN.

Latest stop-line raw-LSD candidate-generation probe result:

- branch/worktree: `exp/lane-family-f1/stopline-raw-lsd-candidates`.
- smoke artifact: `runs/pv26_exhaustive_od_lane_train/stopline_raw_lsd_candidates_train64_smoke_val4_20260530/summary.json`.
- exact artifact: `runs/pv26_exhaustive_od_lane_train/stopline_raw_lsd_candidates_train64_exact_val128_20260530/summary.json`.
- changed axis: use OpenCV LSD line segments as the raw-image candidate generator while keeping dense stop-line map features and the train-split MLP verifier replay fixed. This tests candidate generation, not Hough threshold tuning.
- storage contract: the probe reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root, wrote only CSV/summary artifacts, and created no checkpoint or dataset copy. Retained output sizes are about `3.7M` for smoke and `11M` for exact.
- smoke setup: `64` train batches, `4` validation batches, validation epoch `2`, LSD top-k `8`, max raw-LSD candidates `16`, verifier epochs `60`.
- smoke train candidate stats: train candidates `2048`, train oracle-positive `53`.
- smoke val candidate stats: val candidates `128`, val oracle-positive `0`.
- smoke train result:
  - baseline stop-line `0.6617`, TP/FP/FN `88 / 27 / 63`;
  - raw-LSD-only `0.0942`, TP/FP/FN `17 / 193 / 134`;
  - baseline-plus-raw-LSD `0.4794`, TP/FP/FN `93 / 144 / 58`.
- smoke val4 result:
  - baseline stop-line `0.0000`, TP/FP/FN `0 / 3 / 2`;
  - raw-LSD-only `0.0000`, TP/FP/FN `0 / 6 / 2`;
  - baseline-plus-raw-LSD `0.0000`, TP/FP/FN `0 / 6 / 2`.
- exact-val128 candidate stats: val candidates `4096`, val oracle-positive `33`.
- exact-val128 result:
  - baseline stop-line `0.5333`, TP/FP/FN `32 / 28 / 28`;
  - raw-LSD-only `0.0663`, TP/FP/FN `6 / 115 / 54`;
  - baseline-plus-raw-LSD `0.3200`, TP/FP/FN `32 / 108 / 28`.
- 판단: raw-LSD has some exact candidate coverage but cannot select it with the fixed no-GT verifier; the baseline-plus replay preserves TP but adds `+80` FP. Broader-val512 is intentionally skipped. Do not repeat this as LSD refine mode, blur/Canny edge support, LSD top-k, MLP epoch/LR, dense-score weight, or train-threshold tuning.

Latest stop-line raw-support-PCA candidate-generation probe result:

- branch/worktree: `exp/lane-family-f1/stopline-raw-support-pca-candidates`.
- smoke artifact: `runs/pv26_exhaustive_od_lane_train/stopline_raw_support_pca_candidates_train64_smoke_val4_20260530/summary.json`.
- exact artifact: `runs/pv26_exhaustive_od_lane_train/stopline_raw_support_pca_candidates_train64_exact_val128_20260530/summary.json`.
- changed axis: restrict raw-image brightness/edge evidence by predicted stop-line mask/proposal support, connected-component it, and fit each component with weighted PCA endpoints. This tests dense-support-limited raw candidate generation, not Hough/LSD threshold tuning.
- storage contract: the probe reused the existing `seg_dataset/pv26_exhaustive_od_lane_dataset` root, wrote only CSV/summary artifacts, and created no checkpoint or dataset copy. Retained output sizes are about `256K` for smoke and `368K` for exact.
- setup: `64` train batches, validation epoch `2`, max support-PCA candidates `16`, verifier top-k `8`, verifier epochs `60`, device `cuda:0`.
- smoke candidate stats: train candidates `258`, train oracle-positive `7`, val candidates `5`, val oracle-positive `0`.
- smoke val4 result:
  - baseline stop-line `0.0000`, TP/FP/FN `0 / 3 / 2`;
  - raw-support-PCA-only `0.0000`, TP/FP/FN `0 / 0 / 2`;
  - baseline-plus-raw-support-PCA `0.0000`, TP/FP/FN `0 / 3 / 2`.
- exact-val128 candidate stats: val candidates `126`, val oracle-positive `4`.
- exact-val128 result:
  - baseline stop-line `0.5333`, TP/FP/FN `32 / 28 / 28`;
  - raw-support-PCA-only `0.0000`, TP/FP/FN `0 / 16 / 60`;
  - baseline-plus-raw-support-PCA `0.4812`, TP/FP/FN `32 / 41 / 28`.
- 판단: support-PCA is too conservative to cover validation GT and the no-GT verifier does not convert any exact-val128 TP. Baseline union only adds FP, so broader-val512 is intentionally skipped. Do not repeat this as support threshold, morphology, PCA percentile, verifier epoch/LR, top-k, or score-threshold tuning.

Latest lane task-mask context gate:

- branch/worktree: `exp/lane-family-f1/lane-task-mask-context-gate`.
- code commit: `15e3fb0`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_task_mask_context_val512_epoch2/summary.json`.
- changed axis: keep the current flip-centerline lane runtime path and stop-line/crosswalk postprocess contract, then suppress only lane centerline logits where the predicted crosswalk mask is confident.
- val512 reference `flip_centerline_avg`: objective `0.6216194906`, lane/stop/cross F1 `0.5577 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 2206 / 4959`.
- val512 best `flip_centerline_avg_lane_cross_comp050`: objective `0.6230558331`, lane/stop/cross F1 `0.5628 / 0.4235 / 0.6187`, lane TP/FP/FN `4532 / 2097 / 4945`.
- 판단: this is a retained partial positive and the current objective-best broader runtime composite. It narrows the lane gap but does not close lane `0.60`, and it does nothing for the stop-line bottleneck.

Latest lane task-conflict negative-loss smoke:

- branch/worktree: `exp/lane-family-f1/lane-task-conflict-negative-loss-smoke`.
- code commit: `975d619`.
- changed axis: keep the training schedule, source checkpoint, heads-only phase-4 probe, and core centerline/crosswalk-retain contract fixed, then add only an opt-in lane seg-first auxiliary that penalizes lane centerline probability on GT crosswalk ignore pixels. The main lane loss still ignores stop-line/crosswalk masks.
- implementation: added `lane_segfirst_task_conflict_negative_mode`, `lane_segfirst_task_conflict_negative_weight`, and `lane_segfirst_task_conflict_negative_margin` to train config and `PV26MultiTaskLoss`; default is disabled.
- verification: py_compile passed for the touched train/loss/probe/test files; focused pytest passed with `68` tests; real CUDA train/val smoke completed from the retained phase-4 checkpoint.
- same-slice val4 reference (`core_centerline_refine_cross_retain`, 1 epoch, 32 train batches, 4 val batches): phase objective `0.6250663426`; lane/stop/cross F1 `0.5156 / 0.0000 / 0.7273`; lane TP/FP/FN `33 / 16 / 46`; stop-line TP/FP/FN `0 / 1 / 2`; crosswalk TP/FP/FN `4 / 2 / 1`.
- same-slice crosswalk-conflict negative result: phase objective `0.6250358501`; lane/stop/cross F1 `0.5156 / 0.0000 / 0.7273`; lane TP/FP/FN `33 / 16 / 46`; stop-line TP/FP/FN `0 / 1 / 2`; crosswalk TP/FP/FN `4 / 2 / 1`.
- 판단: the opt-in loss path is runtime-safe, but the first controlled smoke is assignment-flat and slightly lower on phase objective. Do not broaden this branch or repeat it as a source/weight/margin sweep without a new assignment-moving signal.

Latest lane temporal-neighbor union smoke:

- branch/worktree: `exp/lane-family-f1/lane-temporal-neighbor-union-smoke`.
- changed axis: keep the checkpoint and `flip_centerline_avg` lane path fixed, decode immediate `sample_id` neighbor frames, and add only neighbor lane candidates that are supported by the current frame centerline map and are not near-duplicates of current lanes.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_temporal_neighbor_union_smoke_20260514/analysis_exports/smoke_val4_epoch2/summary.json`.
- smoke val4 support: all `16` sampled validation frames had at least one immediate temporal neighbor; `105` neighbor candidates were audited and `6` were selected by the fixed no-GT support/dedupe rule.
- baseline lane/stop/cross F1: `0.5899 / 0.0000 / 0.5455`, lane TP/FP/FN `41 / 12 / 45`.
- temporal-neighbor result: lane/stop/cross F1 `0.5655 / 0.0000 / 0.5455`, lane TP/FP/FN `41 / 18 / 45`.
- selected candidates that would match a baseline FN: `0 / 6`.
- 판단: immediate-frame lane union adds FP without recovering TP on the smoke slice. Do not broaden to val128/val512 or repeat as neighbor gap, center-threshold, dedupe-distance, or add-cap tuning without a new alignment/FP-control signal.

Latest lane attribute-agnostic duplicate suppression smoke:

- branch/worktree: `exp/lane-family-f1/lane-attr-agnostic-duplicate-suppression-smoke`.
- changed axis: keep the same retained checkpoint, source run, `flip_centerline_avg` lane path, stop-line settings, and hull crosswalk fixed, then replay one fixed `24px` lane duplicate suppression pass that ignores `class_name` and `lane_type`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_attr_agnostic_duplicate_suppression_20260514/analysis_exports/smoke_val4_epoch2/summary.json`.
- smoke val4 result: `suppressed_count=0`, `suppressed_cross_schema_count=0`, `suppressed_would_match_gt_count=0`.
- baseline lane/stop/cross F1 stayed `0.5899 / 0.0000 / 0.5455`; lane TP/FP/FN stayed `41 / 12 / 45`.
- 판단: cross-attribute near-duplicates are not present on the smoke slice, so this is a no-op. Do not broaden to val128/val512 or repeat as attribute-agnostic duplicate-distance/schema/tie-break tuning.

Latest stop-line scale dense TTA smoke:

- branch/worktree: `exp/lane-family-f1/stopline-scale-dense-tta-smoke`.
- code commit: `26a7f7e`.
- changed axis: keep checkpoint, decode thresholds, current flip-centerline lane/crosswalk competition, and hull crosswalk fixed, then average only stop-line score maps, geometry maps, or both across resized input scales `0.875` and `1.125`.
- smoke val4 support: lane/stop/cross `86 / 2 / 7`.
- val4 reference `flip_centerline_avg_lane_cross_comp050`: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, stop-line TP/FP/FN `0 / 3 / 2`.
- val4 `flip_centerline_avg_lane_cross_comp050_stop_scale_score_avg`: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, stop-line TP/FP/FN `0 / 3 / 2`.
- val4 `flip_centerline_avg_lane_cross_comp050_stop_scale_geometry_avg`: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, stop-line TP/FP/FN `0 / 3 / 2`.
- val4 `flip_centerline_avg_lane_cross_comp050_stop_scale_all_avg`: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, stop-line TP/FP/FN `0 / 3 / 2`.
- 판단: stop-line scale dense TTA did not move TP/FP/FN even on the smoke slice. Do not broaden to val128/val512 or repeat as a scale-factor/map-subset sweep without a new no-GT stop-line signal.

Latest lane endpoint support-extension smoke:

- branch/worktree: `exp/lane-family-f1/lane-endpoint-support-extension-smoke`.
- code commit: `1cc9c5a`.
- changed axis: keep the current flip-centerline + fixed crosswalk-mask lane gate and stop-line/crosswalk contract fixed, then extend decoded lane endpoints only when predicted lane centerline and support maps continue past the endpoint.
- smoke val4 reference `flip_centerline_avg_lane_cross_comp050`: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`, lane mean point distance `13.3898`.
- smoke val4 support-extension: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`, lane mean point distance `12.5226`.
- movement: `8` lanes extended, `24` endpoint points added.
- 판단: support-conditioned endpoint extension moved geometry and improved mean point distance slightly, but did not change lane TP/FP/FN. Do not broaden or repeat as endpoint support threshold, step, or max-length tuning without a new signal that first changes assignment metrics.

Latest lane centerline thinning smoke:

- branch/worktree: `exp/lane-family-f1/lane-centerline-thinning-smoke`.
- code commit: `6c3767b`.
- changed axis: keep the current flip-centerline + fixed crosswalk-mask lane gate and stop-line/crosswalk contract fixed, then apply Zhang-Suen thinning to the merged lane centerline probability before vectorization.
- smoke val4 reference `flip_centerline_avg_lane_cross_comp050`: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`, lane mean point distance `13.3898`.
- smoke val4 thinning `flip_centerline_avg_lane_cross_comp050_thin035`: lane/stop/cross F1 `0.5185 / 0.0000 / 0.5455`, lane TP/FP/FN `35 / 14 / 51`, lane mean point distance `15.3421`.
- 판단: centerline skeletonization is negative on the smoke slice: it removes TP, adds FP, worsens FN, and degrades geometry distance. Do not broaden or repeat as thinning threshold, morphology, or skeletonization-kernel tuning without a new assignment-moving signal.

Latest lane component-polyfit vectorizer smoke:

- branch/worktree: `exp/lane-family-f1/lane-component-polyfit-smoke`.
- code commit: `b8f8ec8`.
- changed axis: keep the current flip-centerline + fixed crosswalk-mask lane gate and stop-line/crosswalk contract fixed, then replace row-scan tangent lane vectorization with one fixed quadratic polyfit readout per connected centerline component.
- smoke val4 reference `flip_centerline_avg_lane_cross_comp050`: lane/stop/cross F1 `0.5839 / 0.0000 / 0.5455`, lane TP/FP/FN `40 / 11 / 46`, lane mean point distance `13.3898`.
- smoke val4 component-polyfit: lane/stop/cross F1 `0.4923 / 0.0000 / 0.5455`, lane TP/FP/FN `32 / 12 / 54`, lane mean point distance `11.6325`.
- 판단: component polyfit improves matched-point distance for the remaining matched lanes but destroys assignment recall. Do not broaden or repeat as polynomial degree, row-stride, or component-size tuning without a new TP-preserving assignment signal.

Latest stop-line lane-crossing extent readout:

- branch/worktree: `exp/lane-family-f1/stopline-lane-extent-readout`.
- code commit: `938d00d`.
- artifact smoke: `analysis_exports/stopline_lane_extent_readout_smoke_val4_epoch2/variants.csv`.
- artifact exact: `analysis_exports/stopline_lane_extent_readout_val128_epoch2/variants.csv`.
- changed axis: keep the fixed candidate pool and lane-context ranking, then infer stop-line candidate extent from predicted lane crossings along the candidate axis. This tests no-GT midpoint/extent recovery, not another threshold sweep.
- exact val128 baseline stop-line F1: `0.4483`, TP/FP/FN `26 / 30 / 34`.
- exact val128 lane-cross c2 reference: `0.3448`, TP/FP/FN `15 / 12 / 45`.
- exact val128 lane-extent c2: `0.0690`, TP/FP/FN `3 / 24 / 57`.
- 판단: predicted lane crossings do not preserve TP after geometry repair. The variant emits the same `27` stop-lines as lane-cross c2 but collapses TP from `15` to `3`, so do not repeat this as a lane-crossing distance, margin, or top-k sweep.

Latest stop-line fit-far visual + task-mask competition smoke:

- branch/worktree: `exp/lane-family-f1/stopline-mask-task-competition-smoke`.
- code commit: `f20f494`.
- artifacts:
  - `analysis_exports/stopline_readout_component_audit_val128_epoch2/summary.json`
  - `analysis_exports/stopline_fit_far_visual_audit_val128_epoch2/manifest.json`
  - `analysis_exports/stopline_mask_task_competition_smoke_val128_epoch2/summary.json`
- changed axis: keep checkpoint and postprocess fixed, first inspect production-missed stop-line GTs with retained dense mask/center evidence, then test whether stop-line mask logits should be suppressed by crosswalk or lane task probabilities before decode.
- read-only audit: val128 stop-line GT `60`; production TP `26`; GT tube mask max `>=0.50` for `51 / 60`; GT tube center max `>=0.50` for `50 / 60`; no-anchor fit close `34 / 60`; anchored fit close `33 / 60`.
- exact val128 baseline lane/stop/cross F1: `0.5267 / 0.4483 / 0.5854`, stop-line TP/FP/FN `26 / 30 / 34`.
- best task-mask competition: crosswalk-only suppression strength `1.0`, lane/stop/cross F1 `0.5267 / 0.4615 / 0.5854`, stop-line TP/FP/FN `27 / 30 / 33`.
- lane-inclusive suppression collapses stop-line recall: best lane-inclusive row has stop-line F1 `0.1429`, TP/FP/FN `6 / 18 / 54`.
- 판단: visual evidence confirms dense signal often exists, but task-mask competition only recovers one TP and remains below known stop-line references. Do not broaden or repeat as strength/source/mask-threshold sweeps. With no new stop-line premise from this pass, pivot back to lane instance-stability work while preserving the current stop-line/crosswalk retention contract.

Latest stop-line local-x auxiliary smoke:

- branch/worktree: `exp/lane-family-f1/stopline-local-x-aux-smoke`.
- code commit: `12eb157`.
- artifact: `analysis_exports/stopline_local_x_aux_smoke_val64_epoch1/summary.json`.
- changed axis: keep the source stage-4 freeze policy, LR, and loss weights fixed, then add only `stopline_local_x_aux_weight=0.5`.
- smoke setup: `1` epoch, `128` train batches, `64` validation batches, seed checkpoint evaluated on the same val64 slice.
- same-val64 seed baseline lane/stop/cross F1: `0.5469 / 0.2000 / 0.6923`, phase objective `0.6050`.
- local-x aux epoch1 lane/stop/cross F1: `0.5196 / 0.2222 / 0.6988`, phase objective `0.5901`.
- 판단: local-x auxiliary gives only `+0.0222` stop-line F1 on a low-support smoke slice while losing `-0.0273` lane F1 and `-0.0148` objective. Do not broaden or repeat as a local-x aux weight-only/schedule sweep.

Latest stop-line centerline center-target smoke:

- branch/worktree: `exp/lane-family-f1/stopline-centerline-center-target-smoke`.
- code commit: `55758ab`.
- artifact: `analysis_exports/stopline_centerline_center_target_smoke_val64_epoch1/summary.json`.
- changed axis: keep the same stage-4 freeze policy, LR, and loss weights, then set only `stopline_center_target_mode=centerline`.
- smoke setup: `1` epoch, `128` train batches, `64` validation batches, seed checkpoint evaluated on the same val64 slice.
- same-val64 seed baseline lane/stop/cross F1: `0.5469 / 0.2000 / 0.6923`, phase objective `0.6050`.
- centerline target epoch1 lane/stop/cross F1: `0.5202 / 0.1455 / 0.6988`, phase objective `0.5812`.
- 판단: centerline center-target supervision worsens the actual stop-line metric and also drops lane/objective. Do not broaden or repeat as a target-mode sweep.

Latest stop-line no-oracle axis-offset budget:

- branch/worktree: `exp/lane-family-f1/stopline-axis-offset-budget`.
- code commit: `68585c1`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_axis_offset_budget_20260513/analysis_exports/val512_epoch2/summary.json`.
- changed axis: keep the projection-competition reference fixed, then replace only positive-no-oracle local candidates with GT-only axis-projection oracle variants to separate along-line center error, normal error, and length error.
- local positive-no-oracle rows: `51`; axis-dominant rows: `49`; abs-normal-offset q50/q90: `2.31px / 13.47px`; abs-along-offset q50/q90: `68.59px / 137.64px`.
- projection reference stop-line F1: `0.5164`, TP/FP/FN `126 / 91 / 145`.
- axis-projection + fixed minlen best (`minlen=240`) stop-line F1: `0.5547`, TP/FP/FN `137 / 86 / 134`.
- axis-projection + GT-length oracle stop-line F1: `0.6559`, TP/FP/FN `162 / 61 / 109`, close to full GT-midpoint+GT-length oracle `0.6599`.
- 판단: no-oracle local candidates are mostly wrong along the stop-line axis, not off the line. But fixed-length axis shift is still below `0.60`; a real production path needs a no-GT signal for both along-axis midpoint shift and stop-line extent/length, not another selector-only or symmetric extension sweep.

Latest stop-line axis-projected offset readout:

- branch/worktree: `exp/lane-family-f1/stopline-axis-projected-offset-readout`.
- code commit: `107a0f5`.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/stopline_axis_projected_offset_readout_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: keep the same predicted proposal + angle-mask extent readout, but project the existing predicted center offset onto the predicted stop-line angle axis before decoding.
- exact val128 existing pred-offset reference: stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`, mean point distance `13.62`.
- exact val128 axis-projected offset: stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`, mean point distance `13.51`.
- 판단: current model center-offset already gives the same matched set after axis projection; the tiny point-distance improvement is not a stop-line F1 path. Do not broaden or repeat this as a `top_k`/threshold sweep.

Latest stop-line axis-profile readout:

- branch/worktree: `exp/lane-family-f1/stopline-axis-profile-readout`.
- tool: `tools/probe_pv26_stopline_axis_profile_readout.py`.
- changed axis: keep the existing checkpoint fixed, replace only stop-line predictions, anchor on predicted proposal cells or predicted offset, then read midpoint/length from the predicted mask profile along the predicted stop-line axis.
- val4 smoke: baseline and all profile variants had stop-line F1 `0.0000`; support was only TP/FP/FN `0 / 3 / 2` or `0 / 4 / 2`, so val128 was needed before closing the axis.
- exact val128 best: `axis_profile_cell_top1_s060_mask050_band4` and `axis_profile_offset_top1_s060_mask050_band4` both reached stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`; baseline was `0.4483`, TP/FP/FN `26 / 30 / 34`.
- 판단: proposal-cell axis profile recovers the same matched set as the existing predicted angle/mask-extent and axis-projected-offset references, but does not beat the known exact stop-line references (`0.5085` / PCA `0.5133`) or the broader projection-competition reference `0.5164`. Do not broaden or repeat as a proposal-source/top-k/mask-threshold/normal-band sweep.

Latest stop-line axis-window recenter readout:

- branch/worktree: `exp/lane-family-f1/stopline-axis-window-recenter`.
- code commit: `e4b6dd2`.
- changed axis: slide the candidate center along the predicted stop-line axis with a fixed mask/proposal line-support window, then reuse the existing mask-extent decoder.
- exact val128: `axiswin_extent_max_top3_s040_h16_r24_step4_fallback` reached stop-line F1 `0.4306`, TP/FP/FN `31 / 53 / 29`, below baseline `0.4483` and selector reference `0.5085`.
- 판단: axis-window center selection fired (`axis_window_center_ok=215`), but it added too many FP and lost to existing references. Do not broaden or repeat as a radius/step/scoring-length/top-K/proposal-threshold/fallback sweep.

Latest stop-line symmetric axis-profile readout:

- branch/worktree: `exp/lane-family-f1/stopline-symmetric-axis-profile-readout`.
- code commit: `f8bf316`.
- artifact smoke: `runs/pv26_exhaustive_od_lane_train/stopline_symmetric_axis_profile_20260513/analysis_exports/smoke_val4_epoch2/summary.json`.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/stopline_symmetric_axis_profile_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: keep the proposal-cell anchor as the stop-line midpoint, then use the mask profile only to choose a symmetric half-extent along the predicted axis.
- smoke val4: all profile/symmetric variants stayed at stop-line F1 `0.0000`, TP/FP/FN `0 / 3 / 2` or fallback `0 / 4 / 2`, so exact val128 was needed.
- exact val128: existing profile top1 stayed `0.5085`, TP/FP/FN `30 / 28 / 30`; symmetric top1 dropped to `0.2203`, TP/FP/FN `13 / 45 / 47`; symmetric top3 was `0.2185`, TP/FP/FN `13 / 46 / 47`.
- 판단: the mask-profile asymmetry was not just harmful midpoint drift; forcing the proposal cell to be midpoint loses too many true positives and adds false positives. Do not broaden or repeat as a symmetric extent/top-k/proposal-threshold/normal-band sweep.

Latest stop-line axis score-profile readout:

- branch/worktree: `exp/lane-family-f1/stopline-axis-score-profile-readout`.
- code commit: `a6bb715`.
- artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_axis_score_profile_20260513/analysis_exports/{smoke_val4_epoch2,val128_epoch2,val512_epoch2}/summary.json`.
- changed axis: keep the existing proposal-cell axis-profile readout fixed, but weight along-axis projection quantiles with `center`, `selector`, or fused score maps instead of mask probabilities.
- exact val128: selector-profile top1 reached stop-line F1 `0.5254`, TP/FP/FN `31 / 27 / 29`, beating the mask-profile top1 reference `0.5085`, `30 / 28 / 30`.
- broader val512: the same selector-profile top1 fell to stop-line F1 `0.4303`, TP/FP/FN `105 / 112 / 166`; best score-profile run was still only mask-profile top3 at `0.4531`, TP/FP/FN `111 / 108 / 160`, far below projection-competition `0.5164`, `126 / 91 / 145`.
- 판단: score-profile weighting was an exact-split small positive but does not generalize to broader validation. Do not repeat it as a center/selector/fused profile-source or top-k/threshold/normal-band sweep.

Latest stop-line fragment axis contract:

- branch/worktree: `exp/lane-family-f1/stopline-fragment-axis-contract`.
- model-side code commit: `6cc3f52`; low-disk metric-only helper commit: `ae1fbc2`.
- changed axis: train the fragment-center offset as a stop-line-axis scalar instead of 2D xy offset, and decode fragment extent with the same axis projection.
- exact val128 metric-only run completed without checkpoint/TensorBoard writes; run size was `2.5MB`, checkpoint paths were `null`, and skipped steps were `0`.
- epoch1 lane/stop/cross F1: `0.5081 / 0.0625 / 0.6835`, objective `0.5351`.
- epoch2 lane/stop/cross F1: `0.5225 / 0.1905 / 0.5714`, objective `0.5522`.
- 판단: this closes the axis-scalar fragment contract as a performance negative. It fixed the disk-full observability problem, but the actual stop-line result is far below tangent-link exact `0.4483`, PCA val128 `0.5133`, angle-mask production `0.5085`, and projection-competition broader reference `0.5164`; do not broaden or repeat it as an aux-weight/top-k/min-score/epoch sweep.

Latest stop-line axis-support span audit:

- branch/worktree: `exp/lane-family-f1/stopline-axis-support-span-audit`.
- code commit: `3856bbe`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_axis_support_span_audit_20260513/analysis_exports/val512_epoch2/summary.json`.
- changed axis: keep the projection-competition reference fixed, then replace positive-no-oracle local candidates with a no-GT same-axis support span built from same-sample high-score candidates.
- projection-competition reference: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- best support-span replay (`minmembers=8`): stop-line F1 `0.5085`, TP/FP/FN `119 / 78 / 152`.
- support span length ratio q50 improves from `0.544` to `1.013`, but midpoint distance q50 only moves `68.59px -> 61.92px` and q90 worsens to `215.68px`.
- 판단: same-axis support span reduces FP only by dropping too many TP; it is not the missing no-GT midpoint/extent signal. Do not repeat as a top-k/min-score/member-count/angle/normal-threshold sweep.

Latest stop-line raw-stripe midpoint audit:

- branch/worktree: `exp/lane-family-f1/stopline-raw-stripe-midpoint-audit`.
- code commit: `5264e23`.
- artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_raw_stripe_midpoint_audit_20260513/analysis_exports/{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- changed axis: keep the current checkpoint, decoded candidates, and production postprocess fixed; sample raw-image center-stripe versus side-band contrast along each candidate axis, then replay one fixed top-k confidence raw-stripe replacement.
- exact val128 baseline: stop-line F1 `0.4483`, TP/FP/FN `26 / 30 / 34`, mean point distance `17.92`.
- exact val128 raw-stripe replay: stop-line F1 `0.0685`, TP/FP/FN `5 / 81 / 55`, mean point distance `24.77`.
- feature summary: `382 / 405` candidate rows produced a raw-valid stripe, but raw-improved-to-positive rows were `0`; nearest-GT distance q50 worsened from `36.18px` to raw-repaired `156.92px`.
- 판단: raw-image stripe contrast is not the missing no-GT along-axis midpoint/extent signal for current stop-line candidates. Do not broaden to val512 or repeat as raw-stripe top-k/confidence/band/smoothing/threshold sweeps without a materially new non-photometric FP-control or midpoint source.

Latest stop-line flip-consensus readout:

- branch/worktree: `exp/lane-family-f1/stopline-flip-consensus-readout`.
- code commit: `8bbde1b`.
- artifacts: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_flip_consensus_{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- changed axis: keep the checkpoint fixed, decode normal and horizontal-flip stop-line candidate pools separately, then emit only candidates whose decoded raw geometry agrees within `40px`; the second variant averages matched points.
- smoke val4: both consensus variants were identical to baseline stop-line F1 `0.0000`, TP/FP/FN `0 / 3 / 2`.
- exact val128: baseline stop-line F1 `0.4483`, TP/FP/FN `26 / 30 / 34`; best consensus `0.4667`, TP/FP/FN `28 / 32 / 32`.
- geometry: point averaging improved mean stop-line distance `17.92 -> 14.35` and angle error `2.12 -> 1.66`, but the matched set stayed only `28` TP.
- 판단: candidate-level flip agreement is a small exact-only positive but remains below PCA val128 `0.5133`, angle-mask production `0.5085`, and broader projection-competition `0.5164`. Do not broaden or repeat as agreement-distance/top-k/point-average tuning without a new TP-preserving candidate-generation signal.

Latest stop-line flip-union readout:

- branch/worktree: `exp/lane-family-f1/stopline-flip-union-readout`.
- code commit: `c0509f7`.
- artifacts: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_flip_union_{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- changed axis: keep the fixed checkpoint and decoded normal/flip candidate pools, but emit flip-only or normal+flip union candidates instead of requiring candidate agreement.
- smoke val4: all union variants still had stop-line F1 `0.0000`; they only increased FP from baseline `3` to `4-8`.
- exact val128: best consensus remained `0.4667`, TP/FP/FN `28 / 32 / 32`; baseline was `0.4483`, `26 / 30 / 34`.
- union result: `flip_only_top1` reached only `0.4306`, TP/FP/FN `31 / 53 / 29`; `normal_flip_union_top1` reached `0.4304`, `34 / 64 / 26`; `normal_flip_union_top2` raised TP to `37` but FP to `136`, dropping F1 to `0.3176`.
- 판단: flip pass does contain some extra TP candidates, but not a TP-preserving candidate-generation signal. Union variants expand FP faster than recall, so do not broaden or repeat as flip-only/normal+flip top-k or component-count tuning without a new FP-control signal.

Latest lane FN recovery audit:

- branch/worktree: `exp/lane-family-f1/lane-fn-nearby-fp-recovery-audit`.
- code commit: `978fc88`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_fn_recovery_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: no training and no production decoder change; replay the current flip-centerline broader composite, then inspect each missed GT lane for predicted centerline evidence on the GT polyline and nearby unmatched row-scan tracks.
- baseline lane/stop/cross F1: `0.5577 / 0.4235 / 0.6187`.
- lane TP/FP/FN: `4518 / 2206 / 4959`.
- FN evidence: `2066 / 4959` missed lanes have `gt_center_point_mean >= 0.30`; `1363 / 4959` have `>= 0.50`; `1698 / 4959` have an unmatched predicted lane within `120px`; `2588 / 4959` satisfy `center_mean >= 0.30` or unmatched distance `<=120px`.
- no-new-FP upper-bound control: recovering the `center_mean >= 0.30 or unmatched <=120px` subset would imply lane F1 `0.7564`, but this is diagnostic only and uses GT to count recoverable FNs.
- 판단: lane still has recall-side headroom that is not explained by another FP selector threshold. The next lane branch should convert this into a recall-preserving decoder/model-side instance recovery contract; do not claim this read-only upper-bound as production lane success.

Latest lane FN joint-strata audit:

- branch/worktree: `exp/lane-family-f1/lane-fn-joint-strata-audit`.
- code commit: `13bb290`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_fn_joint_strata_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: read existing broader-val512 FN rows and split recovery headroom by GT centerline evidence and nearest unmatched predicted track distance.
- lane F1 `0.60` at current FP requires `489` recovered FN.
- `center_mean >= 0.50 AND unmatched <=80px`: `563` FN, no-new-FP upper-bound lane F1 `0.6062`.
- `center_mean >= 0.50 AND unmatched <=120px`: `836` FN, no-new-FP upper-bound lane F1 `0.6285`.
- `center_mean >= 0.50 WITHOUT unmatched <=120px`: `527` FN, no-new-FP upper-bound lane F1 `0.6032`.
- `unmatched <=120px WITHOUT center_mean >=0.50`: `862` FN, no-new-FP upper-bound lane F1 `0.6306`.
- 판단: the lane budget is not one bucket. Both nearby-track repair and centerline-only generation are individually large enough on GT-labeled upper bounds, but the centerline-snap smoke shows that simple local x-snapping is not the production contract.

Latest lane FN pair-geometry audit:

- branch/worktree: `exp/lane-family-f1/lane-fn-pair-geometry-audit`.
- code commit: `941ddd8`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_fn_pair_geometry_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: replay the same current flip-centerline composite, but add nearest-prediction geometry features for each FN: length ratio, angle error, center offset, endpoint distance, sample distance, and overlap.
- `center>=0.50 AND unmatched<=80px`: count `563`, nearest distance q50 `53.25px`, length ratio q50 `0.974`, angle error q50 `1.27deg`, center distance q50 `50.11px`, y-overlap q50 `0.905`.
- `center>=0.50 AND unmatched<=120px`: count `836`, nearest distance q50 `64.31px`, length ratio q50 `0.881`, angle error q50 `1.53deg`, center distance q50 `59.98px`, y-overlap q50 `0.858`.
- `center>=0.50 WITHOUT unmatched<=120px`: count `527`, nearest distance q50 `186.96px`, length ratio q50 `2.091`, center distance q50 `182.80px`, y-overlap q50 `0.332`.
- 판단: the strongest nearby-track bucket is not primarily an angle or length-ratio failure. It is mostly a center/position offset around the match threshold. The centerline-only bucket is a different mechanism and likely needs new instance generation, not repair of the current nearest track.

Latest lane lateral-duplicate budget audit:

- branch/worktree: `exp/lane-family-f1/lane-lateral-duplicate-budget-audit`.
- code commit: `50038b9`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_lateral_duplicate_budget_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: no decoder change; compute the FP budget a lateral-duplicate style recovery would have if it recovered GT-labeled nearby-track FN buckets.
- baseline lane TP/FP/FN/F1: `4518 / 2206 / 4959 / 0.5577`.
- `unmatched<=80 and center>=0.50`: `563` recoverable FN; no-added-FP upper-bound F1 `0.6062`, but only `172` added FP can be tolerated.
- `unmatched<=120 and center>=0.50`: `836` recoverable FN; no-added-FP upper-bound F1 `0.6285`, with `809` added FP tolerance.
- `unmatched<=120 any center`: `1698` recoverable FN; no-added-FP upper-bound F1 `0.6946`, with `2821` added FP tolerance. If all current FP were duplicated and this whole bucket were recovered, the oracle-budget F1 is still `0.6184`.
- 판단: this is not production success, but it says a one-axis lateral-duplicate smoke is not mathematically dead if it targets a large nearby-track bucket. The next implementation must prove TP recovery and added-FP cost together; do not turn this into an offset/radius sweep.

Latest lane FP-repair oracle audit:

- branch/worktree: `exp/lane-family-f1/lane-fp-repair-oracle-audit`.
- code commit: `0a9aa01`.
- artifacts:
  - `runs/pv26_exhaustive_od_lane_train/lane_fp_repair_oracle_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
  - `runs/pv26_exhaustive_od_lane_train/lane_fp_repair_oracle_audit_20260513/analysis_exports/fp_repair_broader_val512_epoch2/summary.json`.
- changed axis: no decoder change; re-generate broader-val512 FN rows, then count each nearest unmatched prediction once as an oracle repair of an existing FP into a TP (`TP+1`, `FP-1`, `FN-1`).
- baseline lane TP/FP/FN/F1: `4518 / 2206 / 4959 / 0.5577`.
- `unmatched<=120 any center`: `1698` FN rows map to `1272` unique repairable unmatched predictions; oracle-repair lane F1 `0.7148`, TP/FP/FN `5790 / 934 / 3687`.
- `unmatched<=80 any center`: `1007` FN rows map to `854` unique repairable unmatched predictions; oracle-repair lane F1 `0.6632`, TP/FP/FN `5372 / 1352 / 4105`.
- `unmatched<=80 and center>=0.50`: `563` FN rows map to `533` unique repairable unmatched predictions; oracle-repair lane F1 `0.6235`, TP/FP/FN `5051 / 1673 / 4426`.
- 판단: existing unmatched predictions contain enough GT-labeled repair headroom even after deduplicating by prediction id, but this is oracle planning evidence only. It supports a no-GT FP-to-TP repair contract, not another duplicate append, translation radius, or post-hoc threshold sweep.

Latest lane repairable-unmatched feature audit:

- branch/worktree: `exp/lane-family-f1/lane-repairable-unmatched-feature-audit`.
- code commit: `b31f364`.
- artifacts:
  - `runs/pv26_exhaustive_od_lane_train/lane_repairable_unmatched_feature_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
  - `runs/pv26_exhaustive_od_lane_train/lane_repairable_unmatched_feature_audit_20260513/analysis_exports/feature_auc_broader_val512_epoch2/summary.json`.
- changed axis: extend the read-only FN recovery probe to export one row per unmatched prediction with no-GT track/map features and GT-derived repair labels, then score single-feature separability by AUC/AP.
- unmatched prediction rows: `2206`.
- repairable labels: `526` rows for `repairable_le80_center050`; `1393` rows for `repairable_le120_any_center`.
- tight label best single feature: `pred_polyline_length` AUC `0.7205`, AP `0.4278`; `pred_center_point_mean` AUC `0.7106`, AP `0.4205`.
- broad label best single feature: `pred_polyline_length` AUC `0.6614`, AP `0.7368`; `pred_center_point_mean` AUC `0.6295`, AP `0.7374`.
- feature summary: repairable `<=80/center>=0.50` rows are longer and stronger on centerline (`length q50 339.26`, `center_mean q50 0.9293`) than non-repairable `<=120` rows (`length q50 192.67`, `center_mean q50 0.8142`), but the single-feature precision at the positive-count cutoff is only `0.4563` for the tight label.
- 판단: no-GT features contain a real but moderate repairability signal. This supports a learned/contextual repair contract, but it is not strong enough to justify a single-feature threshold gate or another post-hoc selector replay as production.

Latest lane repairability model replay:

- branch/worktree: `exp/lane-family-f1/lane-repairability-model-replay`.
- code commits: `0b8af52`, `4960e0f`, `885948d`.
- artifacts:
  - `runs/pv26_exhaustive_od_lane_train/lane_repairability_model_replay_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
  - `runs/pv26_exhaustive_od_lane_train/lane_repairability_model_replay_20260513/analysis_exports/broader_val512_epoch2/repairability_model_replay.csv`.
  - `runs/pv26_exhaustive_od_lane_train/lane_repairability_model_replay_20260513/analysis_exports/broader_val512_epoch2/repairability_model_parameters.json`.
- changed axis: read-only 2-fold out-of-fold logistic ranker over no-GT unmatched-track/context features, followed by actual FP-to-TP oracle replay (`TP+1`, `FP-1`, `FN-1`) for selected repairable unmatched predictions. Commit `885948d` removes GT-derived `sample_unmatched_pred_count` from the ranker feature set.
- baseline lane TP/FP/FN/F1: `4518 / 2206 / 4959 / 0.5577`.
- tight `repairable_le80_center050`: OOF AUC/AP `0.7629 / 0.4742`; top-526 positive-budget replay selects `266` repairable rows and gives lane F1 `0.5906`; top-750 gives `330` repairs and F1 `0.5985`; top-1000 gives `396` repairs and F1 `0.6066`, but precision falls to `0.3960`.
- broad `repairable_le120_any_center`: OOF AUC/AP `0.6821 / 0.7491`; top-500 selects `405 / 500` repairable rows and gives lane F1 `0.6077`; top-1000 selects `770` repairs and gives F1 `0.6528`; positive-budget top-1393 selects `1016` repairs and gives F1 `0.6832`.
- 판단: multi-feature no-GT ranker has enough broad-bucket planning signal to justify a real model-side/decoder-side repair contract. This is still not production success because the replay assumes selected repairable unmatched predictions can actually be geometrically repaired into matched instances. Do not treat top-K replay as a threshold gate or success metric by itself.

Latest lane centerline-duplicate smoke:

- branch/worktree: `exp/lane-family-f1/lane-centerline-duplicate-smoke`.
- code commit: `61be845`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_centerline_duplicate_smoke_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: keep the original row-scan-tangent track and emit a centerline-translated duplicate only when the translation moves the track.
- smoke val4 result: lane F1 `0.5594`, TP/FP/FN `40 / 17 / 46`; stop-line/crosswalk `0.0000 / 0.5455`.
- comparison: row-scan-tangent smoke reference was `0.5899`, TP/FP/FN `41 / 12 / 45`; replacement centerline translation was `0.5674`, TP/FP/FN `40 / 15 / 46`.
- 판단: preserving the original track avoids replacing it, but the duplicate adds FP without recovering TP. Do not broaden this branch or repeat as a duplicate offset/radius sweep.

Latest lane ranked-translate repair smoke:

- branch/worktree: `exp/lane-family-f1/lane-ranked-translate-repair-smoke`.
- code commit: `621480f`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_ranked_translate_repair_smoke_20260513/analysis_exports/smoke_val4_epoch2/summary.json`.
- changed axis: use the exported no-GT broad repairability ranker parameters from the replay artifact, score decoded lane predictions, and translate only the val-size-scaled top-ranked budget toward the local centerline ridge.
- smoke val4 budget: `repair_topk=4`, `candidate_count=53`, `selected_count=4`, `selected_moved_count=0`.
- baseline and repaired lane/stop/cross F1 were identical: `0.5899 / 0.0000 / 0.5455`.
- lane TP/FP/FN stayed `41 / 12 / 45`; lane delta was `0` TP, `0` FP, `0` FN, `0.0` F1.
- 판단: the broad ranker replay does not transfer to this simple geometry repair. The top-ranked rows were already on strong predicted centerline support and the fixed translate operation did not move them, so do not broaden this branch or repeat as a ranker top-K / translation-radius sweep.

Latest lane ranked local-snap repair smoke:

- branch/worktree: `exp/lane-family-f1/lane-ranked-local-snap-repair-smoke`.
- code commit: `3c30b7a`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/lane_ranked_local_snap_repair_smoke_val4_epoch2/summary.json`.
- changed axis: keep the same broad no-GT repairability ranker and val-size-scaled top-500/2048 budget, but replace whole-track x translation with pointwise local 2D snapping to the predicted centerline ridge.
- smoke val4 movement: selected `4` rows, all moved; moved points `76`, mean move about `2.75-3.44` map px.
- smoke val4 metrics: baseline and local-snap repaired lane/stop/cross F1 were identical: `0.5899 / 0.0000 / 0.5455`; lane TP/FP/FN stayed `41 / 12 / 45`.
- 판단: local 2D snapping finally moves selected geometry, but the movement does not cross any matching boundary on the smoke slice. Do not broaden to val128 or repeat as ranked local snap/radius tuning without a new signal that first shows actual TP/FP/FN movement.

Latest lane ranked affine-snap repair smoke:

- branch/worktree: `exp/lane-family-f1/lane-ranked-affine-centerline-repair-smoke`.
- code commit: `b895653`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/lane_ranked_affine_snap_repair_smoke_val4_epoch2/summary.json`.
- changed axis: keep the same broad no-GT repairability ranker and repair budget, but fit one coherent 2D affine transform per selected lane from original points to local centerline-snap targets, then apply that transform to the whole lane instance.
- smoke val4 movement: selected `4` rows, all moved; moved points `77`; selected affine mean move about `1.16-1.37` map px and max move about `2.86-3.79` map px.
- smoke val4 metrics: baseline and affine-snap repaired lane/stop/cross F1 were identical: `0.5493 / 0.0000 / 0.5455`; lane TP/FP/FN stayed `39 / 17 / 47`.
- 판단: coherent affine movement is smaller than pointwise snapping and still does not cross any matching boundary on the smoke slice. Do not broaden to val128 or repeat centerline-peak geometry repair as affine/local-snap/radius variants without a new signal that first changes TP/FP/FN.

Latest lane ranked component-path repair smoke:

- branch/worktree: `exp/lane-family-f1/lane-ranked-component-path-repair-smoke`.
- code commit: `4d65328`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/lane_ranked_component_path_repair_smoke_val4_epoch2/summary.json`.
- changed axis: keep the same broad no-GT repairability ranker and repair budget, but replace selected lane points by projecting them onto the nearest predicted centerline connected-component rows.
- smoke val4 movement: selected `4` rows, all moved; moved points `35`.
- smoke val4 metrics: baseline and component-row repaired lane/stop/cross F1 were identical: `0.5899 / 0.0000 / 0.5455`; lane TP/FP/FN stayed `41 / 12 / 45`.
- 판단: component-path projection moves selected geometry but still does not change lane assignment. Do not broaden to val128 or repeat component-row/path projection without a new signal that first changes TP/FP/FN.

Latest lane ranked row-profile repair smoke:

- branch/worktree: `exp/lane-family-f1/lane-ranked-row-profile-repair-smoke`.
- changed axis: keep the same broad no-GT repairability ranker, val-size-scaled top-500/2048 budget, fixed `flip_centerline_avg` runtime lane path, and fixed stop-line/crosswalk settings, but replace connected-component projection with row-wise softargmax projection over the predicted centerline profile at radius `16`.
- artifact retention: no durable run artifact retained; this was a compact smoke and the summary metrics are recorded here after cleanup.
- smoke val4 budget: `repair_topk=4`, `candidate_count=53`, `selected_count=4`.
- smoke val4 movement: all selected rows moved; moved points `81`.
- baseline and repaired lane/stop/cross F1 were identical: `0.5899 / 0.0000 / 0.5455`.
- lane TP/FP/FN stayed `41 / 12 / 45`; lane delta was `0` TP, `0` FP, `0` FN, `0.0` F1.
- 판단: row-wise soft centerline-profile projection moves more selected points than component-row projection, but it still does not change assignment. Do not broaden to val128 or repeat this as a row-profile radius/window/softmax sweep without a new signal that first changes TP/FP/FN.

Latest lane point-repair oracle replay:

- branch/worktree: `exp/lane-family-f1/lane-point-repair-replay`.
- code commit: `d5a0c4e`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_point_repair_replay_20260514/analysis_exports/smoke_val4_oracle_le120/summary.json`.
- artifact val128: `runs/pv26_exhaustive_od_lane_train/lane_point_repair_replay_20260514/analysis_exports/val128_oracle_le120/summary.json`.
- changed axis: run the real validation pipeline, replace selected unmatched lane prediction points with their nearest currently missed GT lane points, then recompute lane/stop-line/crosswalk metrics through the evaluator.
- smoke val4 oracle result: lane F1 `0.5899 -> 0.6906`, TP/FP/FN `41 / 12 / 45 -> 48 / 5 / 38`.
- val128 oracle result: lane F1 `0.5854 -> 0.7332`, TP/FP/FN `1200 / 510 / 1190 -> 1503 / 207 / 887`; selected `322` of `510` unmatched predictions, with `18` duplicate targets and actual `+303` TP.
- 판단: this proves actual lane point movement can pass the lane F1 gate when the target geometry is known, and it validates the replay machinery. It is oracle-only and leaves stop-line at `0.4483`, so it is not production success or all-task 0.6 evidence. The next lane step must infer comparable target geometry from no-GT signals.

Latest lane point-repair regression premise:

- branch/worktree: `exp/lane-family-f1/lane-point-repair-regression-premise`.
- code commit: `199af80`.
- artifact val128: `runs/pv26_exhaustive_od_lane_train/lane_point_repair_regression_premise_20260514/analysis_exports/val128_selected_translation_l2_10/summary.json`.
- changed axis: use only prediction-side features from the oracle replay candidate CSV to predict a single pred-to-target translation vector with two-fold heldout ridge regression.
- heldout result on the `322` oracle-selected val128 candidates: close-to-GT count improved only `24 -> 44`, while distance q50/q90 worsened `65.31 / 106.31 -> 66.89 / 110.11`; improved/worsened rows were `159 / 163`.
- 판단: a simple no-GT feature-to-translation regressor is too weak and unstable to justify live decoder integration. Do not repeat this as an l2/feature/threshold tuning exercise without a new geometry signal.

Latest lane repair geometry export:

- branch/worktree: `exp/lane-family-f1/lane-repair-geometry-export`.
- code commit: `a522ef7`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_repair_geometry_export_20260513/analysis_exports/smoke_val4_epoch2/summary.json`.
- artifact val128: `runs/pv26_exhaustive_od_lane_train/lane_repair_geometry_export_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: export compact `pred_points_json`, `nearest_fn_gt_points_json`, and FN-side GT/nearest-pred point JSON columns from the existing lane FN/unmatched repair audit.
- smoke val4 check: `lane_unmatched_prediction_repair_rows.csv` now has point JSON for `12` unmatched predictions; `7` are `repairable_le120_any_center`.
- val128 check: baseline lane TP/FP/FN/F1 is `1200 / 510 / 1190 / 0.5854`; `510` unmatched predictions were exported, with `128` tight repairable rows and `322` broad repairable rows. CSV inspection confirmed `pred_points_json`, `nearest_fn_gt_points_json`, `gt_points_json`, and nearest-pred point JSON columns are populated.
- 판단: this is export plumbing only, not a decoder or F1 improvement. It exists so the next lane branch can replay actual moved lane geometry and recompute TP/FP/FN instead of relying on aggregate distance columns.

Latest lane repair replay tooling status:

- branch/worktree: `exp/lane-family-f1/restore-lane-repair-replay`.
- restored active tools: `tools/probe_pv26_lane_instance_evidence.py`, `tools/probe_pv26_lane_fn_recovery_audit.py`, and `tools/replay_pv26_lane_point_repair.py`.
- restored focused tests: `test/test_lane_instance_evidence_probe.py`, `test/test_lane_fn_recovery_audit.py`, and `test/test_lane_point_repair_replay.py`.
- detached-worktree safety: lane FN/replay defaults are repo-relative again, and retained-run smoke execution passes explicit checkpoint, source-run, and dataset-root paths.
- one-batch CUDA smoke on the retained merged checkpoint completed with oracle-only point repair: baseline lane TP/FP/FN/F1 `10 / 6 / 11 / 0.5405`, repaired `16 / 0 / 5 / 0.8649`, selected `6` rows with `0` duplicate targets.
- current-best lane variant plumbing: the point-repair replay now also accepts the fixed `flip_centerline_avg_lane_cross_comp050` lane path. A one-batch oracle-only smoke with that variant produced the same baseline/repaired lane movement `10 / 6 / 11 / 0.5405 -> 16 / 0 / 5 / 0.8649`, selected `6` rows, and `0` duplicate targets.
- Current `develop` also regenerated the exact-val128 lane FN/repair export from the retained merged checkpoint: baseline lane TP/FP/FN/F1 `1200 / 510 / 1190 / 0.5854`, unmatched predictions `510`, tight repairable rows `128`, broad repairable rows `322`.
- 판단: this restores the lane repair replay machinery after artifact/worktree cleanup and aligns it with the fixed current-best lane runtime variant. The smoke is oracle-only and one batch, so it is not production success or all-task `0.60` evidence.

Latest lane repairability ranker tooling status:

- branch/worktree: `exp/lane-family-f1/restore-lane-repairability-ranker`.
- restored active tool: `tools/analyze_pv26_lane_repairability_model_replay.py`.
- restored focused test: `test/test_lane_repairability_model_replay.py`.
- archived broader-val512 unmatched-row replay regenerated the known ranker premise: broad label OOF AUC/AP `0.6821 / 0.7491`, broad top500 oracle-repair lane F1 `0.6077`, and parameter export.
- 판단: this restores the no-GT repairability selection scorer surface only. It still assumes selected repairable unmatched predictions can be repaired, so it is not a geometry repair and not production F1 success.

Latest lane residual-component candidates:

- branch/worktree: `exp/lane-family-f1/lane-residual-component-candidates`.
- code commit: `93addb6`.
- artifact smoke: `runs/pv26_exhaustive_od_lane_train/lane_residual_component_candidates_20260513/analysis_exports/smoke_val4_epoch2/summary.json`.
- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane_residual_component_candidates_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: keep the checkpoint and flip-centerline lane baseline fixed, then append only lane candidates generated from predicted centerline/support residue not already covered by decoded lane tracks.
- smoke val4: lane F1 `0.5714`, TP/FP/FN `40 / 14 / 46`, versus baseline `0.5588`, `38 / 12 / 48`.
- exact val128: lane F1 `0.5669`, TP/FP/FN `1195 / 631 / 1195`, versus baseline `0.5739`, `1175 / 530 / 1215`.
- 판단: uncovered centerline residue can recover TP, but the FP cost is much larger at val128. Do not broaden to val512 or repeat as residual threshold/component-size/coverage-width/min-length/per-sample-cap tuning without a new no-GT FP-control signal.

Latest lane residual proximity-gate audit:

- branch/worktree: `exp/lane-family-f1/lane-residual-shape-fp-control-audit`.
- code commit: `914f59a`.
- artifacts: `runs/pv26_exhaustive_od_lane_train/lane_residual_shape_fp_control_audit_20260513/analysis_exports/{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- changed axis: keep the residual-candidate generator fixed, add candidate-level feature labeling, and replay one fixed support/shape/proximity gate that keeps residual candidates only when they are lane-like and within `200px` mean-point distance of an existing baseline lane.
- smoke val4: gated residual lane F1 `0.5755`, TP/FP/FN `40 / 13 / 46`, versus baseline `0.5588`, `38 / 12 / 48`; candidate gate kept `3 / 4` residual candidates, preserving both residual TPs and dropping one FP.
- exact val128: gated residual lane F1 `0.5706`, TP/FP/FN `1190 / 591 / 1200`, versus baseline `0.5739`, `1175 / 530 / 1215`; candidate gate kept `76 / 121` residual candidates, including `15` matched residual TPs but also `61` FPs.
- 판단: baseline-proximity/support gating is better than raw residual append but still below baseline at val128. Do not broaden to val512 or repeat residual append as proximity/length/support/component/per-sample threshold sweeps without a stronger new FP-control signal.

Latest lane semantic vote mode audit:

- branch/worktree: `exp/lane-family-f1/lane-semantic-vote-mode-audit`.
- code commit: `4954b98`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_semantic_vote_mode_audit_20260513/analysis_exports/smoke_val4_epoch2/summary.json`.
- artifact exact val128: `runs/pv26_exhaustive_od_lane_train/lane_semantic_vote_mode_audit_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: keep the current `flip_centerline_avg` lane replay fixed and vary only the lane vectorizer semantic vote mode across `component`, `centerline`, `centerline_excess`, and `component_core`.
- smoke val4 result: all four modes were identical, lane F1 `0.5899`, TP/FP/FN `41 / 12 / 45`.
- exact val128 result: all four modes had identical lane TP/FP/FN/F1 `1200 / 510 / 1190 / 0.5854`; stop-line/crosswalk stayed `0.4364 / 0.5988`. Only lane score / `phase_objective` jittered slightly (`0.62957..0.62975`).
- 판단: lane class/type semantic voting is not the active F1 bottleneck for the current flip-centerline composite. Do not broaden this to val512 or repeat as semantic-vote weighting/class-type-vote sweeps unless a new diagnostic first shows class/type misvote is causing metric FN/FP.

Latest lane dual-checkpoint centerline ensemble smoke:

- branch/worktree: `exp/lane-family-f1/lane-dual-checkpoint-centerline-ensemble`.
- code commit: `4834dcd`.
- artifact smoke: `runs/pv26_exhaustive_od_lane_train/lane_dual_checkpoint_centerline_ensemble_20260513/analysis_exports/smoke_val4_epoch2/summary.json`.
- changed axis: keep the current lane-head transplant as primary and the original tangent-link checkpoint as secondary, then average only lane centerline logits; stop-line/crosswalk outputs remain from the primary normal pass.
- smoke val4: `flip_centerline_avg` reference lane F1 `0.5899`, TP/FP/FN `41 / 12 / 45`; `dual_checkpoint_flip_centerline_avg` lane F1 `0.5672`, `38 / 10 / 48`; `dual_checkpoint_centerline_avg` lane F1 `0.5263`, `35 / 12 / 51`.
- 판단: the original tangent-link checkpoint does not add complementary lane centerline evidence under fixed averaging; it removes too many TP. Do not broaden to val128 or repeat as checkpoint/weight/threshold averaging sweeps without a new diagnostic showing complementary TP recovery.

Latest lane row-scan tangent global-assignment smoke:

- branch/worktree: `exp/lane-family-f1/lane-row-scan-tangent-global-smoke`.
- code commit: `7d761ac`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/lane_row_scan_tangent_global_smoke_val4_epoch2/metrics.csv`.
- changed axis: keep the existing row-scan-tangent cost/thresholds fixed, but replace greedy per-cluster assignment with per-row Hungarian track-to-cluster assignment.
- verification: `py_compile`, `pytest -q test/test_lane_segfirst_vectorizer.py`, then val4 checkpoint replay.
- smoke val4 result: row-scan-tangent reference lane F1 `0.5899`, TP/FP/FN `41 / 12 / 45`; global-assignment lane F1 `0.5152`, TP/FP/FN `34 / 12 / 52`; stop-line/crosswalk F1 `0.0000 / 0.4000`.
- 판단: global assignment fixes a synthetic greedy-conflict unit case, but in real val4 it loses seven lane TP without reducing FP. Do not broaden to val128/val512 or repeat as a Hungarian/global row-assignment variant unless a new recall-preserving signal first shows TP recovery.

Latest lane scale-centerline TTA smoke:

- branch/worktree: `exp/lane-family-f1/lane-scale-centerline-tta-smoke`.
- code commit: `37804a4`.
- artifact downscale smoke: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_scale_centerline_tta_smoke_val4_epoch2/summary.json`.
- artifact upscale smoke: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_scale1125_centerline_tta_smoke_val4_epoch2/summary.json`.
- changed axis: keep the current lane-head transplant, stop-line overrides, and hull crosswalk fixed; run one scale-resized input pass and average only the resized lane centerline logits back into the normal-pass dense map.
- verification: `py_compile`, `pytest -q test/test_lane_flip_tta_probe.py`, then two val4 smokes for scale factors `0.875` and `1.125`.
- downscale `0.875`: `flip_centerline_avg` lane F1 `0.5899`, TP/FP/FN `41 / 12 / 45`; `flip_scale_centerline_avg` lane F1 `0.5606`, `37 / 9 / 49`; `scale_centerline_avg` lane F1 `0.5414`, `36 / 11 / 50`.
- upscale `1.125`: `flip_centerline_avg` lane F1 `0.5899`, TP/FP/FN `41 / 12 / 45`; `flip_scale_centerline_avg` lane F1 `0.5778`, `39 / 10 / 47`; `scale_centerline_avg` lane F1 `0.5373`, `36 / 12 / 50`.
- 판단: single-scale centerline TTA raises the phase objective in one smoke row by reducing FP, but it loses lane TP versus the existing flip baseline in both scale directions. Do not broaden to val128/val512 or repeat as a scale-factor/interpolation sweep without a new recall-preserving premise.

Latest lane soft-ridge recovery readout smoke:

- branch/worktree: `exp/lane-family-f1/lane-fn-nearby-fp-recovery-audit`.
- code commit: `ee85fd8`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_soft_ridge_recovery_audit_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: add opt-in `row_scan_tangent_soft_ridge`, which selects per-row centerline probability ridge peaks before tangent linking, then replay it with the current flip-centerline broader composite settings at `lane_obj_threshold=0.30`.
- smoke baseline row-scan-tangent lane F1: `0.5899`, TP/FP/FN `41 / 12 / 45`.
- soft-ridge smoke lane F1: `0.5429`, TP/FP/FN `38 / 16 / 48`.
- 판단: soft-ridge peak picking loses TP and adds FP even on val4 smoke. Do not broaden this readout to val512 or repeat it as a lane threshold sweep without a new non-GT signal that explains how ridge candidates avoid this regression.

Latest lane centerline-snap recovery readout smoke:

- branch/worktree: `exp/lane-family-f1/lane-unmatched-track-pair-audit`.
- code commit: `43735cf`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_centerline_snap_recovery_audit_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: add opt-in `row_scan_tangent_centerline_snap`, which preserves row-scan-tangent instance topology and only snaps existing track x coordinates to local same-row centerline peaks.
- prior row-scan-tangent smoke lane F1: `0.5899`, TP/FP/FN `41 / 12 / 45`.
- centerline-snap smoke lane F1: `0.5674`, TP/FP/FN `40 / 15 / 46`.
- 판단: centerline snapping is less damaging than global soft-ridge peak generation but still loses TP and adds FP relative to the same smoke reference. Do not broaden this readout to val512 or repeat it as a snap-radius sweep without a new FP-control signal.

Latest lane track-level translation readout smoke:

- branch/worktree: `exp/lane-family-f1/lane-track-translation-readout`.
- code commit: `01b3ac1`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_track_translation_readout_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: add opt-in `row_scan_tangent_centerline_translate`, which preserves row-scan-tangent instance topology and y coordinates, then chooses one integer x offset for the whole decoded track by mean centerline score.
- prior row-scan-tangent smoke lane F1: `0.5899`, TP/FP/FN `41 / 12 / 45`.
- track-translation smoke lane F1: `0.5674`, TP/FP/FN `40 / 15 / 46`.
- 판단: track-level uniform translation does not rescue the pair-geometry center-offset bucket. It matches the centerline-snap regression pattern, so do not broaden this readout to val512 or repeat it as a translation-radius/offset sweep without a materially new non-GT FP-control signal.

Latest lane raw-vectorizer drop audit:

- branch/worktree: `exp/lane-family-f1/lane-raw-vectorizer-drop-audit`.
- code commit: `0545315`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_raw_vectorizer_drop_audit_20260513/analysis_exports/broader_val512_epoch2/summary.json`.
- changed axis: no production decoder change; for every FN lane, compare final predictions to raw row-scan-tangent vectorizer candidates before bbox-area/aspect geometry filters.
- broader val512 audit baseline in this replay: lane/stop/cross F1 `0.5534 / 0.4235 / 0.6187`, lane TP/FP/FN `4564 / 2453 / 4913`.
- all FN: raw vectorizer already has a `<=40px` candidate for `854 / 4913`; `497` of those fail bbox-area and `37` fail aspect.
- `center>=0.50 without unmatched<=120px`: count `512`, raw vectorizer `<=40px` count `207`, area-filter drops `119`, aspect drops `14`, pass-filter/assignment cases `82`.
- 판단: center-only FN is not pure candidate-generation absence. This justified one guarded area-filter rescue probe, but not a blind bbox-area/aspect sweep because many misses are raw `80/120px` or pass-filter assignment cases.

Latest lane guarded area-rescue readout:

- branch/worktree: `exp/lane-family-f1/lane-guarded-area-rescue-readout`.
- code commit: `a1efacd`.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_guarded_area_rescue_readout_20260513/analysis_exports/val128_epoch2_rescue1_area1024/summary.json`.
- changed axis: add an opt-in decoder override that can rescue at most one lane candidate per sample when it fails only bbox-area filtering, has bbox area at least `1024px`, and still passes aspect.
- raw-audit replay reference on val128: lane F1 `0.5797`, TP/FP/FN `1204 / 560 / 1186`.
- guarded area-rescue val128 result: lane F1 `0.5716`, TP/FP/FN `1247 / 726 / 1143`; stop-line/crosswalk retained at `0.4364 / 0.5988`.
- 판단: area rescue recovers `43` lane TP but adds `166` FP, so it lowers F1 and should not be broadened to val512. Do not repeat this as a max-per-sample/min-area/bbox-filter sweep without a new FP-control signal.

Latest lane area-rescue center-score gate:

- branch/worktree: `exp/lane-family-f1/lane-area-rescue-center-score-gate`.
- code commit: `4482443`.
- artifact exact val128: `runs/pv26_exhaustive_od_lane_train/lane_area_rescue_center_score_gate_20260513/analysis_exports/val128_epoch2_center075/summary.json`.
- artifact broader val512: `runs/pv26_exhaustive_od_lane_train/lane_area_rescue_center_score_gate_20260513/analysis_exports/val512_epoch2_center075/summary.json`.
- changed axis: keep area rescue opt-in, then require rescued lane candidates to have vectorizer track-level `lane_centerline_track_mean >= 0.75`.
- exact val128 result: lane F1 `0.5846`, TP/FP/FN `1239 / 610 / 1151`; stop-line/crosswalk `0.4483 / 0.5988`.
- broader val512 result: lane F1 `0.5548`, TP/FP/FN `4644 / 2621 / 4833`; stop-line/crosswalk `0.4083 / 0.6187`.
- 판단: centerline mean gating fixes part of the val128 FP problem versus ungated area rescue, but broader val512 still stays below the current broader lane best `0.5628`. This is weak partial/negative evidence, not a default or a threshold-sweep path.

Latest lane area-rescue center-q10 gate:

- branch/worktree: `exp/lane-family-f1/lane-area-rescue-center-q10-gate`.
- code commit: `2d16971`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_area_rescue_center_q10_gate_20260513/analysis_exports/smoke_val4_epoch2_center075_q10060/summary.json`.
- changed axis: keep the previous area rescue and `lane_centerline_track_mean >= 0.75`, then also require `lane_centerline_track_q10 >= 0.60` as a stricter track-wide FP-control signal.
- smoke val4 result: lane F1 `0.5816`, TP/FP/FN `41 / 14 / 45`; stop-line/crosswalk `0.0000 / 0.5455`.
- comparison to center-mean smoke: lane F1 regressed from `0.5972` to `0.5816`, moving TP/FP/FN from `43 / 15 / 43` to `41 / 14 / 45`.
- 판단: q10 gating removes only one FP while losing two TP on the first smoke gate. Do not broaden this branch to val128/val512 or repeat area rescue as a q10/quantile threshold sweep without a materially new recall-preserving signal.

Latest lane row-scan hysteresis readout:

- branch/worktree: `exp/lane-family-f1/lane-row-scan-hysteresis-readout`.
- code commit: `ac4d498`.
- artifact smoke val4: `runs/pv26_exhaustive_od_lane_train/lane_row_scan_hysteresis_readout_20260513/analysis_exports/smoke_val4_epoch2_t030/summary.json`.
- changed axis: generate row-scan-tangent candidates from a lower centerline threshold, but keep only low-threshold connected components that contain a high-confidence seed.
- smoke val4 result: lane F1 `0.5652`, TP/FP/FN `39 / 13 / 47`; stop-line/crosswalk `0.0000 / 0.5455`.
- 판단: hysteresis is less damaging than soft-ridge peak generation, but it still loses two TP versus the same row-scan-tangent smoke reference and stays below the centerline-snap/translation smoke. Do not broaden or repeat as a low/high threshold sweep without a new instance-level recall-preserving signal.

Latest stop-line recovery-budget audit:

- branch/worktree: `exp/lane-family-f1/stopline-recovery-budget-audit`.
- code commit: `ff70095`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_recovery_budget_audit_20260513/analysis_exports/projection_competition_val512/summary.json`.
- reference: projection-competition readout `proj_comp_length_s090_top2_second_frag5`, stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- candidate-manifest buckets: positive top-oracle `113`, positive misrank `29`, positive no-oracle `62`, GT-negative candidate-bearing `165`.
- target math: at current FP, stop-line needs `+30` recovered TP to reach F1 `0.60`; FP-only recovery would need removing `68 / 91` FP without losing TP.
- selector-only ceiling: recovering all `29` positive-misrank samples gives stop-line F1 `0.5996`, still just below target; misrank plus one no-oracle recovery gives `0.6023`.
- no-oracle upper bound: recovering all `62` positive-no-oracle samples gives F1 `0.6836` and could tolerate up to `76` added FP while staying at `>=0.60`.
- 판단: another selector/logistic/photometric/projection-threshold sweep is not the next useful stop-line axis. The next stop-line branch must add a no-GT candidate-generation or midpoint-recovery signal that reaches currently no-oracle positives while controlling added FP.

Latest stop-line no-oracle fragment-extension budget:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `fd6fab0`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_extension_budget_val512_epoch2/summary.json`.
- changed axis: replace only positive-no-oracle samples in the projection-competition replay with simple min-length extensions of current candidate fragments; compare top, longest-high-score, and nearest-GT-oracle candidate selectors.
- baseline projection competition remains best: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- best extension variants reach only `0.4713`, TP/FP/FN `119 / 115 / 152`.
- 판단: positive no-oracle is not solved by simply extending short fragments around their current midpoint. Do not implement or sweep a min-length fragment-extension postprocess without a new no-GT centering/candidate-generation signal.

Latest stop-line no-oracle proposal-recall bucket audit:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `07218b9`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_proposal_recall_bucket_val512_epoch2/summary.json`.
- changed axis: no new model run; join existing val512 per-GT proposal recall with candidate-manifest failure buckets.
- positive-no-oracle GT count in the joined max-source rows: `69`.
- max-source local signal: `max_r8 >= 0.6` is `44 / 69`, `max_r4 >= 0.6` is `36 / 69`.
- top-k proximity: `top20_hit_r8` is `61 / 69`, while `top1_hit_r8` is only `12 / 69` and `top3_hit_r8` is `27 / 69`.
- 판단: many no-oracle positives still have dense `max(center, selector)` signal near the GT center. The failure is not primarily dense-map absence; it is current top candidate selection / midpoint centering / geometry decode. The next stop-line branch should generate candidates from local max-source neighborhoods or learn a richer emit/select contract, not stretch selected fragments.

Latest stop-line no-oracle anchor-shift audit:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `0670701`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_anchor_shift_val512_epoch2/summary.json`.
- changed axis: replace positive-no-oracle samples by shifting current candidate segments from decoded center toward the proposal-cell anchor; no model run and no production change.
- baseline projection competition remains best: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- all anchor-shift variants collapse to F1 `0.4475`, TP/FP/FN `113 / 121 / 158`.
- 판단: the no-oracle gap is not solved by a simple proposal-anchor vs decoded-anchor correction. Next stop-line work needs a richer local-neighborhood candidate geometry signal, not anchor shifting.

Latest stop-line no-oracle local proposal geometry audit:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `eee5085`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_local_proposal_geometry_val512_epoch2/summary.json`.
- changed axis: join positive-no-oracle GT rows to the nearest exported max-source proposal candidate, testing whether local proposal cells near GT already decode usable geometry.
- positive-no-oracle GT rows: `69`; `top20_hit_r8` is `61`, and nearest exported candidate is within r8 for `51`.
- local candidate quality: `nearest_gt_distance <= 40` is `0 / 51`; q50 local candidate nearest distance `95.36px`, midpoint distance `68.59px`, length ratio `0.5437`, angle error `2.59deg`.
- 판단: local score/proposal cells are often near GT, but the exported candidate geometry still does not become matched stop-line segments. The next stop-line edit must create a new local geometry readout from the score island or change the emit/select/readout contract; just selecting local top20 cells is not enough.

Latest stop-line no-oracle local recenter budget:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `063c2fa`, hardened by `0281346` to prefer per-sample affine for proposal-cell-to-raw recentering.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/no_oracle_local_recenter_budget_val512_epoch2/summary.json`.
- changed axis: replay GT-joined local proposal candidates with progressively stronger oracles: raw local row, proposal-anchor recenter, min-length recenter, GT-length oracle, and GT-midpoint+GT-length oracle.
- method check: selected local samples with per-sample affine `47 / 47`; sample-affine fallback count `0`.
- baseline projection competition: stop-line F1 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- local raw/anchor/minlen/GT-length all stay below baseline: best non-midpoint oracle is `0.4939`, TP/FP/FN `122 / 101 / 149`.
- GT-midpoint+GT-length oracle opens the budget: F1 `0.6599`, TP/FP/FN `163 / 60 / 108`.
- 판단: local angle evidence is usable only if the center/midpoint is correct. Length or anchor correction is not enough; the next production branch must infer the missing stop-line midpoint/center without GT.

Latest stop-line score-island midpoint readout:

- branch/worktree: `exp/lane-family-f1/stopline-mask-midpoint-recovery-readout`.
- code commit: `b5f29f5`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_mask_midpoint_recovery_readout_20260513/analysis_exports/score_island_midpoint_readout_val128_epoch2/summary.json`.
- changed axis: add opt-in score-island weighted center mode to the existing predicted proposal + angle-mask extent readout; no GT centers, no training, stop-line predictions only.
- exact val128 baseline stop-line F1: `0.4483`, TP/FP/FN `26 / 30 / 34`.
- existing selector-center reference remains best in this replay: `pred_selector_top1_s060_mask050_band4` stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`.
- best island variant reaches only `0.4354`, TP/FP/FN `32 / 55 / 28`; wider island variants regress to `0.4218`.
- 판단: score-island centroiding changes center placement slightly but increases FP and does not recover the midpoint budget. Do not broaden to val512 or turn this into a radius/relative-threshold sweep without a new non-GT FP-control signal.

Latest stop-line score-island linefit readout:

- branch/worktree: `exp/lane-family-f1/stopline-score-island-linefit-readout`.
- code commit: `e89db20`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_score_island_linefit_readout_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: fit both center and axis from the local max(center, selector) score island, then reuse the existing mask-extent line generation; no GT center/angle, no training.
- exact val128 baseline stop-line F1: `0.4483`, TP/FP/FN `26 / 30 / 34`.
- existing selector-center reference remains best in this replay: `pred_selector_top1_s060_mask050_band4` stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`.
- score-island linefit reaches only `0.4110`, TP/FP/FN `30 / 56 / 30`.
- 판단: score-island linefit recovers no extra TP over selector-center and adds too many FP. Do not broaden to val512 or repeat as an island radius/relative-threshold/linefit sweep without a materially new FP-control signal.

Latest stop-line detector-context FP audit:

- branch/worktree: `exp/lane-family-f1/stopline-detector-context-fp-audit`.
- code commit: `bf24a93`.
- artifacts:
  - `runs/pv26_exhaustive_od_lane_train/stopline_detector_context_fp_audit_20260513/analysis_exports/smoke_val4_epoch2/summary.json`.
  - `runs/pv26_exhaustive_od_lane_train/stopline_detector_context_fp_audit_20260513/analysis_exports/val128_epoch2/summary.json`.
- changed axis: keep the current candidate pool fixed, then add predicted `traffic_light` / `sign` proximity features and fixed signal-context candidate variants; no GT context, no training, no production decoder change.
- exact val128 baseline stop-line F1: `0.4483`, TP/FP/FN `26 / 30 / 34`.
- score-threshold reference remains stronger: `max_top10_score_s080` stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`.
- detector signal-context variants fail: `max_top10_signal_context_c1` reaches only `0.1538`, TP/FP/FN `6 / 12 / 54`; `max_top20_signal_context_c1` reaches only `0.1190`, TP/FP/FN `5 / 19 / 55`; `gap4_max_top50_signal_context_c1` reaches only `0.1395`, TP/FP/FN `6 / 20 / 54`.
- candidate feature check: top10/gap10 signal-near gating preserves only `25 / 156` oracle-positive rows (`0.160` positive recall), top20/gap10 preserves `26 / 168` (`0.155`), and gap4/top50 preserves `121 / 723` (`0.167`).
- 판단: predicted traffic-light/sign proximity does suppress emissions, but it is not recall-preserving; it removes most valid stop-line candidates and falls far below baseline and selector-center references. Do not repeat this as detector signal proximity threshold/radius/score sweeps without a new TP-preserving signal.

Latest stop-line geometry-regression premise:

- branch/worktree: `exp/lane-family-f1/stopline-geometry-regression-premise`.
- code commit: `b88ae0e`.
- artifact: `runs/pv26_exhaustive_od_lane_train/stopline_geometry_regression_premise_20260513/analysis_exports/val128_from_detector_context_top1/summary.json`.
- changed axis: keep the fixed `max_top10_score_s080`, `max_components=1` selected candidates from the detector-context candidate CSV, then train ridge regressors on GT-derived along-axis midpoint and length corrections using no-GT candidate features only.
- scope: candidate-bearing rows from the input CSV, not a full validation replay.
- train split: baseline stop-line F1 `0.4314`, geometry-regressed F1 `0.6667`, TP/FP/FN `17 / 7 / 10`.
- heldout split: baseline stop-line F1 `0.5938`, geometry-regressed F1 `0.0938`, TP/FP/FN `3 / 31 / 27`.
- heldout delta: `-0.5000` F1, `-16` TP, `+16` FP, `+16` FN.
- 판단: no-GT feature regression overfits the train half and destroys heldout geometry. Do not treat train-split correction as a model-side readout premise or repeat this as ridge-alpha/feature-subset/threshold tuning without a materially different geometry signal.

Current best exact lane-retention probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_segment_mil_lane_head_only_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_230022/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_segment_mil_lane_head_only`
- objective: `0.6193428422373919`
- lane / stop-line / crosswalk F1: `0.5660 / 0.4483 / 0.5926`
- lane TP/FP/FN: `1162 / 554 / 1228`
- stop TP/FP/FN: `26 / 30 / 34`
- cross TP/FP/FN: `48 / 33 / 33`
- 판단: exact val128에서는 tangent-link exact reference `0.6187165763`을 아주 조금 넘는 새 best지만, lane gain은 `+0.0027` 수준이고 stop-line은 여전히 `0.4483`이라 broader-val512 확장 조건으로 보지 않는다.

Latest lane instance-validator probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_instance_validator_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_193428/phase_4/history/epochs.jsonl`
- validator-logit audit: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_instance_validator_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_193428/analysis_exports/validator_logits_on_tangent_candidates_val128_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_instance_validator`
- changed axis: add an opt-in `lane_seg_instance_validator_logits` head, supervise predicted centerline components as valid/invalid candidates, then decode with `row_scan_tangent_instance_validator`.
- best exact val128 epoch2 objective: `0.6142642145`
- lane / stop-line / crosswalk F1: `0.5523 / 0.4522 / 0.5854`
- lane TP/FP/FN: `1099 / 491 / 1291`
- stop TP/FP/FN: `26 / 29 / 34`
- validator-logit audit on tangent-link candidates: heldout baseline lane F1 `0.5576`; best validator-only row threshold `0.5577`; logistic all-feature threshold `0.5649`; oracle TP-only selector `0.6344`.
- 판단: stop-line is slightly above tangent-link exact `0.4483`, but lane regresses below tangent-link `0.5633`, row-distribution `0.5659`, and segment-MIL lane-head-only `0.5660`; objective is also below all current exact references. The first validator contract is wired and runnable, and the oracle audit says lane still has FP-selector headroom, but the learned validator logits do not provide a recall-preserving production gate. This `weight=0.35`, threshold `0.45` axis is closed.

Latest lane upper-trunk capacity probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_upper_trunk_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_113104/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_upper_trunk`
- changed axis: keep the core target, `row_scan_tangent` vectorizer, stop-line/crosswalk settings, sampler, and train/val volume fixed, but reopen `lane_family_plus_upper_trunk` with trunk LR `2e-6` and head LR `1e-4`.
- best exact val128 epoch2 objective: `0.6169717875`
- lane / stop-line / crosswalk F1: `0.5597 / 0.4561 / 0.5854`
- lane TP/FP/FN: `1115 / 479 / 1275`
- stop TP/FP/FN: `26 / 28 / 34`
- cross TP/FP/FN: `48 / 35 / 33`
- 판단: stop-line is slightly higher than tangent-link exact `0.4483`, but lane falls below tangent-link `0.5633`, row-distribution `0.5659`, and segment-MIL lane-head-only `0.5660`; objective is also below the current exact references. Upper-trunk capacity/freeze-scope-only is not a broader-val512 expansion path.

Latest lane flip-consistency training probe:

- branch/worktree: `exp/lane-family-f1/lane-flip-consistency-row-scan-tangent`.
- code commits: `2dcd776` adds the opt-in train-time flip-consistency loss; `e19dbd1` adds the row-scan-tangent probe preset and single-axis guard test.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_flip_consistency_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_002056/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_flip_consistency`
- changed axis: keep `row_scan_tangent`, loss weights, freeze policy, stop-line/crosswalk settings, and sampler fixed, then add only `lane_flip_consistency_weight=0.25`.
- exact val128 epoch2 objective: `0.5991166581`
- lane / stop-line / crosswalk F1: `0.5542 / 0.3966 / 0.5548`
- task-best F1: lane `0.5542` at epoch2, stop-line `0.3966` at epoch2, crosswalk `0.6711` at epoch1.
- skipped steps: `0`.
- 판단: runtime is stable, but the exact metric regresses below tangent-link reference `0.6187`, `0.5633 / 0.4483 / 0.5854`; the epoch1 crosswalk spike is not enough because the best objective checkpoint misses all three task targets. Flip-consistency-regularizer-only is not a broader-val512 expansion path.

Latest lane endpoint-extension readout probe:

- branch/worktree: `exp/lane-family-f1/lane-endpoint-extension-readout`.
- code commit: `8f5c3a0` adds a read-only fixed-distance endpoint-extension probe on top of the row-scan-tangent readout.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane_endpoint_extend_val128_epoch2/summary.json`.
- changed axis: keep the checkpoint, row-scan-tangent decoder, stop-line/crosswalk predictions, sampler, and validation slice fixed, then extend decoded lane polyline endpoints by fixed raw-pixel distances before matching.
- exact val128 baseline objective and lane/stop/cross F1: `0.6187165763`, `0.5633 / 0.4483 / 0.5854`.
- best extension by objective: `top32`, objective `0.5998086929`, lane/stop/cross F1 `0.5503 / 0.4483 / 0.5854`.
- lane TP/FP/FN baseline: `1121 / 469 / 1269`; `top32`: `1095 / 495 / 1295`.
- 판단: fixed top/bottom endpoint extension hurts lane TP and FP at exact val128, so it is not a broader-val512 expansion path. Do not repeat as a distance sweep unless a new conditioning signal decides when extension is safe.

Latest lane positive-core flip-consistency training probe:

- branch/worktree: `exp/lane-family-f1/lane-positive-core-flip-consistency`.
- code commit: `572f8a4` adds `lane_flip_consistency_mask_mode` and a row-scan-tangent positive-core consistency preset.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_positive_flip_consistency_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_012145/phase_4/history/epochs.jsonl`.
- changed axis: keep row-scan-tangent, checkpoint, task weights, sampler, and consistency weight fixed, then restrict flip-consistency MSE to supervised `lane_seg_centerline_core` positive pixels.
- exact val128 epoch2 objective: `0.5981680601`
- lane / stop-line / crosswalk F1: `0.5513 / 0.3966 / 0.5548`
- task-best F1: lane `0.5513` epoch2, stop-line `0.3966` epoch2, crosswalk `0.6711` epoch1.
- skipped steps: `0`.
- 판단: runtime is stable, but positive-core-only consistency does not rescue the global flip-consistency regression. It is below tangent-link `0.6187`, `0.5633 / 0.4483 / 0.5854` and slightly below global flip-consistency objective `0.5991166581`, so do not broaden to val512.

Latest lane soft-skeleton topology-loss probe:

- branch/worktree: `exp/lane-family-f1/lane-centerline-soft-skeleton`.
- code commit: `e7a6ebf` adds opt-in soft-skeleton/clDice-style centerline topology loss and a row-scan-tangent probe preset.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_soft_skeleton_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_020600/phase_4/history/epochs.jsonl`.
- changed axis: keep row-scan-tangent, checkpoint, task weights, sampler, stop-line, and crosswalk fixed, then add only `lane_segfirst_soft_skeleton_weight=0.25` with `6` skeleton iterations.
- exact val128 epoch2 objective: `0.6152102115`
- lane / stop-line / crosswalk F1: `0.5609 / 0.4348 / 0.5854`
- lane TP/FP/FN: `1114 / 468 / 1276`
- task-best F1: lane `0.5609` epoch2, stop-line `0.4348` epoch2, crosswalk `0.6790` epoch1.
- skipped steps: `0`.
- 판단: runtime is stable, but topology-loss-only does not reach the tangent-link exact reference `0.6187165763`, `0.5633 / 0.4483 / 0.5854`; it also underperforms the earlier soft-shell auxiliary lane result. Do not broaden to val512 or repeat this as a weight/iteration sweep.

Latest lane instance-embedding row-link probe:

- branch/worktree: `exp/lane-family-f1/lane-embedding-row-scan-link`.
- code commit: `7b84900` adds an opt-in dense lane instance embedding head, `lane_seg_instance_id` target, embedding pull/push loss, and `row_scan_tangent_embedding` decoder.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_embedding_link_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_024937/phase_4/history/epochs.jsonl`.
- changed axis: keep row-scan-tangent, checkpoint, task weights, sampler, stop-line, and crosswalk fixed, then add only `lane_segfirst_loss_weights.instance_embedding=0.35` and decode with `row_scan_tangent_embedding`.
- exact val128 epoch2 objective: `0.6150797781`.
- lane / stop-line / crosswalk F1: `0.5594 / 0.4348 / 0.5854`.
- lane TP/FP/FN: `1114 / 479 / 1276`.
- task-best F1: lane `0.5594` epoch2, stop-line `0.4348` epoch2, crosswalk `0.6790` epoch1.
- skipped steps: `0`.
- 판단: runtime and checkpoint handoff are stable, but the embedding-link axis is below tangent-link exact `0.6187165763`, `0.5633 / 0.4483 / 0.5854`; it also fails to beat soft-skeleton or segment-MIL lane-head-only exact references. Do not broaden to val512 or repeat as an embedding distance/weight sweep.

Latest lane instance safety-gate replay:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_instance_safety_gate_replay_20260513/analysis_exports/broader_val512_flip_centerline_avg_safety_gate_epoch2/summary.json`.
- source features: `lane60_lane_flip_instance_evidence_20260512` broader flip-centerline instance evidence CSV.
- changed axis: keep the current flip-centerline row-scan candidates fixed, then preserve each sample's top-K candidates by `pred_index` while applying the existing logistic row gate to the rest.
- best heldout variant: `keep_topk=0`, threshold `0.1535836312`, lane F1 `0.5650`, TP/FP/FN `2142 / 818 / 2480`.
- full-split lane F1 for the same selected variant: `0.5705`, TP/FP/FN `4406 / 1564 / 5071`.
- `keep_topk=1..8` all reduce heldout/full lane F1 versus logistic-only because FP comes back faster than TP.
- 판단: top-K safety fallback does not solve the logistic gate's recall tradeoff and remains below the earlier full split-count logistic diagnostic `0.5738`, current lane oracle `0.6457`, and the `0.60` target. Do not convert this into a production safety gate.

Latest lane instance score-gate replay:

- branch/worktree: `exp/lane-family-f1/lane-instance-validator-score-rank`.
- code commit: `7975e77` adds an opt-in `row_scan_tangent_instance_score_gate` decoder that generates row-scan tangent candidates from the unmasked centerline, then filters completed lane candidates by their mean `instance_validator` score.
- replay artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_instance_score_gate_replay_20260513/analysis_exports/instance_score_gate_replay_val128_epoch2/summary.json`.
- source checkpoint: previous `core_centerline_refine_row_scan_tangent_instance_validator` best checkpoint, evaluated with `core_centerline_refine_row_scan_tangent_instance_score_gate`.
- exact val128 epoch2 objective: `0.6143914132`.
- lane / stop-line / crosswalk F1: `0.5529 / 0.4522 / 0.5854`.
- 판단: candidate-level score gating avoids the hard pixel-mask recall cut in code, but the metric is effectively flat against hard validator `0.6142642145`, `0.5523 / 0.4522 / 0.5854`, and still below tangent-link exact `0.6187165763`, `0.5633 / 0.4483 / 0.5854`. Do not launch full training or sweep validator score thresholds from this checkpoint.

Latest task-head merge with segment-MIL lane + rank-stop head:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_task_head_merge_segment_mil_rank_stop_source_cross_20260512/analysis_exports/exact_val128_segment_mil_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_task_head_merge_segment_mil_rank_stop_source_cross_20260512/analysis_exports/broader_val512_segment_mil_stop_pca_hull_epoch2/summary.json`
- composition: start from the segment-MIL lane-head-only checkpoint, replace lane weights from its `best_lane.pt`, replace stop-line weights from the proposal-rank `best_stop_line.pt`, and keep crosswalk from the source `best.pt`.
- exact val128 objective: `0.6248193729`
- exact lane / stop-line / crosswalk F1: `0.5660 / 0.4918 / 0.5926`
- exact TP/FP/FN lane: `1162 / 554 / 1228`
- exact TP/FP/FN stop-line: `30 / 32 / 30`
- exact TP/FP/FN crosswalk: `48 / 33 / 33`
- broader-val512 objective: `0.6129801219`
- broader lane / stop-line / crosswalk F1: `0.5480 / 0.3976 / 0.6185`
- broader TP/FP/FN lane: `4457 / 2333 / 5020`
- broader TP/FP/FN stop-line: `100 / 132 / 171`
- broader TP/FP/FN crosswalk: `231 / 121 / 164`
- 판단: exact subset에서는 stop-line head transplant가 좋아 보였지만, broader-val512에서는 stop-line FP가 늘어 current broader best objective `0.6230558331`와 stop-line `0.4235`보다 낮다. Task-head composition is exact partial-positive but broader negative; it is not an all-task success path.

Latest lane-head transplant onto original stop/cross base:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/exact_val128_row_scan_tangent_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/broader_val512_stop_pca_hull_epoch2/summary.json`
- composition: base original `best.pt`, lane head from segment-MIL lane-head-only `best_lane.pt`, stop-line/crosswalk heads from original `best.pt`.
- exact val128 objective: `0.6193289563`
- exact lane / stop-line / crosswalk F1: `0.5660 / 0.4483 / 0.5854`
- broader-val512 objective: `0.6176617972`
- broader lane / stop-line / crosswalk F1: `0.5480 / 0.4235 / 0.6187`
- broader TP/FP/FN stop-line: `101 / 105 / 170`
- 판단: this was the cleaner broader composite before flip-centerline TTA and remains the base checkpoint composition for the current runtime best. By itself, lane and stop-line still fail the `>=0.60` goal.

Latest stop-line component-topology validator replay:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_component_topology_rich_val512_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-component-topology-audit`
- changed axis: keep the checkpoint fixed, add predicted-mask component topology features to the stop-line candidate CSV, and replay the existing gap4/top50 rich-validator half-split task probe.
- candidate rows: `15629`; oracle-positive rows: `4683`; oracle-positive rate: `0.2996`.
- standard candidate reference: baseline stop-line F1 `0.4083`, score-threshold production `0.4371`, gap4/top50 oracle `0.6877`.
- held-out rich-logistic task replay: baseline `0.3877`, component-topology rich logistic `0.4126`, `selector_r4_max` threshold `0.4537`.
- 판단: component topology has some offline signal but does not beat the existing selector-only replay, PCA broader reference `0.4699`, or the 0.6 target. Treat it as read-only negative evidence, not a production decoder or model-side validator success.

Latest stop-line sample-tree gate audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_gap4_sample_tree_gate_audit_val512_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-sample-tree-gate-audit`
- changed axis: extend the existing gap4/top50 sample-gate diagnostic with a shallow greedy decision-tree gate over candidate-bearing sample aggregate features.
- train surrogate selection F1: logistic `0.6559`, tree `0.5933`.
- held-out surrogate selection F1: emit-all `0.4320`, logistic `0.5732`, tree `0.5543`.
- held-out tree TP/FP/FN: `51 / 70 / 12`; held-out logistic TP/FP/FN: `47 / 54 / 16`.
- learned tree rule collapsed to `mask_r4_max_max > 0.9962455`.
- 판단: a nonlinear sample gate does not beat the prior logistic surrogate and still is not actual task F1. This closes "make the CSV sample gate slightly more nonlinear" as a useful next stop-line path.

Latest stop-line GT sample-gate oracle audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_oracle_sample_gate_val512_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-oracle-sample-gate-audit`
- changed axis: keep the checkpoint fixed, then use GT stop-line sample presence only to suppress emissions on negative samples while comparing non-oracle score/top-k ranking against local feature ranking.
- baseline stop-line F1: `0.4083`, TP/FP/FN `98 / 111 / 173`.
- score-threshold production reference: `0.4371`, TP/FP/FN `106 / 108 / 165`.
- GT sample gate best: `gt_sample_gate_max_top10_score` stop-line F1 `0.4758`, TP/FP/FN `113 / 91 / 158`.
- GT sample gate with score `>=0.80`: `0.4743`, TP/FP/FN `106 / 70 / 165`.
- GT sample gate + local feature ranking: `gt_sample_gate_gap4_max_top50_selector_r4` stop-line F1 `0.4800`, TP/FP/FN `114 / 90 / 157`; `mask_r4_max` ranking is lower at `0.4589`, TP/FP/FN `109 / 95 / 162`.
- gap4/top50 oracle-positive upper bound: `0.6877`, TP/FP/FN `142 / 0 / 129`.
- 판단: perfect sample-level emission plus the best tested local feature rank only reaches `0.4800`, barely above the PCA broader reference `0.4699` and still far below `0.60`. Stop-line is not solved by sample emission gating or shallow local-feature reranking on the current candidate rows; candidate generation/readout geometry remains the blocker.

Latest stop-line hard-negative sampler probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stopline_hard_negative_sampler_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_204742/phase_4/history/epochs.jsonl`
- branch: `exp/lane-family-f1/stopline-candidate-manifest-audit`
- changed axis: keep row-scan/tangent, core lane target, loss weights, and freeze policy fixed, but add a `stopline_negative` task-positive bucket that samples stop-line-capable lane-source images with no stop-line GT.
- best exact val128 epoch2 objective: `0.6063`
- lane / stop-line / crosswalk F1: `0.5475 / 0.4561 / 0.5644`
- TP/FP/FN stop-line: `26 / 28 / 34`
- 판단: stop-line is only `+0.0078` over tangent-link exact `0.4483`, while lane, crosswalk, and objective are below current exact references. Hard-negative exposure alone is not the missing selector/readout contract and is not expanded to broader-val512.

Latest stop-line candidate manifest failure-mode audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_current_candidate_manifest_failure_audit_val512_epoch2/summary.json`
- tool: `tools/analyze_pv26_stopline_candidate_manifest.py`
- gap4/max candidate-bearing samples: `369`
- buckets: GT-negative candidate-bearing `165`, positive top-oracle `113`, positive misrank `29`, positive no-oracle `62`.
- positive rates: top-oracle `0.5539`, has-oracle `0.6961`.
- positive no-oracle nearest-distance bins: `40_80=28`, `gte80=34`; there are no `<40` cases in that bucket.
- 판단: sampler-only negative exposure is not enough, and selector-only ranking can recover only the `29` misrank samples. A large part of the stop-line gap is still candidate generation / midpoint proposal recovery, not just production thresholding.

Latest enriched stop-line candidate manifest audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_current_candidate_manifest_failure_audit_enriched_val512_epoch2/summary.json`
- changed axis: add GT/candidate length ratios and bucket-level numeric quantiles to the manifest analyzer.
- `positive_no_oracle` top score q50 is `0.9998`, nearest GT distance q50 is `89.98px`, and nearest candidate length ratio q50 is only `0.066`.
- 판단: the failing positive no-oracle samples are not low-score candidates waiting for a ranker. They are high-confidence, short fragments whose centers are far from the GT midpoint. Next stop-line work must change center proposal / extent recovery, not another score or sample-gate threshold.

Latest learned stop-line fragment-to-center extent probe:

- branch/worktree: `exp/lane-family-f1/stopline-fragment-extent-recovery`.
- code commit: `64c7d70` adds an opt-in `stop_line_fragment_center_offset` head, fragment-line targets, fragment auxiliary loss, and fragment-extent postprocess decode.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_stop_fragment_extent_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260513_033355/phase_4/history/epochs.jsonl`.
- changed axis: keep checkpoint, sampler, freeze policy, and task weights fixed, then train a dense fragment pixel -> full stop-line center/angle/extent recovery contract with `stopline_fragment_extent_aux_weight=0.75` and `stop_line_fragment_extent_enabled=true`.
- best exact val128 objective: `0.5499257237` at epoch1.
- epoch1 lane / stop-line / crosswalk F1: `0.5147 / 0.0267 / 0.6625`.
- epoch2 lane / stop-line / crosswalk F1: `0.5222 / 0.0519 / 0.5389`.
- epoch2 stop-line TP/FP/FN: `4 / 90 / 56`.
- fragment-disabled replay on the same checkpoints: epoch1 stop-line F1 `0.1233`, epoch2 stop-line F1 `0.2517`.
- skipped steps: `0`.
- 판단: runtime and shape-aware handoff are stable, but the learned fragment-to-center extent head collapses stop-line task F1 far below tangent-link exact `0.4483`, PCA val128 `0.5133`, and read-only fragment-extent replay `0.4918`. Disabling the new decode recovers some F1 but not the baseline, so both the decode path and the auxiliary-trained checkpoint are weak. Do not broaden to val512 or repeat as fragment top-k/min-score/aux-weight sweep.

Latest stop-line fragment-union readout replay:

- branch/worktree: `exp/lane-family-f1/stopline-fragment-union-readout`.
- code commit: `0b8aa6f` adds `tools/probe_pv26_stopline_fragment_union_readout.py` and regression coverage.
- follow-up code commit: `ed9fc12` promotes the same readout into an opt-in model-output postprocess/evaluator path on the experiment branch.
- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_union_readout_val512_epoch2/summary.json`.
- productionized replay artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_union_postprocess_val512_epoch2/summary.json`.
- changed axis: keep the current-composite candidate CSV fixed, then cluster same-line high-score short stop-line fragments and emit the merged segment. GT is used only for evaluation, not prediction.
- support accounting: total stop-line support `271`, candidate-bearing GT `257`, missing-GT FN add-on `14`.
- best broader-val512 variant: `union_a12_o36_s080_c2_fallback_top`.
- lane / stop-line / crosswalk F1: `0.5480 / 0.4948 / 0.6187`.
- TP/FP/FN stop-line: `120 / 94 / 151`.
- non-fallback best: `union_a16_o48_s080_c2`, stop-line F1 `0.4936`, TP/FP/FN `116 / 83 / 155`.
- productionized replay result: same broader-val512 stop-line F1 `0.4948453608`, TP/FP/FN `120 / 94 / 151`, lane/crosswalk F1 `0.5480 / 0.6187`, phase objective `0.6373915315`.
- 판단: fragment union is the first current-composite stop-line replay in this lane to beat score-threshold production `0.4371`, PCA broader reference `0.4699`, and GT sample-gate + same-row feature rank `0.4800`, and the opt-in evaluator path reproduces the CSV result. It is still below `0.60`, remains an experiment-branch opt-in rather than a default decoder, and should not trigger another threshold-only sweep.

Latest stop-line fragment follow-up replays:

- seed-extension branch/worktree: `exp/lane-family-f1/stopline-fragment-seed-extend`.
- seed-extension code commit: `900c9b8`.
- seed-extension artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_seed_extend_readout_val512_epoch2/summary.json`.
- seed-extension changed axis: preserve high-score union seed clusters, then let lower-score same-line fragments extend those clusters without creating standalone predictions.
- seed-extension result: best `seed_extend_s080_e065_c2_fallback_top`, stop-line F1 `0.4742`, TP/FP/FN `115 / 99 / 156`.
- length-competition branch/worktree: `exp/lane-family-f1/stopline-fragment-length-competition`.
- length-competition code commit: `1619fd9`.
- length-competition audit commit: `206dc26`.
- length-competition artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_length_competition_readout_val512_epoch2/summary.json`.
- length-competition delta audit artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_readout_delta_val512_epoch2/summary.json`.
- length-competition changed axis: keep union groups, let high-confidence single candidates compete with them, and rank the one emitted stop-line by no-GT length evidence.
- length-competition result: best `length_comp_single090_length`, lane / stop-line / crosswalk F1 `0.5480 / 0.5031 / 0.6187`, stop-line TP/FP/FN `121 / 89 / 150`.
- delta audit: union vs length labels are `fp_removed=3`, `tp_added=2`, `tp_lost=1`, `same=363`.
- multi-instance branch/worktree: `exp/lane-family-f1/stopline-fragment-multi-instance`.
- multi-instance code commit: `4c4dc91`.
- multi-instance second-gate commit: `e795098`.
- multi-instance artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_multi_instance_readout_val512_epoch2/summary.json`.
- multi-instance gated artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_multi_instance_gated_readout_val512_epoch2/summary.json`.
- multi-instance changed axis: keep fragment-union grouping but allow up to two predictions per sample to test whether the one-stop-line emit cap hides valid additional GT stop-lines.
- multi-instance result: best `multi_a16_o48_s080_c2_top2_fallback`, lane / stop-line / crosswalk F1 `0.5480 / 0.5040 / 0.6187`, stop-line TP/FP/FN `125 / 100 / 146`.
- multi-instance gated result: best `multi_a16_o48_s080_c2_top2_second_lenratio070`, lane / stop-line / crosswalk F1 `0.5480 / 0.5061 / 0.6187`, stop-line TP/FP/FN `124 / 95 / 147`.
- projection-split branch/worktree: `exp/lane-family-f1/stopline-fragment-projection-split`.
- projection-split code commit: `a3636aa`.
- projection-split artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_projection_split_readout_val512_epoch2/summary.json`.
- projection-split changed axis: split over-merged same-line fragment-union groups by large projection gaps before merging, then apply the same conservative second-instance gate.
- projection-split result: best `proj_split_a16_o48_gap320_c2_top2_second_lenratio070`, lane / stop-line / crosswalk F1 `0.5480 / 0.5112 / 0.6187`, stop-line TP/FP/FN `125 / 93 / 146`.
- projection-competition code commit: `e00df5f`.
- projection-competition artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_fragment_projection_competition_readout_val512_epoch2/summary.json`.
- projection-competition changed axis: let projection-split union groups and high-confidence single candidates compete by no-GT length evidence, while retaining a conservative second-fragment gate.
- projection-competition result: best `proj_comp_length_s090_top2_second_frag5`, lane / stop-line / crosswalk F1 `0.5480 / 0.5164 / 0.6187`, stop-line TP/FP/FN `126 / 91 / 145`.
- projection selector-audit branch/worktree: `exp/lane-family-f1/stopline-projection-selector-audit`.
- projection selector-audit code commits: `4d16e2f`, `d3871ee`.
- projection selector-audit artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_projection_competition_selector_audit_val512_epoch2/summary.json`.
- projection selector-audit result: best single-feature no-GT gate is `component_svd_thickness_mean >= 4.681936370001899`, stop-line F1 `0.5250`, TP/FP/FN `126 / 83 / 145`; GT-presence oracle control reaches `0.5575` but is not a production selector. Multifeature logistic overfits: train F1 `0.6635`, held-out F1 `0.4369` vs held-out baseline `0.4848`, full replay F1 `0.5336`.
- projection photometric-selector branch/worktree: `exp/lane-family-f1/stopline-photometric-selector-audit`.
- projection photometric-selector code commit: `44f4f4e`.
- projection photometric-selector artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/stopline_projection_photometric_selector_audit_val512_epoch2/summary.json`.
- projection photometric-selector result: best raw-image photometric rule is `photo_abs_contrast_side12 >= 8.923120498657227`, stop-line F1 `0.5217`, TP/FP/FN `126 / 86 / 145`; adding photometric features does not beat the prior thickness gate (`0.5250`). Logistic with photometric features overfits harder: train F1 `0.6832`, held-out F1 `0.4433` vs held-out baseline `0.4848`, full replay F1 `0.5442`.
- row/x candidate-consistency branch/worktree: `exp/lane-family-f1/stopline-rowx-candidate-consistency-audit`.
- row/x candidate-consistency code commit: `4c7c164`.
- row/x candidate-consistency artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_rowx_candidate_consistency_audit_20260513/analysis_exports/{val128_epoch2,val512_epoch2}/summary.json`.
- row/x candidate-consistency result: val128 held-out looked promising (`baseline 0.5161`, row/x rich logistic `0.5667`, selector_r4 `0.5846`), but val512 held-out stayed weak (`baseline 0.3877`, row/x rich logistic `0.4087`, selector_r4 `0.4259`) and did not beat PCA/projection references.
- normal-support recenter branch/worktree: `exp/lane-family-f1/stopline-normal-support-recenter-readout`.
- normal-support recenter code commit: `6898a5f`.
- normal-support recenter artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_normal_support_recenter_readout_20260513/analysis_exports/{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- normal-support recenter result: exact val128 baseline is stop-line F1 `0.4483`, TP/FP/FN `26 / 30 / 34`; selector-center angle-mask reference is `0.5085`, `30 / 28 / 30`; best normal-scan variants regress to `0.4354`, `32 / 55 / 28`.
- raw-edge recenter branch/worktree: `exp/lane-family-f1/stopline-raw-edge-recenter-readout`.
- raw-edge recenter code commit: `5abfb41`.
- raw-edge recenter artifacts: `runs/pv26_exhaustive_od_lane_train/stopline_raw_edge_recenter_readout_20260513/analysis_exports/{smoke_val4_epoch2,val128_epoch2}/summary.json`.
- raw-edge recenter result: exact val128 baseline is stop-line F1 `0.4483`, TP/FP/FN `26 / 30 / 34`; selector-center angle-mask reference is `0.5085`, `30 / 28 / 30`; raw-edge normal scan regresses to `0.2993`, `22 / 65 / 38`, and the offset-penalty variant regresses further to `0.2585`, `19 / 68 / 41`.
- 판단: low-score extension regresses and is closed. Length competition is a small partial-positive over fragment union (`0.4948 -> 0.5031`, `+1 TP`, `-5 FP`) but the delta audit shows the gain is confined to six samples and mostly fallback suppression, not broad geometry recovery. Multi-instance and second-instance gating are weak partial positives. Projection splitting gives a real FP improvement, and projection-competition is the current ungated local stop-line readout reference (`0.5164`). The selector audits can suppress some FP but either stay weak (`0.5250` single-feature, `0.5217` photometric) or fail held-out generalization (`0.4369` exported-feature logistic, `0.4433` photometric logistic), so they are not deployable selector paths. Row/x projection consistency also fails to become a stronger held-out selector on val512. Normal-support recentering adds TP on exact val128 but doubles FP enough to fall below baseline, and raw-edge contrast recentering collapses TP while adding FP, so neither is a midpoint recovery path. Do not repeat these as feature-rank, single-score, top-K, second-prediction threshold, projection-gap, projection-competition length/min-score, projection-competition single-feature selector, exported-feature logistic selector, raw-image photometric selector, row/x consistency selector, normal-support recenter, or raw-edge recenter sweeps. The next stop-line step must improve candidate generation/midpoint recovery or introduce a materially new no-GT signal, while preserving crosswalk hull retention and lane composition explicitly.

Latest stop-line center-rank margin probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_center_rank_margin_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_212813/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_stop_center_rank_margin`
- changed axis: add opt-in `stopline_center_rank_margin_weight=0.35`, which pushes the GT stop-line midpoint proposal above top hard-negative center/selector proposals.
- best exact val128 epoch2 objective: `0.6170`
- lane / stop-line / crosswalk F1: `0.5605 / 0.4602 / 0.5854`
- TP/FP/FN stop-line: `26 / 27 / 34`
- 판단: valid runtime and a tiny stop-line gain, but still below tangent-link exact objective `0.6187`, segment-MIL lane-head-only exact objective `0.6193`, and geometry-validator stop-line `0.4655`. Do not broaden; margin-only midpoint ranking is not enough.

Latest stop-line center-rank margin readout replay:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_center_rank_margin_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_212813/analysis_exports/stopline_pred_angle_mask_extent_val128_epoch2/summary.json`
- tool: `tools/probe_pv26_stopline_pred_angle_mask_extent.py`
- changed axis: keep the trained center-rank checkpoint fixed, then replay production-style predicted-proposal + angle-anchored mask-extent stop-line readout.
- best exact val128 variant: `pred_selector_top1_s060_mask050_band4`
- lane / stop-line / crosswalk F1: `0.5252 / 0.5042 / 0.5854`
- TP/FP/FN stop-line: `30 / 29 / 30`
- 판단: readout replay recovers stop-line over the checkpoint baseline `0.4602`, but it still misses the prior angle-mask production exact reference `0.5085` and PCA val128 reference `0.5133`. Do not broaden this checkpoint through angle-mask replay.

Latest center-rank proposal recall audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_center_rank_margin_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_212813/analysis_exports/stopline_proposal_recall_val128_epoch2/summary.json`
- tool: `tools/probe_pv26_stopline_proposal_recall.py`
- changed axis: keep the center-rank checkpoint fixed and measure whether GT stop-line centers moved into better proposal-map top-k positions.
- exact val128 `max` source: `max_r8 >= 0.6` is `50/60`, top3-hit-r8 is `39/60`, top10-hit-r8 is `52/60`, raw rank top3 is `11/60`.
- prior original-checkpoint val128 `max` source: `50/60`, `37/60`, `53/60`, raw rank top3 `9/60`.
- 판단: center-rank margin only gives tiny top3/raw-rank movement and does not improve local recall or top10 recall. This does not recover the positive no-oracle bucket and is not a candidate-generation fix.

Latest lane flip-TTA audit:

- artifact exact: not retained in active runs after artifact pruning; exact metrics below are historical.
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/broader_val512_current_best_flip_centerline_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-flip-tta-audit`
- changed axis: keep checkpoint, stop-line/crosswalk outputs, stop-line `mask=0.80`, and crosswalk hull decode fixed; run a horizontal-flip forward pass and merge only the lane centerline logits by averaging.
- exact val128 objective: `0.6296149306`
- exact lane / stop-line / crosswalk F1: `0.5854 / 0.4364 / 0.5988`
- broader-val512 objective: `0.6216194906`
- broader lane / stop-line / crosswalk F1: `0.5577 / 0.4235 / 0.6187`
- broader lane TP/FP/FN: `4518 / 2206 / 4959`
- 판단: flip centerline averaging is a valid runtime/postprocess partial-positive: it improves broader lane F1 by `+0.0098` and objective by about `+0.0040` over the prior transplanted composite. It does not close lane 0.6, and it does nothing for the stop-line bottleneck.

Latest segment-MIL lane-head-only + flip-TTA replay:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_tta_on_segment_mil_lane_head_only_20260512/analysis_exports/exact_val128_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_tta_on_segment_mil_lane_head_only_20260512/analysis_exports/broader_val512_epoch2/summary.json`
- changed axis: keep the `core_centerline_refine_row_scan_tangent_segment_mil_lane_head_only` checkpoint fixed, then apply only `flip_centerline_avg` lane TTA with stop-line `mask=0.80` and crosswalk hull decode.
- exact val128 `flip_centerline_avg`: objective `0.6298`, lane/stop/cross F1 `0.5854 / 0.4364 / 0.6061`.
- broader-val512 `flip_centerline_avg`: objective `0.6207`, lane/stop/cross F1 `0.5577 / 0.4184 / 0.6185`.
- 판단: exact looks slightly stronger than the segment-MIL baseline, but broader does not beat the current broader best objective `0.6231`, and stop-line is lower than the current broader best `0.4235`. Do not promote this combination.

Latest lane geometry-filter probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512/analysis_exports/lane_geometry_filter_probe_val128_epoch2/summary.json`
- changed axis: keep the lane-head transplant checkpoint fixed and test stricter lane bbox area/aspect filters on exact val128 before any broader replay.
- baseline lane F1: `0.5660`, TP/FP/FN `1162 / 554 / 1228`.
- best stricter candidate by objective, `lane_bbox_area_8192`, lane F1: `0.5580`, TP/FP/FN `1097 / 445 / 1293`.
- 판단: stricter geometry filters reduce FP but lose too many TP. This is not a lane 0.6 path and is not broadened.

Latest stop-line flip-TTA audit:

- artifact smoke: `runs/pv26_exhaustive_od_lane_train/lane60_stopline_flip_tta_audit_20260512/analysis_exports/smoke_val4_current_best_epoch2/summary.json`
- branch: `exp/lane-family-f1/stopline-flip-tta-audit`
- changed axis: keep checkpoint, lane/crosswalk outputs, stop-line `mask=0.80`, and crosswalk hull decode fixed; run a horizontal-flip forward pass and merge only stop-line dense score maps and/or geometry maps by averaging.
- smoke val4 variants: `baseline`, `flip_stop_score_avg`, `flip_stop_geometry_avg`, `flip_stop_all_avg`.
- smoke val4 result: all four variants are identical at objective `0.6395`, lane/stop-line/crosswalk F1 `0.5507 / 0.0000 / 0.5455`, stop-line TP/FP/FN `0 / 3 / 2`.
- 판단: this has no stop-line signal even as a smoke probe, so it was not expanded to val128/broader. Do not treat simple stop-line flip score/geometry averaging as a current improvement path.

Latest lane flip-consistency instance-evidence audit:

- artifact exact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/exact_val128_current_best_epoch2/summary.json`
- artifact broader: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/broader_val512_current_best_epoch2/summary.json`
- artifact exact flip-centerline-avg follow-up: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/exact_val128_flip_centerline_avg_instance_gate_epoch2/summary.json`
- artifact broader flip-centerline-avg follow-up: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/broader_val512_flip_centerline_avg_instance_gate_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-flip-instance-evidence`
- changed axis: keep the current transplanted composite and row-scan tangent lane decode fixed, add normal/flip centerline agreement features to the existing read-only lane instance evidence validator, then replay the same row gate on top of the existing `flip_centerline_avg` runtime lane baseline.
- exact val128 heldout result: logistic instance gate lane F1 `0.6002` vs heldout baseline `0.5843`, but stop-line/crosswalk heldout F1 `0.4667 / 0.5952`.
- broader val512 heldout result: logistic instance gate lane F1 `0.5660` vs heldout baseline `0.5473`; full split-count replay implies lane F1 `0.5682`, TP/FP/FN `4332 / 1439 / 5145`, compared with baseline `0.5480`, `4457 / 2333 / 5020`.
- no-flip broader ablation: `runs/pv26_exhaustive_od_lane_train/lane60_lane_flip_instance_evidence_20260512/analysis_exports/broader_val512_no_flip_ablation_epoch2/summary.json`; logistic heldout lane F1 `0.5663`, full split-count lane F1 `0.5675`.
- flip-centerline-avg follow-up: exact heldout logistic lane F1 `0.6125`, but broader heldout lane F1 only `0.5694`; broader full split-count lane F1 is `0.5738`, TP/FP/FN `4455 / 1596 / 5022`, with stop-line/crosswalk unchanged at `0.4235 / 0.6187`.
- flip-only single features are weaker than the logistic mix on broader heldout: `flip_center_point_mean` lane F1 `0.5587`, `center_consensus_point_mean` `0.5583`.
- 판단: lane instance gating has real row-level signal and can suppress many lane FP, and the strongest read-only lane diagnostic is now `0.5738` full split-count on top of `flip_centerline_avg`. It still trades away TP/FN, leaves broader lane below `0.60`, and does not address stop-line. It is read-only post-hoc evidence, not a production decoder or all-task success.

Latest lane instance oracle-selection audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_instance_oracle_selection_20260512/analysis_exports/broader_val512_flip_centerline_avg_oracle_selection_retention_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-instance-oracle-selection-audit`
- changed axis: keep the current `flip_centerline_avg` runtime lane baseline plus stop-line/crosswalk retention overrides fixed, then add a GT-matching oracle selector that keeps only row-scan lane predictions already matched as TP. This is read-only diagnostic evidence, not production decode.
- broader full baseline: lane/stop/cross F1 `0.5577 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 2206 / 4959`.
- oracle TP-only selector: lane/stop/cross F1 `0.6457 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 0 / 4959`.
- learned logistic heldout reference under the same retention config: lane F1 `0.5700`, TP/FP/FN `2173 / 830 / 2449`, below the oracle selector and below lane `0.60`.
- 판단: current row-scan/flip candidate set has enough matched lane predictions for a perfect selector to exceed lane `0.60`; the lane gap is primarily FP suppression / instance selection, not only missing lane candidates. But the available logistic/post-hoc selector still falls far short, and stop-line remains unchanged at `0.4235`, so all-task success is still blocked.

Latest lane row-scan duplicate suppression audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_duplicate_suppression_20260512/analysis_exports/broader_val512_flip_centerline_avg_duplicate_d24_retention_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-row-scan-duplicate-suppression`
- changed axis: keep the current `flip_centerline_avg` runtime lane baseline, stop-line retention override, and crosswalk hull decode fixed, then suppress only same-class/same-type lane rows whose mean point distance is `<=24` pixels.
- broader full baseline: lane/stop/cross F1 `0.5577 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 2206 / 4959`.
- duplicate suppression: lane/stop/cross F1 `0.5578 / 0.4235 / 0.6187`, lane TP/FP/FN `4518 / 2205 / 4959`.
- held-out duplicate suppression similarly removes only one FP: lane F1 `0.5562` vs baseline `0.5561`, TP/FP/FN `2212 / 1120 / 2410`.
- 판단: the oracle selector gap is not explained by near-duplicate row-scan emissions. A simple same-schema geometric dedupe is a no-op for the current broader FP problem, so the next lane path still needs a stronger instance selection/readout contract rather than distance-threshold dedupe.

Latest lane per-sample top-k cap audit:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_lane_sample_topk_cap_20260512/analysis_exports/broader_val512_flip_centerline_avg_sample_top5_logistic_retention_epoch2/summary.json`
- branch: `exp/lane-family-f1/lane-sample-topk-cap-audit`
- changed axis: keep the same `flip_centerline_avg` runtime lane baseline plus stop-line/crosswalk retention overrides fixed, then cap each sample to its top 5 row-scan lane predictions by the existing logistic evidence score.
- broader full result: lane/stop/cross F1 `0.5618 / 0.4235 / 0.6187`, lane TP/FP/FN `4490 / 2016 / 4987`, vs baseline `0.5577`, `4518 / 2206 / 4959`.
- broader held-out result: lane F1 `0.5596`, TP/FP/FN `2201 / 1043 / 2421`, vs held-out baseline `0.5561`, `2212 / 1121 / 2410`.
- 판단: per-sample cap removes some FP but also drops TP. The gain is small and remains below the stronger logistic threshold diagnostic (`0.5700` held-out) and far below the oracle TP-only selector (`0.6457` full). Simple sample-level emission capping is not enough to recover the lane oracle gap.

Git-history backfill: closed architecture probes not in the current code line:

- `exp/lane-family-f1/lane-centerline-dilated-context`: centerline branch dilation was operational but exact objective `0.6155`, lane/stop/cross F1 `0.5617 / 0.4348 / 0.5854`, below tangent-link and segment-MIL lane-head-only references.
- `exp/lane-family-f1/lane-support-conditioned-centerline`: detached support-conditioned centerline refinement was operational but exact objective `0.6152`, lane/stop/cross F1 `0.5607 / 0.4348 / 0.5854`, below the same references.
- `exp/lane-family-f1/stopline-center-stem`: wiring the unused stop-line `center_stem` into center outputs was a valid architecture cleanup but exact stop-line F1 fell to `0.3704`, TP/FP/FN `20 / 28 / 40`.
- 판단: these are retained as negative branch evidence, not merged code. The next lane/stop-line axis should not repeat receptive-field widening, support-conditioned centerline refinement, or stop-line center-stem cleanup.

Latest lane vectorizer scope probe:

- artifact: `analysis_exports/lane_row_scan_tangent_component_val128_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_component`
- objective: `0.6106257251980678`
- lane / stop-line / crosswalk F1: `0.5305 / 0.4483 / 0.5854`
- lane TP/FP/FN: `1030 / 463 / 1360`
- 판단: component 내부로만 tangent row-scan 연결을 제한하면 baseline보다는 lane이 조금 오르지만 current tangent-link exact reference `0.5633`, `1121 / 469 / 1269`에 크게 못 미친다. over-link를 줄이는 대신 TP를 많이 잃는 쪽이라 broader-val512로 확장하지 않는다.

Latest lane row-scan link oracle audit:

- artifact: `analysis_exports/lane_tangent_oracle_val128_epoch2/summary.json`
- experiment: `core_centerline_refine_row_scan_tangent_link`
- read-only variants: keep the checkpoint fixed, then replace only selected seg-first lane maps before postprocess.
- baseline exact val128 lane F1: `0.5633`, lane TP/FP/FN `1121 / 469 / 1269`.
- `gt_tangent_axis` lane F1: `0.5611`, lane TP/FP/FN `1116 / 472 / 1274`.
- `gt_centerline_core` lane F1: `0.6778`, lane TP/FP/FN `1243 / 35 / 1147`.
- `gt_centerline_core_gt_tangent` lane F1: `0.6781`, lane TP/FP/FN `1244 / 35 / 1146`.
- 판단: tangent-axis oracle alone does not rescue row-scan linking; GT centerline core does. The next lane axis should target predicted centerline coverage/quality as an instance-level contract, not another tangent/link cost sweep.

Latest lane centerline instance-balance probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_instance_balance_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_071343/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_instance_balance`
- changed axis: add `lane_seg_centerline_instance_weight` and opt-in `lane_segfirst_instance_centerline_weight=0.35`.
- best exact val128 epoch2 objective: `0.6161312489`
- lane / stop-line / crosswalk F1: `0.5654 / 0.4310 / 0.5854`
- lane TP/FP/FN: `1171 / 581 / 1219`
- 판단: lane F1 is slightly above tangent-link exact `0.5633`, but below segment-MIL lane-head-only `0.5660`; stop-line regresses below tangent-link `0.4483`. Instance-balanced positive centerline loss alone is not a broader-val512 expansion path.

Latest lane soft-instance shell probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_soft_instance_shell_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_093000/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_soft_instance_shell`
- changed axis: keep the core centerline target and add a separate instance-balanced soft-shell BCE around each lane centerline with `lane_segfirst_soft_instance_centerline_weight=0.35`.
- best exact val128 epoch2 objective: `0.6159063607`
- lane / stop-line / crosswalk F1: `0.5641 / 0.4348 / 0.5854`
- lane TP/FP/FN: `1144 / 522 / 1246`
- stop TP/FP/FN: `25 / 30 / 35`
- 판단: lane F1 is only noise-level above tangent-link exact `0.5633` and below segment-MIL lane-head-only `0.5660`; stop-line still regresses below tangent-link `0.4483`. Soft-shell instance-balanced side supervision is not a broader-val512 expansion path.

Latest lane centerline soft-ignore probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_soft_ignore_band_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_095254/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_soft_ignore_band`
- changed axis: keep `lane_segfirst_centerline_target_mode=core`, but exclude non-core pixels with `lane_seg_centerline_soft >= 0.20` from centerline BCE/Dice/Focal loss.
- best exact val128 epoch2 objective: `0.6088389162`
- lane / stop-line / crosswalk F1: `0.5360 / 0.4348 / 0.5854`
- lane TP/FP/FN: `1032 / 429 / 1358`
- stop TP/FP/FN: `25 / 30 / 35`
- 판단: soft-shell negative masking cuts FP but loses too much lane TP, far below tangent-link exact `0.5633`, instance-balance `0.5654`, soft-instance shell `0.5641`, and segment-MIL lane-head-only `0.5660`. Soft-band ignore/threshold-only masking is not a broader-val512 expansion path.

Latest lane centerline threshold oracle audit:

- artifact: `analysis_exports/lane_centerline_threshold_oracle_val128_epoch2/summary.json`
- script: `tools/probe_pv26_lane_centerline_threshold_oracle.py`
- experiment: fixed `core_centerline_refine_row_scan_tangent_link` checkpoint, then compare global `lane_obj_threshold` candidates against a per-sample dense-core oracle threshold.
- best global result: `global_t020`, lane/stop/cross F1 `0.5641 / 0.4483 / 0.5988`, lane TP/FP/FN `1153 / 545 / 1237`.
- current reference-like `global_t045`: lane/stop/cross F1 `0.5633 / 0.4483 / 0.5988`, lane TP/FP/FN `1121 / 469 / 1269`.
- `sample_oracle_dense_core`: lane/stop/cross F1 `0.5482 / 0.4483 / 0.5988`, lane TP/FP/FN `1074 / 454 / 1316`, selected-threshold mean/min/max `0.6380 / 0.2000 / 0.8000`.
- 판단: global threshold calibration adds only noise-level lane gain over the current tangent-link exact reference and still stays below the exact lane-head-only best `0.5660`. Sample-wise dense-core thresholding is worse. Lane threshold-only calibration/readout is closed; the next lane axis needs a stronger predicted-centerline instance contract rather than another threshold sweep.

Latest stop-line freeze-policy probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_geometry_validator_stop_head_only_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260511_231815/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_stop_geometry_validator_stop_head_only`
- objective: `0.6179851856657268`
- lane / stop-line / crosswalk F1: `0.5640 / 0.4386 / 0.5926`
- lane TP/FP/FN: `1121 / 464 / 1269`
- stop TP/FP/FN: `25 / 29 / 35`
- cross TP/FP/FN: `48 / 33 / 33`
- 판단: stop-line head만 업데이트해도 geometry-validator stop-line gain이 보존되지 않았다 (`0.4655 -> 0.4386`). head isolation은 lane/crosswalk retention에는 도움되지만 stop-line rescue가 아니므로 broader-val512로 확장하지 않는다.

Latest stop-line candidate-select contract probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_080843/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4`
- changed axis: supervise denser `gap4/top50` stop-line candidates into `stop_line_candidate_validator_logits` and `stop_line_presence_logits`, then read with `stop_line_component_gate_source=max_validator`.
- best exact val128 objective: epoch1 `0.5980293261`; epoch2 dropped to `0.5855145029`.
- epoch2 lane / stop-line / crosswalk F1: `0.5612 / 0.0000 / 0.5854`
- epoch2 stop TP/FP/FN: `0 / 0 / 60`
- 판단: runtime and plumbing are valid, but the candidate-select + presence/max-validator gate suppresses stop-line emission completely. This is below tangent-link, PCA, angle-mask, task-head merge, and geometry-validator references, so broader-val512로 확장하지 않는다.

Latest stop-line candidate-select gate replay:

- artifact default: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_080843/analysis_exports/candidate_select_gate_replay_default_val128_epoch2/summary.json`
- artifact no-presence: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_stop_candidate_select_gap4_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_080843/analysis_exports/candidate_select_gate_replay_max_validator_no_presence_val128_epoch2/summary.json`
- changed axis: evaluator-only replay of the same `best.pt`; no model/loss/sampler changes.
- default `max_validator + presence=0.35`: lane/stop/cross F1 `0.5540 / 0.0000 / 0.5590`, stop TP/FP/FN `0 / 0 / 60`.
- `max_validator + presence=0.0`: lane/stop/cross F1 `0.5540 / 0.4483 / 0.5590`, stop TP/FP/FN `26 / 30 / 34`.
- `center + presence=0.0`: lane/stop/cross F1 `0.5540 / 0.4386 / 0.5590`, stop TP/FP/FN `25 / 29 / 35`.
- 판단: 0-emission failure is mainly the sample-level presence gate, not the `max_validator` component gate alone. But removing presence only recovers the old tangent-link-level stop-line F1 and crosswalk on `best.pt` is weak, so this is diagnostic evidence, not a 0.6 path.

Latest preprocessing/runtime probe:

- artifact: `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_row_scan_tangent_no_aug_from_lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412_default_20260512_044501/phase_4/history/epochs.jsonl`
- experiment: `core_centerline_refine_row_scan_tangent_no_aug`
- changed axis: keep `row_scan_tangent` and loss/postprocess contract fixed, disable stage-4 `train_augmentation`.
- objective: `0.6093231418573035`
- lane / stop-line / crosswalk F1: `0.5595 / 0.4298 / 0.5476`
- lane TP/FP/FN: `1109 / 465 / 1281`
- stop TP/FP/FN: `26 / 35 / 34`
- cross TP/FP/FN: `46 / 41 / 35`
- 판단: augmentation-off wiring and runtime are valid, but exact epoch2 is below tangent-link reference `0.6187`, `0.5633 / 0.4483 / 0.5854`. Crosswalk task-best reached `0.6788` at epoch1 but the selected epoch2 checkpoint loses crosswalk retention, so broader-val512로 확장하지 않는다.

Stop-line Gate 2 follow-up:

- 여러 worktree에서 threshold, sampler, mask loss, mask vectorizer, center target, dense geometry target, selector center, center stem, component fitting, endpoint supervision, feature isolation을 한 축씩 시험했다.
- best decoder-only 후보는 original checkpoint + `component_pca_full_mask080_score094`로 broader-val512 stop-line F1 `0.4699`, TP/FP/FN `113 / 97 / 158`이다.
- 이 값은 기준선 `0.4083`보다 낫지만, 목표 `0.60`까지는 아직 멀다.
- stop-line dense mask pixel F1은 val128 probe에서 `0.6107`까지 나오지만, center heatmap F1은 `0.1385`라서 mask 존재보다 endpoint/geometry/selector 복원이 병목이다.
- 이후 component split, center-cell geometry mask, half-length scale/loss/log target, learned query-vector proposal, endpoint-delta target/readout도 각각 별도 branch에서 닫았다. 모두 stop-line F1 0.6 path가 아니었다.
- query-vector proposal short run은 vector-only exact val128 epoch1/2 stop-line F1이 모두 `0.0000`이고, threshold를 `0.10`까지 낮춰도 TP/FP/FN `0 / 0 / 55`였다. append mode도 mask baseline 수준에 머물렀다.
- selector-map을 component gate/anchor에 opt-in으로 연결한 read-only decode probe도 stop-line F1 `0.2062`로 `stop_mask_only` `0.2593`보다 낮았다. centerline selector map을 단순 component 선택에 쓰는 후처리만으로는 0.6 path가 아니다.
- selector/row/x dense-map audit은 val128 epoch2에서 stop-line mask F1 `0.6646`, row-mask F1 `0.6727`, x-mask F1 `0.7482`를 보였지만 selector-mask F1은 `0.3119`, selector-centerline F1은 `0.4259`에 그쳤다. x projection과 mask signal은 남아 있지만, selector/proposal readout이 full line segment로 결합되지 못하는 쪽이 병목이다.
- row/x projection을 직접 span proposal로 바꾸는 read-only probe도 baseline exact val128 stop-line F1 `0.4483`을 넘지 못했다. best fallback은 `0.4310`, best replacement는 `0.3146`이고, row/x decoder는 512 samples 중 29~30 samples에서만 line을 만들었다. 기존 row/x 신호를 후처리 span으로 꺼내는 방식도 0.6 path가 아니다.
- row/x/mask를 학습 target부터 묶는 간단한 rowx-band selector contract도 짧은 run에서 stop-line을 회복하지 못했다. `core_centerline_refine_stop_selector_rowx_band` exact val128 epoch2는 `phase_objective=0.6036`, lane/stop/cross F1 `0.5260 / 0.4144 / 0.5854`, stop-line TP/FP/FN `23 / 28 / 37`로 기준 `0.5267 / 0.4483 / 0.5854`보다 stop-line이 낮다. epoch1 crosswalk task-best `0.6790`은 joint goal signal이 아니다.
- GT-center + angle-anchored mask extent diagnostic은 half-length scalar 대신 mask extent로 길이를 읽으면 headroom이 있음을 보였다. val128에서 predicted angle + mask extent는 stop-line F1 `0.6126`, TP/FP/FN `34 / 17 / 26`까지 올라가지만, broader-val512에서는 `0.5361`, TP/FP/FN `130 / 84 / 141`로 목표 미달이다. GT angle upper bound도 broader-val512 `0.5608`라서 최종 성공은 아니지만, 다음 production 후보는 half-length scalar가 아니라 center proposal + angle-anchored mask extent readout이다.
- production predicted proposal + angle-anchored mask extent readout도 exact val128에서 PCA reference를 넘지 못했다. best `pred_selector_top1_s060_mask050_band4`는 stop-line F1 `0.5085`, TP/FP/FN `30 / 28 / 30`으로 baseline `0.4483`보다 낫지만 prior PCA val128 reference `0.5133`보다 낮다. broader-val512로 확장하지 않는다.
- proposal recall audit은 broader-val512에서 GT 주변 local proposal signal은 남아 있지만 top-k ranking이 약하다는 쪽을 보였다. `max(center, selector)` map은 `max_r8 >= 0.6`이 `208/271`이지만 top3-r8 hit는 `137/271`, raw rank top3는 `14/271`뿐이다. 다음 stop-line 축은 local score 존재 여부보다 proposal ranking/competition 또는 candidate validation contract다.
- candidate-pool audit은 top10/top20 후보 안에 valid stop-line segment headroom이 있음을 보였다. broader-val512 oracle-positive selection은 stop-line F1 `0.6517`, TP/FP/FN `131 / 0 / 140`까지 가능하지만, production score/length filters best는 `0.4371`, TP/FP/FN `106 / 108 / 165`로 PCA broader reference `0.4699`보다 낮다. 즉 candidate pool은 있으나 score/length만으로는 FP suppression signal이 부족하다.
- candidate generation gap audit은 top-k NMS gap을 `10 -> 6/4`로 줄이고 top-k를 `20 -> 50`까지 넓히면 oracle-positive headroom이 더 있음을 보였다. broader-val512 `gap4_oracle_max_top50_positive`는 stop-line F1 `0.6877`, TP/FP/FN `142 / 0 / 129`로 기존 oracle top20 `0.6517`, `131 / 0 / 140`보다 높다. 하지만 production `gap4_max_top50_score_s080`은 기존 `max_top10_score_s080`과 같은 `0.4371`, TP/FP/FN `106 / 108 / 165`에 머문다. 결론은 dense candidate pool을 더 넓히는 것만으로는 production path가 아니고, denser 후보를 고를 task-aware selector/readout contract가 따로 필요하다는 것이다.
- hard-negative proposal ranking loss short run은 proposal competition을 직접 누르는 opt-in loss를 시험했지만 exact val128에서 확장 조건을 만들지 못했다. stage4-only 2epoch, train512/val128, weight `0.25` best는 lane/stop/cross F1 `0.4754 / 0.4918 / 0.5767`, stop-line TP/FP/FN `30 / 32 / 30`이다. stop-line은 baseline `0.4483`보다 높지만 angle-mask production `0.5085`와 PCA val128 reference `0.5133`보다 낮고 lane/crosswalk가 같이 내려갔다. 같은 checkpoint의 candidate-pool production best도 stop-line F1 `0.4286`이라 broader-val512로 확장하지 않는다.
- candidate scalar-feature validator audit은 existing CSV feature만으로 후보 row label을 어느 정도 분리할 수 있음을 보였다. val512 half-split logistic은 test AUC `0.7894`, AP `0.5699`, candidate-level F1 `0.5651`; score 단일 feature도 test candidate-level F1 `0.5812`다. 하지만 이건 candidate-row 분류이고, 같은 scalar score/length production replay는 stop-line task F1 `0.4371`에 머물렀다. 따라서 scalar CSV classifier를 production validator로 보지 않고, 실제 instance/task-aware validator head가 필요하다.
- candidate rich-feature validator audit은 proposal 좌표, decoded center 좌표, center/selector/mask point/window probability까지 CSV에 추가하면 row-level separability가 더 좋아짐을 보였다. val512 half-split logistic은 33개 feature에서 test AUC/AP/F1 `0.8149 / 0.5950 / 0.6175`, oracle-best row F1 `0.6387`이고, 단일 feature로도 `selector_r4_max`가 row F1 `0.6109`를 냈다. 그러나 candidate-pool task replay 자체는 그대로이고 best production stop-line F1은 여전히 `0.4371`이다. 결론은 "rich map-local signal은 model-side candidate instance validator 후보를 정당화한다"이지 "CSV threshold decoder가 성공했다"가 아니다.
- candidate rich-validator held-out task replay는 row signal이 task selection으로 일부 옮겨지지만 gate를 통과하지 못함을 보였다. val512 앞 256 batch에서 threshold를 맞추고 뒤 256 batch를 평가하면 held-out baseline stop-line F1 `0.3877` 대비 rich logistic `0.4231`, `selector_r4_max` `0.4259`까지 오른다. 하지만 PCA broader reference `0.4699`에도 못 미치고, 0.6 목표와는 거리가 크다. 따라서 rich CSV/logistic threshold replay는 production path가 아니며, stop-line을 계속한다면 단순 score replay가 아니라 실제 model-side instance validator loss/head가 필요하다.
- gap4/top50 denser pool 위의 rich-selector held-out task replay는 약한 개선만 만들었다. 같은 val512 half split에서 `--rich-validator-min-gap 4.0 --rich-validator-top-k 50`을 쓰면 held-out baseline stop-line F1 `0.3877` 대비 rich logistic task threshold `0.4190`, rich logistic row threshold `0.4298`, `selector_r4_max` task threshold `0.4537`까지 오른다. 기존 gap10/top20 `selector_r4_max` `0.4259`보다는 높지만 PCA broader reference `0.4699`와 0.6 목표에는 못 미친다. denser 후보와 rich local feature는 selector/readout contract의 전제 evidence일 뿐, CSV threshold replay 자체는 production path가 아니다.
- selector feature-patch validator audit은 dense `stop_line_selector_feature`에서 proposal/decoded center 주변 128-channel patch mean을 꺼내 289-feature logistic replay를 시험했지만 gap4/top50 held-out task F1이 `0.3982`에 그쳤다. 같은 split의 held-out baseline `0.3877`보다는 아주 조금 높지만 기존 `selector_r4_max` task threshold `0.4558`보다 낮고 PCA broader reference `0.4699`와도 멀다. 따라서 raw selector feature embedding을 offline logistic/MLP로 더 키우는 방향은 현재 evidence만으로 production path가 아니다.
- gap4/top50 candidate feature rank diagnostic은 positive sample 안에서는 local mask/selector 신호가 oracle 후보를 꽤 잘 올린다는 것을 보였다. `mask_r4_max`는 oracle-positive가 있는 142 sample 중 top1 hit `116/142` (`0.8169`), top3 hit `124/142` (`0.8732`)다. 하지만 후보가 있는 sample은 369개이고 그중 227개는 oracle-positive가 없는 negative candidate sample이다. 즉 다음 stop-line 병목은 "positive sample 안에서 어떤 후보가 1등인가"만이 아니라, "후보가 있는 sample에서 emit/no-emit을 어떻게 가르는가"다. 단순 ranking feature나 threshold replay는 production path가 아니다.
- gap4/top50 sample-level emit gate audit은 이 premise를 더 좁혔다. 같은 candidate rows에서 held-out emit-all surrogate selection F1은 `0.4320`이고, sample logistic gate는 held-out sample F1 `0.6585`, surrogate selection F1 `0.5732`까지 올렸다. 하지만 이것은 oracle-label surrogate이고 task F1이나 production decoder가 아니므로, 다음 stop-line 축은 이를 실제 task replay 또는 model-side emit gate로 검증해야 한다.
- gap4/top50 sample-gate actual task replay는 surrogate gain이 evaluator task F1로 전이되지 않음을 보였다. 같은 threshold `0.2471111762446392`에서 train stop-line F1은 `0.4269 -> 0.4640`으로 올랐지만 held-out은 `0.3877 -> 0.3843`으로 낮아졌다. Full val512 합산도 stop-line TP/FP/FN `102 / 106 / 169`, F1 `0.4259`로 PCA broader reference `0.4699`와 0.6 목표에 못 미친다. CSV logistic sample gate는 production path가 아니다.
- gap4/top50 baseline-preserving rescue replay는 sample-gate 후보를 baseline에 fallback/append하면 train overfit을 줄일 수 있는지 봤지만 역시 gate를 넘지 못했다. Full val512 합산 기준 best rescue는 `baseline_or_gate` stop-line F1 `0.4165`, TP/FP/FN `106 / 132 / 165`이고, `baseline_plus_gate_max2`는 `0.4141`, `100 / 112 / 171`이다. Baseline `0.4083`보다는 약간 높지만 기존 sample-gate actual replay `0.4259`, score-threshold production `0.4371`, PCA broader reference `0.4699`보다 낮다. baseline 보존형 sample-gate rescue도 production path가 아니다.
- model-side candidate instance validator head는 opt-in 구현과 runtime smoke는 통과했지만 exact val128 gate에서 실패했다. Direct `validator` gate는 epoch1/2 stop-line F1이 모두 `0.0000`이고, 같은 epoch2 checkpoint를 baseline `center` gate로 되돌려도 stop-line F1 `0.4211`로 기준 `0.4483`보다 낮다. cold validator gate를 바로 production gate로 쓰는 방식과 현재 hard-negative auxiliary 조합은 0.6 path가 아니다.
- candidate validator calibrated replay는 direct collapse를 일부 회복했지만 exact val128 gate를 넘지 못했다. Same checkpoint에서 `product_validator` + fallback best는 lane/stop/cross F1 `0.5271 / 0.4918 / 0.5854`, stop-line TP/FP/FN `30 / 32 / 30`이다. direct validator `0.0000`과 center-gate replay `0.4211`보다는 낫지만 PCA val128 reference `0.5133`과 angle-mask production `0.5085`보다 낮으므로 broader-val512로 확장하지 않는다.
- candidate validator warm-bias smoke는 direct dense validator gate 실패가 단순 초기 bias 문제인지 확인했다. bias `2.0`과 `4.0` 모두 train8/val4 smoke에서 stop-line F1 `0.0000`, TP/FP/FN `0 / 2 / 2`라 exact로 확장하지 않는다.
- stop-line proposal competition loss는 center/selector proposal map을 GT center heatmap distribution 쪽으로 직접 경쟁시키는 opt-in loss를 시험했지만 exact val128 gate에서 실패했다. Best epoch2 objective는 `0.6051`, lane/stop/cross F1은 `0.5552 / 0.4107 / 0.5854`, stop-line TP/FP/FN은 `23 / 29 / 37`이다. Prior row-scan reference `0.6144`, `0.5522 / 0.4483 / 0.5854`보다 objective와 stop-line이 낮고 loss 계산 비용도 커서 proposal-distribution-loss-only는 0.6 path가 아니다.
- modern task-head merge replay는 retrain 없이 row-scan lane decode와 proposal-rank stop-line head를 결합하면 exact val128 objective가 `0.6196`까지 오름을 보였다. Best exact merge는 lane/stop/cross F1 `0.5522 / 0.4918 / 0.5854`, stop-line TP/FP/FN `30 / 32 / 30`이다. 그러나 broader-val512에서는 objective `0.5977`, lane/stop/cross F1 `0.5279 / 0.4079 / 0.5854`, stop-line TP/FP/FN `103 / 131 / 168`로 row-scan/PCA integration보다 낫지 않다. Crosswalk head까지 transplant한 merge도 exact crosswalk가 `0.5535`로 내려가므로 사용하지 않는다.
- stop-line candidate-assignment loss는 기존 center/selector top-k 후보를 GT endpoint segment에 직접 assign하고 true/false score, hard-negative rank, normalized geometry loss를 주는 opt-in loss로 시험했지만 exact val128에서 실패했다. Best objective는 epoch1 `0.5939`이고, epoch2 lane/stop/cross F1은 `0.5555 / 0.0571 / 0.5818`, stop-line TP/FP/FN은 `2 / 8 / 58`이다. Runtime은 `skipped_steps=0`로 깨끗했으므로 구현 실패가 아니라 valid negative evidence다. 같은 center/selector top-k 후보 위에 assignment-loss만 얹는 방식은 0.6 path가 아니다.
- endpoint-delta target/readout short run은 val128 epoch1/2 stop-line F1이 모두 `0.0000`이고 best objective도 `0.5880863169`라 기준선보다 낮았다. 새 dense geometry channel을 바로 decode에 쓰는 형태도 현재는 0.6 path가 아니다.
- heatmap-support geometry target fill은 center heatmap support 전체에 offset/angle/half-length target을 채우는 opt-in target 계약을 시험했지만 exact val128 epoch2 stop-line F1이 `0.2338`로 무너졌다. dense stop-line mask F1도 `0.4854`, center heatmap F1도 `0.0815`로 기준 `0.6118 / 0.1396`보다 낮아 같은 형태로 반복하지 않는다.
- row-center auxiliary는 row selector에 centerline-row pressure를 추가해 predicted center/proposal reliability를 올리는지 봤다. exact val128 epoch2 objective `0.6060`, lane/stop/cross F1 `0.5265 / 0.4248 / 0.5818`로 기준 exact `0.6089`, `0.5267 / 0.4483 / 0.5854`보다 낮아서 row-center-aux-only도 0.6 path가 아니다.
- read-only component/readout audit은 broader-val512 GT 271개 중 production TP `98`, anchorless component fit close `123`, anchored fit close `119`를 보였다. GT tube의 mask/center signal은 각각 `223/271`, `220/271`에서 `>=0.50`로 남아 있지만, production FN 173개 중 anchorless fit으로 새로 40px 안에 들어오는 것은 34개뿐이다. 따라서 단순 anchor swap/no-anchor PCA만으로는 0.6 path가 아니다.
- git branch history의 older stop-line oracle/repair 계열도 같이 보면, GT target mask oracle은 stop-line F1 `0.8228`까지 가능했고 GT-overlap oracle은 `component_fit_or_endpoint_error=97` 중 `50`개를 40px 안으로 복구했다. 하지만 그 뒤 non-oracle core-row trim, high-confidence cleanup/PCA endpoint trim, split fit은 각각 기준 PCA reference를 넘지 못했다. oracle headroom은 실제지만, 단순 row-band/core-row, high-confidence subcomponent cleanup, PCA endpoint quantile trim, component split fitting만으로는 회수되지 않는다.
- fit-far visual audit은 production FN, GT tube mask/center `>=0.50`, no-anchor distance `>40px` bucket 상위 18개를 렌더링했다. 18개 모두 production stop-line은 1개씩 있고, 14개는 component_count도 1이라 "아예 안 나옴"보다 single connected component 안에서 wrong line segment를 읽는 문제가 강하다.
- component-conditioned local extraction probe는 predicted component 안에서 center/selector/fused score로 local support를 골라 다시 fit했지만 exact val128 stop-line F1이 기준 `0.4483`을 넘지 못했다. best replacement는 `0.4464`, append-top2는 TP를 늘리는 대신 FP가 크게 늘어 best `0.4054`였다.
- component-split readout probe는 component 내부 multi-line 후보를 pair/Hough-like split으로 만들었지만 exact val128에서 기준을 크게 밑돌았다. best append-top2는 stop-line F1 `0.3421`, TP/FP/FN `26 / 66 / 34`로 baseline TP를 유지하는 대신 FP를 크게 늘렸고, replacement 계열은 TP를 잃어 `0.2857` 이하로 떨어졌다.
- component proposal readout probe는 predicted mask component마다 center/selector/max proposal 하나를 고르고 predicted angle + mask extent로 segment를 읽는 decode-only contract를 시험했다. exact val128 best는 selector/max component proposal stop-line F1 `0.4308`, TP/FP/FN `28 / 42 / 32`로 baseline `0.4483`, TP/FP/FN `26 / 30 / 34`보다 낮다. recall은 `+2 TP`지만 FP가 `+12`라 PCA val128 reference `0.5133`과 angle-mask production `0.5085`에 못 미쳤다.
- mask-ridge readout probe는 center/selector top-k를 쓰지 않고 predicted mask component의 distance-transform ridge에서 center/axis를 읽는 decode-only contract를 시험했다. exact val128 best `ridge_horiz_component_band3_mask080`는 stop-line F1 `0.4538`, TP/FP/FN `27 / 32 / 33`으로 baseline `0.4483`보다 `+0.0055`뿐이고 PCA val128 `0.5133`, angle-mask production `0.5085`보다 낮아 broader-val512로 확장하지 않는다.
- line-support readout probe는 top-k center/selector proposal을 predicted angle + mask extent로 segment화한 뒤 segment 위의 mask/center/selector support를 다시 점수화했다. exact val128 best는 `max_top20_line_fused_gate_m045` stop-line F1 `0.4651`, TP/FP/FN `30 / 39 / 30`으로 baseline `0.4483`, `26 / 30 / 34`보다 높지만 PCA val128 `0.5133`, angle-mask production `0.5085`, task-head merge `0.4918`보다 낮다. line-support reranking만으로는 broader-val512로 확장하지 않는다.
- candidate consensus readout은 top-k 후보끼리 가까운 segment agreement가 high-score isolated FP를 걸러내는지 봤지만 exact val128 best consensus `max_top20_consensus_d48_c3` stop-line F1은 `0.4706`, TP/FP/FN `28 / 31 / 32`다. Baseline `0.4483`보다는 높지만 `max_top10_score_s080`/angle-mask production `0.5085`, PCA val128 `0.5133`, task-head merge `0.4918`보다 낮아 broader-val512로 확장하지 않는다. Oracle-positive selection은 stop-line F1 `0.7368`이지만 production signal이 아니다.
- lane-context readout은 `row_scan_tangent` lane prediction과 stop-line 후보의 교차/거리 feature로 isolated FP를 줄일 수 있는지 봤지만 exact val128 best lane-context `max_top10_lane_cross48_c1` stop-line F1은 `0.4696`, TP/FP/FN `27 / 28 / 33`이다. Baseline `0.4483`보다는 높지만 score-threshold `0.5085`, PCA `0.5133`, task-head merge `0.4918`보다 낮아 broader-val512로 확장하지 않는다.
- crosswalk-context readout은 predicted crosswalk polygon proximity가 stop-line FP를 줄이는지 봤지만 exact val128에서 더 나빴다. `max_top10_crosswalk_context` stop-line F1은 `0.3810`, TP/FP/FN `28 / 59 / 32`이고 `max_top20_crosswalk_near48_c1`은 `0.2105`, TP/FP/FN `10 / 25 / 50`이다. Predicted roadmark context만으로는 candidate selector가 되지 않아 broader-val512로 확장하지 않는다.
- delayed candidate-validator auxiliary는 dense validator map을 바로 production gate로 쓰지 않고 center gate를 유지한 채 hard-negative aux만 학습한 뒤 validator map mixing을 replay했지만 exact val128 gate를 넘지 못했다. Training best epoch2는 lane/stop/cross F1 `0.5546 / 0.4425 / 0.5854`, stop-line TP/FP/FN `25 / 28 / 35`이고, same-checkpoint replay best `product_validator_top3_s040_mask050_band4_fallback`도 stop-line F1 `0.4793`, TP/FP/FN `29 / 32 / 31`로 angle-mask `0.5085`, PCA `0.5133`, task-head merge `0.4918`보다 낮다.
- model-side presence emit gate는 sample-level `stop_line_presence_logits`와 postprocess emission threshold를 학습/적용했지만 exact val128 stop-line F1을 `0.4112`, TP/FP/FN `22 / 25 / 38`로 낮췄다. FP는 조금 줄었지만 TP가 더 줄어 tangent-link exact `0.4483`, PCA/angle-mask references보다 낮아 broader-val512로 확장하지 않는다.
- proposal-stat emit gate는 stop-line mask/center/selector/row/x dense-map max/mean statistics를 `stop_line_presence_logits`에 더했지만 exact val128 stop-line F1이 `0.4074`, TP/FP/FN `22 / 26 / 38`로 더 낮아졌다. Dense proposal-map presence evidence도 TP를 회복하지 못하므로 broader-val512로 확장하지 않는다.
- 남은 stop-line 방향은 새 query row, endpoint-delta channel, local score window, pairwise component split, component별 proposal readout, mask-ridge readout-only, top-k line-support reranking, candidate agreement/consensus-only, lane-context readout-only, crosswalk-context readout-only, selector-map gate, row/x span proposal, simple rowx-band selector target, 단순 predicted center/selector threshold proposal, proposal-distribution KL loss-only, dense candidate-validator aux/map-mixing, sample-gate threshold/replay/rescue, presence-only/proposal-stat emit gate, mask-wide angle-field auxiliary, stopline_mask_angle_aux_weight-only가 아니다. stop-line을 계속한다면 current center/selector map을 섞는 보정이 아니라 후보 생성/readout contract 자체를 바꾸거나, lane instance stability로 돌아가 stop-line/crosswalk retention을 같이 보는 쪽으로 좁힌다.

Lane Gate 3 dense-map probe:

- command: `python3 tools/probe_pv26_lane60_dense_maps.py --checkpoint .../phase_4/checkpoints/best.pt --preset default --phase-index 4 --max-val-batches 128 --device auto`
- lane centerline core best pixel F1은 `0.5729`이고, lane support best pixel F1은 `0.7971`이다.
- 결론: support map은 이미 충분히 강하고, lane은 vectorizer만의 문제가 아니라 centerline core 품질이 아직 0.6 직전에서 막혀 있다.
- centerline-to-vector recovery audit은 같은 vectorizer에 GT centerline을 넣으면 broader-val512 epoch2 lane F1 `0.6630`까지 복구됨을 보였다. 반면 current predicted centerline은 best threshold `0.35`에서도 lane F1 `0.5169`다.
- predicted semantic attrs를 GT로 바꿔도 `pred_full`과 geometry F1은 같으므로, 현 lane 병목은 color/type attr가 아니라 predicted centerline coverage/quality다.
- core target width를 `1 -> 3`으로 넓힌 `core_centerline_refine_core_width3` short probe는 exact val128 epoch2 `phase_objective=0.6002`까지 올랐지만 lane/stop/cross F1은 `0.5232 / 0.4348 / 0.5854`다. lane F1도 기준 exact `0.5267`보다 낮아서 target-width-only widening은 0.6 path가 아니다.
- centerline error-bucket audit val512는 5092개 supervised lane 중 `recall@0.45 < 0.25`가 291개임을 보였다. 가장 약한 bucket은 `bottom_y < 0.50` miss rate `0.1329`, near-vertical `0.1290`, right-side `x >= 0.66` `0.0821`, left-side `x < 0.33` `0.0705`다.
- side-band centerline BCE positive weighting은 exact val128 lane F1을 `0.5331`로 소폭 올렸지만 phase objective `0.6083`은 기준 `0.6089`보다 낮고 stop-line F1도 `0.4348`로 후퇴했다. lane centerline-core pixel F1도 `0.5705`로 기준 `0.5729`보다 낮아서 side-BCE-only는 0.6 path가 아니다.
- side-band centerline probability margin loss는 exact val128 epoch2 objective를 `0.6097`로 아주 조금 올렸고 lane/stop/cross F1은 `0.5352 / 0.4522 / 0.5854`였다. 하지만 lane centerline-core pixel F1은 `0.5573`으로 기준 `0.5729`보다 더 낮아서 side-margin-only도 centerline 병목 해결이 아니다.
- geometry-risk recall loss는 side/truncated/near-vertical lane을 instance bucket으로 찍어 exact val128 lane F1을 `0.5405`까지 올렸지만 phase objective `0.6086`은 기준보다 낮고 stop-line F1도 `0.4348`로 후퇴했다. lane centerline-core pixel F1도 `0.5572`라 기준 `0.5729`보다 낮아서 risk-recall-only도 centerline 병목 해결이 아니다.
- geometry-risk local Tversky loss는 같은 risk instance 주변 false-positive를 같이 벌주도록 local support mask를 추가했다. exact val128 epoch2 objective는 `0.6089`, lane/stop/cross F1은 `0.5306 / 0.4522 / 0.5854`였고 lane centerline-core pixel F1은 `0.5738`이다. 기준 `0.5729` 대비 gain이 `+0.0009`뿐이고 vectorized lane F1도 recall-only보다 낮아서 broader-val512로 확장하지 않는다.
- lane negative-pixel probability margin은 explicit negative target pixels에서 centerline confidence를 낮추는 반대 압력을 시험했다. exact val128 epoch2 objective `0.6058`, lane/stop/cross F1 `0.5261 / 0.4348 / 0.5854`로 기준 미달이고, lane centerline-core pixel F1도 `0.5736`으로 기준 대비 `+0.0007`뿐이라 broader-val512로 확장하지 않는다.
- support-bridge postprocess는 support를 lane source로 대체하지 않고 centerline binary의 짧은 gap만 high-confidence support 안에서 closing하는 contract를 시험했다. 같은 decode probe baseline lane F1 `0.5222` 대비 best bridge는 `0.4758`이라 recall을 잃었고, support bridge/closing-only도 0.6 path가 아니다.
- endpoint coverage loss는 lane의 visible endpoint heatmap을 추가하고 endpoint positive에서 centerline confidence를 직접 올리는 opt-in loss를 시험했다. exact val128 epoch2 objective `0.6035`, lane/stop/cross F1 `0.5212 / 0.4348 / 0.5854`로 기준 미달이고, lane centerline-core pixel F1도 `0.5670`으로 기준 `0.5729`보다 낮아서 endpoint-only coverage도 0.6 path가 아니다.
- row-scan vectorizer는 connected-component별 vectorization 대신 row cluster track을 opt-in으로 이어서 centerline fragment continuity를 복구했다. exact val128 epoch2 objective는 `0.6144`, lane/stop/cross F1은 `0.5522 / 0.4483 / 0.5854`이고, broader-val512 objective는 `0.5981`, lane/stop/cross F1은 `0.5279 / 0.4083 / 0.5854`다. 18-sample visual comparison grid(`analysis_exports/row_scan_visual_compare_epoch2/row_scan_component_comparison_grid.png`)에서도 일부 side/fragment lane 복구는 보이지만 sample 10처럼 extra/zig track over-link risk가 남아, opt-in partial-positive이지 deployment default는 아니다.
- row-scan tangent-link vectorizer는 predicted `tangent_axis`를 row-cluster linking cost에만 쓰는 opt-in decode 축이다. exact val128 epoch2 objective는 `0.6187`, lane/stop/cross F1은 `0.5633 / 0.4483 / 0.5854`이고, broader-val512 objective는 `0.6027`, lane/stop/cross F1은 `0.5407 / 0.4083 / 0.5854`다. row-scan보다 lane continuity는 실제로 좋아졌지만 stop-line/crosswalk는 그대로라 all-task 0.6 success가 아니며, tangent-link-only cost sweep은 반복하지 않는다.
- row-scan tangent-link + stop-line `mask=0.80` integration exact replay는 objective `0.6171`, lane/stop/cross F1 `0.5633 / 0.4364 / 0.5854`로 tangent-link 단독보다 stop-line이 내려갔다. 이 결합은 broader로 확장하지 않는다.
- row-scan + stop-line PCA-threshold integration replay는 row-scan lane gain과 stop-line `mask=0.80`, `min_instance_score=0.94` override를 같이 적용했다. exact val128 lane/stop/cross F1은 `0.5522 / 0.4364 / 0.5854`, broader-val512는 `0.5279 / 0.4235 / 0.5854`이고 broader objective는 `0.6019`다. objective만 보면 0.6을 넘지만 task별 F1 목표와 PCA-only stop-line reference에는 미달이라 success/default가 아니다.
- row-scan geometry guard probe는 length/bottom/gap/dx/turn-angle guard를 exact val128 epoch2에서 비교했다. best lane F1은 `row_gap24_row_dx12`의 `0.5526`으로 기존 row-scan `0.5522` 대비 `+0.0004`뿐이고 FP가 `486 -> 512`로 늘었다. turn-angle guard는 FP를 줄였지만 TP를 더 잃어 best `0.5365`라 default 승격 path가 아니다.
- row-scan residual filter export는 broader-val512 lane TP/FP/FN `4153 / 2105 / 5324`를 남겼다. FN은 left `46.9%`, truncated bottom `<0.50` `27.9%`, aspect `>=3` `65.9%`에 몰리고, FP는 side `74.8%`와 right `40.6%` 비중이 높다.
- residual local separation loss는 left/truncated/high-aspect GT core를 올리고 주변 ring negative를 누르는 opt-in target/loss를 시험했다. exact val128 epoch2 lane F1은 `0.5476`으로 기준 `0.5267`보다 높았지만, stop-line F1은 `0.4310`으로 기준 `0.4483`보다 낮고 phase objective도 `0.6085`로 기준 `0.6089`보다 낮다. lane partial-positive일 뿐 채택/확장하지 않는다.
- BCE-focus calibration은 broader-val512 lane F1을 `0.5101 -> 0.5344`로 올렸지만 stop-line/crosswalk가 내려갔고, centerline-core pixel F1도 `0.5729 -> 0.5680`으로 낮아졌다. goal success가 아니라 lane-vectorized metric partial-positive다.
- BCE-focus + PCA stop-line decoder integration best는 broader-val512 lane/stop/cross F1 `0.5344 / 0.4583 / 0.5741`이고, stop-balance + PCA replay best도 `0.5372 / 0.4528 / 0.5812`에 그쳤다.
- lane instance evidence validator audit은 row-scan lane TP/FP를 map-local evidence로 어느 정도 분리했다. val128 held-out logistic row AUC/AP는 `0.8407 / 0.9312`지만, held-out task replay lane F1은 `0.5691 -> 0.5749`로 `+0.0058`뿐이고 TP를 `566 -> 537`로 잃었다. post-hoc instance evidence threshold는 0.6 path가 아니다.
- row-scan tangent-stability short run은 audit에서 신호가 있던 tangent alignment를 training-side로 당겨 봤지만 exact val128 gate를 통과하지 못했다. `tangent` loss를 `0.35 -> 1.0`으로 올린 best epoch2는 lane/stop/cross F1 `0.5555 / 0.4348 / 0.5854`, objective `0.6122`이고 dense lane centerline-core F1은 `0.5765`다. 기존 row-scan exact `0.5522 / 0.4483 / 0.5854`, objective `0.6144`, dense core `0.5729` 대비 lane/core gain은 작고 stop-line이 내려가므로 broader-val512로 확장하지 않는다.
- row-scan dynamic hard-negative margin loss는 current high-confidence predicted centerline 후보 중 GT support 밖 top-k만 margin 아래로 누르는 training-side FP suppression을 시험했다. exact val128 epoch2는 lane/stop/cross F1 `0.5556 / 0.4348 / 0.5854`, objective `0.6121`, lane TP/FP/FN `1104 / 480 / 1286`이다. prior row-scan exact objective `0.6144`와 stop-line `0.4483`보다 낮고 tangent-stability와 같은 stop-line regression을 보이므로 broader-val512로 확장하지 않는다.
- row-scan centerline focal short run은 centerline BCE/Dice 위에 focal 항을 추가했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6134`, lane/stop/cross F1은 `0.5551 / 0.4522 / 0.5854`, lane TP/FP/FN `1103 / 481 / 1287`, stop-line TP/FP/FN `26 / 29 / 34`다. lane/stop-line task F1은 prior row-scan보다 조금 높지만 selection objective가 `0.6144`를 넘지 못해 broader-val512로 확장하지 않는다.
- row-scan risk-bucket sampler는 residual FN에서 보인 left/truncated/high-aspect lane을 `lane_risk` task-positive bucket으로 직접 노출했다. exact val128 epoch2 objective는 `0.5957`, lane/stop/cross F1은 `0.5494 / 0.3964 / 0.5478`, lane TP/FP/FN은 `1081 / 464 / 1309`다. lane FP는 줄었지만 recall이 안 올라가고 stop-line/crosswalk retention을 잃으므로 sampler-exposure-only 축은 0.6 path가 아니다.
- row-anchor recall loss는 visible GT anchor-row x 위치 주변 centerline logit을 직접 올리는 opt-in auxiliary로 row-scan predicted centerline evidence를 안정화하려 했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6106`, lane/stop/cross F1은 `0.5512 / 0.4310 / 0.5854`, lane TP/FP/FN은 `1141 / 609 / 1249`다. Lane TP는 row-scan reference보다 늘었지만 FP가 더 커지고 stop-line이 내려가므로 broader-val512로 확장하지 않는다.
- row-anchor local contrast loss는 같은 anchor row의 GT-near band 밖 high-logit negative를 같이 누르며 positive-only FP 증가를 줄였지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6134`, lane/stop/cross F1은 `0.5559 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1133 / 553 / 1257`이다. Positive-only 대비 FP는 줄었지만 prior row-scan objective `0.6144`보다 낮고 lane gain도 exact-only small gain이라 broader-val512로 확장하지 않는다.
- inter-lane gap margin loss는 같은 row에서 인접 GT lane 사이 gap을 negative로 눌러 row-scan over-link/side FP를 줄이는 training-side auxiliary를 시험했다. exact val128 epoch2 objective는 `0.6122`, lane/stop/cross F1은 `0.5560 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1104 / 477 / 1286`, stop-line TP/FP/FN은 `25 / 30 / 35`다. Prior row-scan보다 lane F1은 조금 높지만 objective `0.6144`와 stop-line `0.4483`을 넘지 못해 broader-val512로 확장하지 않는다.
- segment continuity contrast loss는 visible GT lane anchor 사이 segment를 샘플링해 centerline positive를 당기고 양옆 normal-offset negative margin을 누르는 training-side continuity contract를 시험했다. exact val128 epoch2 objective는 `0.6127`, lane/stop/cross F1은 `0.5594 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1142 / 551 / 1248`, stop-line TP/FP/FN은 `25 / 30 / 35`다. 최근 lane-loss 후보 중 lane F1은 가장 높지만 objective `0.6144`와 stop-line `0.4483`을 넘지 못해 broader-val512로 확장하지 않는다.
- row-scan-tangent balanced-retain은 stop-line/crosswalk loss weight를 같이 올려 tangent-link lane gain을 보존하면서 다른 roadmark를 회복하려 했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6147`, lane/stop/cross F1은 `0.5607 / 0.4310 / 0.5854`, lane TP/FP/FN은 `1117 / 477 / 1273`, stop-line TP/FP/FN은 `25 / 31 / 35`다. Crosswalk task-best는 epoch1 `0.6748`이지만 최종 best objective와 stop-line은 tangent-link exact reference `0.6187`, `0.5633 / 0.4483 / 0.5854`보다 낮아 broader-val512로 확장하지 않는다.
- stop-line local-centerline selector target은 center/selector target을 GT center 주변 local centerline으로 좁혀 proposal map ranking을 개선하려 했지만 exact val128 gate를 넘지 못했다. epoch2 objective는 `0.6167`, lane/stop/cross F1은 `0.5607 / 0.4522 / 0.5854`, stop-line TP/FP/FN은 `26 / 29 / 34`다. Stop-line은 tangent-link exact `0.4483`보다 `+0.0039`뿐이고 objective/lane은 `0.6187`, `0.5633`보다 낮아 broader-val512로 확장하지 않는다.
- stop-line geometry-aware candidate validator는 current top-k 후보를 predicted endpoint geometry로 GT segment에 라벨링하는 opt-in validator loss를 시험했지만 gate를 넘지 못했다. exact val128 epoch2 objective는 `0.6182`, lane/stop/cross F1은 `0.5607 / 0.4655 / 0.5854`, stop-line TP/FP/FN은 `27 / 29 / 33`이다. Stop-line은 local-centerline selector보다 높지만 objective/lane은 tangent-link exact `0.6187`, `0.5633`보다 낮고, same-checkpoint replay에서 validator-map variants는 baseline `0.4655`보다 모두 낮아 broader-val512로 확장하지 않는다.
- stop-line mask-wide angle-field auxiliary는 existing `stop_line_angle` map을 stop-line mask support 전체에서 sign-invariant axis cosine loss로 supervised했지만 gate를 넘지 못했다. exact val128 epoch2 objective는 `0.6152`, lane/stop/cross F1은 `0.5606 / 0.4348 / 0.5854`, stop-line TP/FP/FN은 `25 / 30 / 35`이다. 이는 tangent-link `0.6187`, `0.5633 / 0.4483 / 0.5854`와 PCA/angle-mask/task-head-merge stop-line references보다 낮아 broader-val512로 확장하지 않는다.
- stop-line hard-negative sampler는 `stopline_negative` task-positive bucket으로 GT-negative lane-source samples를 직접 노출했지만 exact val128 epoch2 objective `0.6063`, lane/stop/cross F1 `0.5475 / 0.4561 / 0.5644`에 그쳤다. Stop-line은 tiny gain이지만 lane/crosswalk/objective retention을 잃으므로 broader-val512로 확장하지 않는다.
- stop-line candidate manifest failure-mode audit은 gap4/max candidate-bearing samples `369`개를 `GT-negative 165 / positive top-oracle 113 / positive misrank 29 / positive no-oracle 62`로 분해했다. Positive no-oracle bucket은 nearest candidate distance가 `40_80=28`, `gte80=34`라 selector-only로는 회복할 수 없고, midpoint proposal/candidate generation 문제가 남아 있다.
- learned stop-line fragment-to-center extent head는 high-confidence short fragment에서 full segment를 직접 복원하려 했지만 exact val128 stop-line F1이 epoch1/2 `0.0267 / 0.0519`로 무너졌다. Fragment decode를 끈 replay도 stop-line F1 `0.1233 / 0.2517`에 그쳐 기존 기준을 회복하지 못한다. Target/head wiring은 안정적이지만 production readout으로 확장할 신호가 없어 val512로 넓히지 않는다.
- stop-line center-rank margin은 GT midpoint proposal logit을 hard-negative proposal보다 높이는 opt-in loss로 no-oracle bucket을 겨냥했지만 exact val128 epoch2 objective `0.6170`, lane/stop/cross F1 `0.5605 / 0.4602 / 0.5854`에 그쳤다. Stop-line은 tangent-link 대비 `+0.0119`지만 objective/lane은 current exact references보다 낮고 geometry-validator stop-line `0.4655`도 못 넘어 broader-val512로 확장하지 않는다.
- denser stop-line candidate-select contract는 gap4/top50 후보를 validator/presence로 직접 학습하고 `max_validator` readout을 썼지만 stop-line emit이 완전히 꺼졌다. exact val128 epoch2 objective는 `0.5855`, lane/stop/cross F1은 `0.5612 / 0.0000 / 0.5854`, stop-line TP/FP/FN은 `0 / 0 / 60`이다. Same-axis threshold/longer run이 아니라 emit/select contract 재설계 없이는 반복하지 않는다.
- stop-line geometry-validator + stop-line-head-only retention schedule은 trunk, detector/TL, lane, crosswalk heads를 고정하고 stop-line head만 업데이트했지만 stop-line을 회복하지 못했다. exact val128 epoch2 objective는 `0.6180`, lane/stop/cross F1은 `0.5640 / 0.4386 / 0.5926`, stop-line TP/FP/FN은 `25 / 29 / 35`다. Geometry-validator reference `0.4655`, tangent-link reference `0.4483`보다 stop-line이 낮아 broader-val512로 확장하지 않는다.
- row-scan tangent support gate는 centerline을 support로 대체하지 않고 low-support centerline pixels만 opt-in으로 제외하는 readout을 시험했지만 exact val128에서 tangent-link와 사실상 같았다. objective는 `0.6187189441`이고 lane/stop/cross F1은 `0.5633 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1121 / 469 / 1269`다. Gain은 `+0.000002` objective 수준이라 broader-val512로 확장하지 않는다.
- row-scan tangent segment MIL은 GT lane segment 위 centerline evidence를 positive-only로 올리는 training-side instance contract를 시험했다. exact val128 epoch2 objective는 `0.6161803030`, lane/stop/cross F1은 `0.5655 / 0.4348 / 0.5854`, lane TP/FP/FN은 `1161 / 555 / 1229`다. Lane F1은 tangent-link reference보다 `+0.0022` 높지만 FP도 늘고 stop-line이 내려가 joint objective gate를 못 넘으므로 broader-val512로 확장하지 않는다.
- row-scan tangent segment MIL + lane-head-only retention schedule은 trunk, detector/TL, stop-line, crosswalk heads를 고정하고 lane head만 업데이트했다. exact val128 epoch2 objective는 `0.6193428422`, lane/stop/cross F1은 `0.5660 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1162 / 554 / 1228`이다. Exact-only best를 만들었지만 tangent-link 대비 objective gain은 `+0.000626`로 작고 stop-line 목표는 전혀 닫히지 않아 broader-val512로 확장하지 않는다.
- row-scan segment-continuity contrast + lane-head-only retention schedule은 segment-continuity의 stop/cross corruption을 freeze로 막으면 lane gain이 남는지 봤지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6157263716`, lane/stop/cross F1은 `0.5590 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1141 / 551 / 1249`이다. Segment-continuity full-head `0.5594`보다 lane이 낮고, tangent-link `0.6187`, segment-MIL lane-head-only `0.6193` objective도 못 넘어 broader-val512로 확장하지 않는다.
- row-scan tangent row-distribution loss는 같은 row의 GT core centerline pixels를 column distribution으로 정규화해 centerline logits에 직접 row-wise pressure를 줬다. Exact val128 epoch2 objective는 `0.6174723013`, lane/stop/cross F1은 `0.5659 / 0.4483 / 0.5854`, lane TP/FP/FN은 `1127 / 466 / 1263`이다. Tangent-link 대비 lane은 `+0.0026`이고 stop/cross는 보존됐지만 objective는 `0.6187`보다 낮고 segment-MIL lane-head-only exact best `0.6193`, `0.5660 / 0.4483 / 0.5926`도 넘지 못해 broader-val512로 확장하지 않는다.
- row-scan tangent segment-MIL + row-distribution 조합은 segment-level positive evidence와 row-wise distribution pressure를 같은 lane-head-only retention schedule에 같이 얹어 봤지만 exact val128 gate를 넘지 못했다. Epoch2 objective는 `0.6186643159`, lane/stop/cross F1은 `0.5621 / 0.4483 / 0.5926`, lane TP/FP/FN은 `1152 / 557 / 1238`이다. Segment-MIL lane-head-only exact best `0.6193`, lane `0.5660`보다 낮고 row-distribution-only lane `0.5659`도 못 넘어 broader-val512로 확장하지 않는다.
- lane anchor-offset instance auxiliary는 각 lane pixel이 bottom-anchor x 위치까지의 offset을 회귀하도록 새 dense target/head/loss를 붙인 training-side instance contract다. Branch `exp/lane-family-f1/lane-anchor-offset-instance-head`, commit `edbd0dc`는 wiring/tests/smoke를 통과했지만 broader-val512 2-epoch probe는 phase objective `0.5921307966`, lane/stop/cross F1 `0.5097 / 0.4025 / 0.5741`, lane TP/FP/FN `3900 / 1926 / 5577`, stop-line TP/FP/FN `98 / 118 / 173`으로 current broader best보다 낮았다. Anchor-offset auxiliary-only는 `weight/LR/epoch/freeze-policy` sweep으로 반복하지 않는다.
- lane legacy row-head fallback/union smoke는 current seg-first row-scan tangent/flip-centerline path에 legacy row-anchor output을 후보로 붙일 수 있는지 확인했다. Branch `exp/lane-family-f1/lane-legacy-row-head-union-smoke-v2`, commit `a1e0771`은 `legacy_only`, `baseline_legacy_union`, `flip_centerline_avg_legacy_union` variants와 7-test probe suite를 추가했지만 val4에서 `legacy_only` lane TP/FP/FN은 `0 / 0 / 86`이고 union variants는 baseline/flip references와 완전히 동일했다. Legacy row-head fallback/union은 threshold/top-K/dedupe sweep으로 반복하지 않는다.
- lane no-GT kNN residual-template premise는 exported repairability scorer의 top-116 선택 위에서 feature-space 이웃의 geometry residual을 전이했다. Close rows는 `8 -> 21`로 조금 올랐지만 q50/q90 distance가 `65.06 / 148.49 -> 75.17 / 212.43`으로 악화되고 best close-count variant도 worsened rows가 improved rows보다 많았다(`66 / 50`). Offline close-count premise일 뿐 live lane TP/FP/FN 증거가 아니므로 k/top-K/feature-distance/weighting sweep으로 반복하지 않는다.
- stop-line sample_id temporal-context audit은 archived exact val128 detector-context candidate rows에서 frame adjacency가 no-GT FP gate가 되는지 봤다. Gap `100`은 positive recall `0.32`, no-oracle positive recall `0.2857`로 너무 많은 TP를 버리고, gap `10000`은 positive recall `0.74`까지 오르지만 negative keep rate도 `0.7027`로 높다. Sparse validation candidate rows에서는 temporal adjacency를 frame-gap/sequence-prefix/score/smoothing sweep으로 반복하지 않는다.
- stop-line flip-consensus readout은 normal/flip decoded candidate agreement를 no-GT FP-control signal로 봤지만 exact val128에서 stop-line F1 `0.4483 -> 0.4667`, TP/FP/FN `26/30/34 -> 28/32/32`에 그쳤다. Geometry distance는 좋아졌지만 stronger stop-line references를 못 넘으므로 agreement-distance/top-k/point-average sweep으로 반복하지 않는다.
- 다음 lane 축은 row-scan/tangent-link 후처리 cost sweep, threshold integration, residual-risk local loss weight, post-hoc row-scan evidence threshold, tangent-loss-only 강화, dynamic hard-negative margin-only, centerline-focal-only, risk-bucket sampler-only, row-anchor-positive-only, row-anchor-contrast-only, inter-lane gap margin-only, segment-continuity-contrast-only, segment-continuity + lane-head-only retention, retention-balance loss-weight-only, segment-MIL-positive-only, row-distribution-only, segment-MIL + row-distribution 조합, lane-head-only retention schedule, anchor-offset auxiliary-only, legacy row-head fallback/union-only를 반복하는 방향이 아니라, current broader lane recall을 보존하는 model-side/decoder-side instance-stability contract이거나 tangent-link lane partial-positive를 stop-line/crosswalk 목표와 같이 끌어올리는 contract로 좁힌다. stop-line을 재개한다면 current center/selector top-k 후보 위 validator/assignment/candidate-select loss, gap/top-k-only 후보 pool 확장, stop-line-head-only freeze schedule, lane/crosswalk context-only filtering이 아니라 emit을 죽이지 않는 새로운 selector/readout contract로 제한한다.
- Gate 4 crosswalk postprocess probe는 exact val128에서 crosswalk F1을 `0.5854 -> 0.6027`로 올렸지만, broader-val512에서는 `0.5845`로 기준 `0.5854`보다 낮았다. crosswalk stricter component threshold는 exact-only partial-positive로 보관하고 채택하지 않는다.

## 3. Active docs surface

Current status set:

- `00A_CURRENT_STATUS.md`: 지금 어디인지.
- `00B_STATUS_HISTORY.md`: 어떻게 여기까지 왔는지.
- `00C_NEXT_GATES.md`: 다음에 할 것과 하지 말 것.

Core contract docs:

- `0_PRD.md`: 범위와 문서 맵.
- `1_DEVELOPMENT_PHILOSOPHY.md`: 작업 철학.
- `2_SYSTEM_ARCHITECTURE.md`: 현재 package/runtime 구조.
- `5_TARGETS_AND_LOSS.md`: target/loss/selection contract.
- `6_TRAINING_AND_EVALUATION.md`: stage schedule, sampler, eval 정책.
- `8_TEST_PLAN_AND_CHECKLIST.md`: 검증 기준.
- `9_EXECUTION_STATUS.md`: 기존 live tracker. 긴 구현 체크리스트 성격이라 점진적으로 status set과 분리한다.
- `11_GIT_BRANCH_WORKFLOW.md`: branch/worktree 운영.
- `17_MODAL_A100_TRAINING_RUNBOOK.md`: Modal A100 절차서.

Legacy 원문:

- 긴 run analysis와 probe 원문은 `docs/legacy/`에 보존한다.
- root docs는 현재 상태와 절차를 빠르게 찾는 표면으로 유지한다.

## 4. 현재 traffic-light 상태

Lane60 fine-tune은 traffic light를 학습한 run이 아니다.

- phase 4 lane60은 `det=0`, `tl_attr=0`이고 det/tl source samples가 없다.
- traffic light evidence는 이전 phase 3 joint run에서 봐야 한다.

현재 판단:

- traffic light box는 나오지만 class detection은 moderate 수준이다.
- matched traffic light box 안에서 attribute combo accuracy는 `~0.81-0.82`까지 나온다.
- end-to-end로 "신호등이 잘 된다"라고 말하려면 detector recall/precision이 먼저 올라야 한다.

가능한 후속 axis:

- traffic source 중심 sampler.
- `det + tl_attr`만 켜는 fine-tune.
- 필요하면 detector loss에서 traffic_light class만 강제 supervised class로 좁히는 config hook 추가.
