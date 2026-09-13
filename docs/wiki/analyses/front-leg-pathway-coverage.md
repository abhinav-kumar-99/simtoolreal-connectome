# Front-leg pathway coverage audit

The actor already contains bilateral circuitry; explicit annotation expansions add 43, 279, or 468 neurons, but do not establish complete functional pathways.

Last updated: 2026-09-13

Related: [Source extraction](../sources/summaries/malecns-front-leg-circuit.md), [Actor](../concepts/connectome-actor.md), [Adaptation](../concepts/connectome-adaptation.md)

## Evidence and reproducibility

The completed audit is `profiles/connectome/front_leg_coverage/summary.json`. It compares the pinned Pugliese extraction (4,310 neurons; 118,920 directed connections) with the official `male-cns:v1.0` annotation and synaptic-partner tables. [Official download documentation](https://male-cns.janelia.org/download/) identifies their format and provenance. The YAML pins both SHA-256 hashes and the original extraction's hashes are independently checked. Raw downloads and generated results remain ignored.

The annotation join resolves 4,309 of the 4,310 original body IDs. Body `825433`, an originally untyped sensory neuron, is absent from the public annotation table. It is retained in all candidate sets, explicitly marked unmatched, and never replaced with another ID.

The original CSV's `hemilineage` column is empty throughout. The public table supplies `trumanHl`, `somaNeuromere`, `entryNerve`, and a curated `mancBodyid`/`mancType` crosswalk. This audit uses native MaleCNS IDs and labels. No FANC or MANC IDs are treated as MaleCNS IDs, and a lineage name does not identify a particular physiologically studied subtype.

## Current graph already includes both sides

The original CSV contains 2,378 intrinsic, 1,236 descending, 328 ascending, 232 sensory, 130 motor, and six other neurons. Motor soma-side annotations divide into 67 left and 63 right cells; both sides include proximal positioning, tibia, tarsus, and substrate-grip modules.

In the original directed graph, 129 non-motor neurons directly project to motor cells of both soma-side groups. Within at most two, three, and four graph hops, respectively 2,837, 3,954, and 4,006 non-motor neurons can reach both groups. This establishes structural bilateral access, not functional coordination, axonal crossing anatomy, or target-side identity. Paths ignore weights and sign, and soma side is not necessarily projection side.

The earlier suggestion that using both legs requires doubling the present graph is superseded. The current extraction is already bilateral. The open issue is coverage of specific sensory, inhibitory and coordination pathways.

## Coverage against explicit annotation groups

All reference groups require `status: Traced`. Sensory groups require `superclass` in `vnc_sensory`/`sensory_ascending`, a listed prothoracic entry nerve (`ProLN`, `ProAN`, `DProN`, `VProN`), and the indicated native functional class. These are annotation-defined populations, not a census of all physiological proprioceptors or touch receptors. The entry-nerve list is an explicit anatomical screening choice, not inferred from graph connectivity.

| Reference group | In current graph | Reference group size | Omitted |
| --- | ---: | ---: | ---: |
| Identified front/prothoracic proprioceptive afferents | 54 | 92 | 38 |
| Front-leg motor neurons (`vnc_motor`, `fl`) | 130 | 135 | 5 |
| T1 13A lineage | 138 | 145 | 7 |
| T1 13B lineage | 106 | 150 | 44 |
| T1 09A lineage | 93 | 170 | 77 |
| T1 23B lineage | 37 | 145 | 108 |
| Front-specific descending subclass (`fl`) | 115 | 115 | 0 |
| DNg11/DNg12 type-name candidates | 47 | 48 | 1 |
| Identified front/prothoracic tactile afferents | 103 | 292 | 189 |

Lineage counts include all traced T1 cells with that label, including some motor and ascending neurons. They overlap other groups; summing omitted counts without a union is incorrect. DNg12 variants also differ in subclass, so prefix matching is a broad candidate screen, not a functional-equivalence assertion.

The 232 original sensory cells are mixed. Among their public annotations, 33 hair-plate cells enter through `PrN` (plus one neck-labelled afferent), and eight campaniform cells enter via `MesoLN` or `MetaLN`. They should not all be described as front-leg proprioceptors. Conversely, a label such as `leg bristle` does not guarantee tactile function: native `class` annotations distinguish gustatory from mechanosensory cells. Unspecified `leg` afferents are not automatically assigned a proprioceptive role.

There are 30 `vnc_sensory` chordotonal cells annotated with `ProLN`, of which 12 are retained, plus six omitted sensory-ascending chordotonal cells. Their metadata does not assign claw/hook/club tuning. These counts must not be compared to a physiological FeCO cell census as a reconstruction-completeness estimate.

## Measured candidate expansions

All sets preserve the existing 4,310 IDs. They are audit proposals only, not new actor artifacts.

| Candidate | Neurons | Added | Node increase | Reference edges at >=5 synapses | Edge increase over reference current |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current IDs | 4,310 | 0 | 0% | 119,967 | 0% |
| Add identified proprioceptors and motors | 4,353 | 43 | 1.00% | 120,096 | 0.11% |
| Also add T1 coordination-lineage and descending candidates | 4,589 | 279 | 6.47% | 125,611 | 4.70% |
| Also add identified tactile afferents | 4,778 | 468 | 10.86% | 127,432 | 6.22% |

The three expansions contain 129, 5,644, and 7,465 thresholded edges touching at least one newly added node. Some annotated additions have no surviving internal edge: 16, 20, and 43 added nodes, respectively. Merely adding their states would not restore a connected feedback pathway under these filters. They remain in the audit lists for review rather than being silently pruned.

The group completion does not substantially close the graph to all external input: the fraction of observed incoming VNC synapses retained after the edge threshold is 46.62%, 46.62%, 46.60%, and 46.83%, respectively. The denominator changes with each set and includes all observed source segments, including unannotated fragments. This is neither a task-performance metric nor evidence that the remaining half of input is required for robot control. `top_external_inputs.csv` ranks omitted partners to guide further review; strong input alone does not justify automatic inclusion.

## Reference scope and mismatch

Counts are reconstructed from individual synaptic-partner rows, with pre/post confidence >=0.5. The YAML enumerates primary-post VNC neuropils and `VNC-unspecified`, excluding brain, cervical connective and peripheral nerves. Connections are summed across batches and included ROIs before applying the >=5-synapse edge threshold. This explicit scope is not assumed identical to the upstream neuPrint ROI query, and the public snapshot may differ from the extraction.

Against the original production edge support, 118,670 edges are shared, 250 occur only in production, and 1,297 only in the reference reconstruction. Of shared edges, 118,395 have identical magnitudes and 275 differ. Hence the current-ID reference has 119,967 edges versus 118,920 in production. Candidate edge percentages above use the same reference throughout. No mismatch was silently corrected and no generated signed deployment graph was produced.

Reference magnitudes are unsigned synapse counts. Deploying any expansion would require transmitter assignments and an explicit policy for uncertain signs, normalization, population interfaces, and dynamical validation. State-size and edge-count ratios are not measured training-throughput ratios.

## Functional interpretation and next decision

The [FeCO study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/) identifies claw/hook pathways for local motor feedback. The [sensory-gating study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13070307/) identifies specific 9A neurons suppressing movement-sensitive feedback under descending control. The [grooming study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12844901/) supports roles for subsets of 13A/13B in coordination. None establishes that every T1 neuron in those lineages has the same function. The [motor-module study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11356479/) motivates selecting premotor partners by motor targets as well as lineage.

The most useful next candidate for anatomical review is the 4,589-cell set, but it is not a fully specified preserved grooming controller. Resolve the exact claw/hook and inhibitory subtypes, examine omitted partners and disconnected candidates, and use axonal target anatomy rather than soma location before choosing a deployable graph. The 4,778-cell touch extension is relevant only with an appropriate tactile interface; the existing 140-feature robot observation does not contain explicit tactile forces/contact sensors.

## Running and outputs

From the repository root:

```bash
.venv/bin/python scripts/audit_front_leg_coverage.py --config configs/connectome/audits/front_leg_coverage.yaml
```

The entrypoint uses NumPy, pandas, PyArrow, SciPy and PyYAML from the existing environment. `sources` pins public files and hashes; `download_missing` permits downloads when absent. The two required public files occupy approximately 6.8 GB combined. `current_config` locates the pinned production CSVs. `selection` owns status, nerve/class filters, lineages and descending-type matching. `connectivity` owns exact ROI labels, confidence, edge threshold, structural hop count and external-partner report length. `output_directory` owns ignored audit artifacts. All paths resolve from the repository root. A repeated run regenerates that output directory's named reports.

`summary.json` is final only when `status: complete`; it records source/config hashes, group membership counts, input fractions, candidate sizes, bilateral reachability and provenance mismatch. `candidate_neurons.csv` contains every candidate ID and native annotations, with explicit group/case flags and an unmatched-ID flag. The `*_added_body_ids.csv` files are exact additions for each case. `unmatched_current_neurons.csv` preserves the unresolved original row. `reference_incoming_edges.csv.gz`, `production_reference_comparison.csv.gz`, `current_sensory_annotation_groups.csv` and `top_external_inputs.csv` provide the supporting edge and annotation evidence.

There is no new separate helper entrypoint. Within the script, `select_groups` performs the native-annotation selection, `aggregate_batch`/`read_reference_edges` stream and aggregate partner rows, `case_metrics` measures retained input and graph size, and `reachability` measures directed paths. It reuses hash/download/CSV-reading helpers from `simtoolreal_shared/connectome_data.py`; that module's behavior is unchanged.

Validation: full public-table audit completed; three focused tests cover ROI/confidence filtering, cross-batch aggregation before threshold, omitted-input denominators, edge direction/hop count, and avoiding gustatory/other-leg/type-prefix false matches. The audit does not launch training or alter the actor.
