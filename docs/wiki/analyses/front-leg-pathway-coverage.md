# Front-leg pathway coverage audit

The actor already contains bilateral circuitry; explicit annotation expansions add 43, 279, or 468 neurons, but do not establish complete functional pathways.

Last updated: 2026-09-13

Related: [Source extraction](../sources/summaries/malecns-front-leg-circuit.md), [Actor](../concepts/connectome-actor.md), [Adaptation](../concepts/connectome-adaptation.md)

## Evidence and reproducibility

The completed audit is `profiles/connectome/front_leg_coverage/summary.json`. It compares the pinned Pugliese extraction (4,310 neurons; 118,920 directed connections) with the official `male-cns:v1.0` annotation and synaptic-partner tables. [Official download documentation](https://male-cns.janelia.org/download/) identifies their format and provenance. The YAML pins both SHA-256 hashes and the original extraction's hashes are independently checked. Raw downloads and generated results remain ignored.

The annotation join resolves 4,309 of the 4,310 original body IDs. Body `825433`, an originally untyped sensory neuron, is absent from the public annotation table. It is retained in all candidate sets, explicitly marked unmatched, and never replaced with another ID.

The original CSV's `hemilineage` column is empty throughout. The public table supplies `trumanHl`, `somaNeuromere`, `entryNerve`, and a curated `mancBodyid`/`mancType` crosswalk. This audit uses native MaleCNS IDs and labels. No FANC or MANC IDs are treated as MaleCNS IDs, and a lineage name does not identify a particular physiologically studied subtype.

## Current graph already includes both sides

An additional participation check on the actual production `biological.npz` (not the public reference reconstruction) uses its 1,468 sensory/descending input indices and 130 motor readout indices. Directed reachability finds 4,281 neurons reachable from those inputs, 4,162 able to reach a motor readout neuron, and 4,133 on an input-to-motor route. There are 10 completely isolated production nodes. The 177 nodes outside an input-to-motor route comprise 29 not reachable from robot inputs and 148 unable to reach the motor readout. Reachability includes zero-hop membership in the input/output sets and is measured to a fixed point; positive gain adaptation does not change this support.

All neurons are still evaluated by the recurrent update. With zero initial state and frozen zero recurrent biases, nodes unreachable from input remain at zero. Input-driven nodes without a motor route can be active and affect the actor-side value head, which reads all states, but cannot directly influence the action mean through the recurrent graph. Structural routes establish possible signal propagation, not measured functional importance or nonzero gradients. A neuron lacking some external biological inputs may still compute from its retained inputs; omission is not equivalent to isolation. The 43 isolated added candidates are measured in the separate reference expansion, so those counts should not be conflated with the 10 production isolates.

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

The five-synapse cutoff originates in the [Pugliese extraction methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/), which remove weak cell-pair connections in MANC/mCNS. It is a count of anatomical synaptic contacts from a particular source neuron to a particular destination, not a firing threshold, number of input neurons, or normalized learned weight. The actual production NPZ has minimum absolute raw value 5, 17,791 edges exactly at 5, and no edges below 5. Our audit explicitly carries that choice through `minimum_synapses: 5`; preparation of the original artifact consumes an already-filtered upstream CSV. The same paper states that its FANC and BANC simulations include all detected synapses without this cutoff. Five is therefore a dataset/model selection choice, not a universal biological rule. Counts of isolated candidate neurons are conditional on it; a threshold-one comparison on the same fixed neuron set would separate weak-edge restoration from expanding the node boundary.

Counts are reconstructed from individual synaptic-partner rows, with pre/post confidence >=0.5. The YAML enumerates primary-post VNC neuropils and `VNC-unspecified`, excluding brain, cervical connective and peripheral nerves. Connections are summed across batches and included ROIs before applying the >=5-synapse edge threshold. This explicit scope is not assumed identical to the upstream neuPrint ROI query, and the public snapshot may differ from the extraction.

Against the original production edge support, 118,670 edges are shared, 250 occur only in production, and 1,297 only in the reference reconstruction. Of shared edges, 118,395 have identical magnitudes and 275 differ. Hence the current-ID reference has 119,967 edges versus 118,920 in production. Candidate edge percentages above use the same reference throughout. No mismatch was silently corrected and no generated signed deployment graph was produced.

Reference magnitudes are unsigned synapse counts. Deploying any expansion would require transmitter assignments and an explicit policy for uncertain signs, normalization, population interfaces, and dynamical validation. State-size and edge-count ratios are not measured training-throughput ratios.

## Functional interpretation and next decision

The 5,644 new-endpoint edges in the coordination candidate divide into 2,722 added-to-current edges (35,092 synaptic contacts), 2,282 current-to-added edges (35,888 contacts), and 640 added-to-added edges (7,785 contacts). These are directed cell pairs above threshold, not 5,644 individual synapses. Their importance depends on pathway placement, not just total count. This partition is derived from `reference_incoming_edges.csv.gz` joined to `candidate_neurons.csv` using `case_coordination_candidates` and `in_current`.

Among newly added 09A-lineage cells, 17 edges (234 synaptic contacts) target the annotation-defined front proprioceptor group. This provides concrete anatomical candidates for feedback regulation; it does not establish that these are the chief-9A or hook-specific cells in the physiological literature. Strong omitted inputs to the existing network include `IN09A014` body 800086 and `IN23B009` body 800537. The latter contributes 1,345 contacts over 58 directed pairs to current cells. Neither a large contact count nor lineage membership proves a particular control computation.

The five omitted front-motor annotations are an accessory tibia flexor (1050156705), two tarsus depressors (1050380236, 1050815462), a trochanter flexor (1051367796), and one untyped motor cell (1052407897). The single omitted DNg11/DNg12-name candidate is DNg12_a body 23628, whose subclass is `nt`, not `fl`; it must not be described as a verified missing front-grooming command neuron. The broader touch candidate adds 189 tactile afferents. [Elabbady et al.](https://www.sciencedirect.com/science/article/pii/S0960982226003659) connect 23B populations with spatial touch maps and targeted grooming in FANC, motivating a subtype-matching investigation rather than assigning that function to every omitted MaleCNS 23B cell.

The [FeCO study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/) identifies claw/hook pathways for local motor feedback. The [sensory-gating study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13070307/) identifies specific 9A neurons suppressing movement-sensitive feedback under descending control. The [grooming study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12844901/) supports roles for subsets of 13A/13B in coordination. None establishes that every T1 neuron in those lineages has the same function. The [motor-module study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11356479/) motivates selecting premotor partners by motor targets as well as lineage.

The most useful next candidate for anatomical review is the 4,589-cell set, but it is not a fully specified preserved grooming controller. Resolve the exact claw/hook and inhibitory subtypes, examine omitted partners and disconnected candidates, and use axonal target anatomy rather than soma location before choosing a deployable graph. The 4,778-cell touch extension is relevant only with an appropriate tactile interface; the existing 140-feature robot observation does not contain explicit tactile forces/contact sensors.

## Compact-circuit sizing hypothesis

The user prefers the 4,778-cell candidate as the broader reference and asked about a smaller circuit for fine motion. No behavioral minimum has been measured. Removing the requirement to retain all 4,310 original cells, the union of the selected motor (135), proprioceptor (92), four T1 lineage (610), and descending groups (157 unique cells across front-specific and DNg11/DNg12-name groups) contains **993 unique neurons**. Adding 292 tactile afferents gives **1,285 unique neurons**. Groups overlap, so raw population counts do not sum to these union sizes.

Within the same public reference at >=5 synapses, that 1,285-cell induced graph has 7,377 directed connections and 114 completely isolated nodes. It is therefore a seed inventory, not a demonstrated compact controller: intermediate partners outside the named lineages are potentially important. A roughly 2,000-cell bilateral candidate, explored over approximately 1,500–2,500 cells, is an engineering hypothesis for adding targeted connecting partners and pruning redundant cells; it is not a measured biological lower bound or a demonstrated sufficient size.

The [Pugliese preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/) describes an 803-neuron FANC front-left motor/local-premotor/DNg100 simulation and a much smaller isolated rhythm-generating motif. Those findings concern motor rhythms, not dexterous arm/hand transfer, and do not establish a minimum for our task. Reduction acceptance should measure fine tracking, perturbation response, coordinated multi-joint/contact behavior, and anatomical input-to-output paths; locomotor oscillation or reward alone is insufficient evidence of preserving the intended computations. No compact extraction or training experiment was launched for this question.

## Evidence-led additions, not a 2,000-cell quota

The user explicitly rejected filling a neuron budget. The approximately 2,000 target above is **superseded as a selection objective**: neuron count must follow justified pathway inclusion. No compact training has been launched and no compact actor artifact has been deployed.

A preliminary structural screen starts with the 1,285 seeds and the 4,778-cell public-reference universe, retaining directed pairs with >=5 contacts. With IDs sorted, build a source-row/destination-column CSR graph; run SciPy `csgraph.dijkstra` with `unweighted=True`, `min_only=True`, and `return_predecessors=True`, first from all proprioceptor/tactile/selected-descending inputs, then from front motors on the transposed graph. Trace the predecessor chain of every reachable seed in both traversals. Union those paths with all seeds. This yields 1,596 cells, 18,663 induced edges and 44 isolated cells. It is one shortest-path footprint, not a minimum or sufficient circuit; equal-hop alternatives and inhibitory feedback need separate consideration.

Screening cells **outside that 1,596 set** gives the following independently counted anatomical motifs. Every link obeys the same >=5-contact rule; membership does not imply physiological necessity.

| Motif | Additional cells | Of these, VNC intrinsic neurons |
| --- | ---: | ---: |
| Front proprioceptor -> cell -> front motor | 276 | 262 |
| Front tactile afferent -> cell -> front motor | 119 | 111 |
| Union of those two sensory-to-motor motifs | 356 | 338 |
| Selected descending neuron -> cell -> front motor | 672 | 600 |

The proprioceptive and tactile sets overlap by 39 cells. The other 18 cells in their union are 15 ascending, two descending and one sensory neuron; do not describe all 356 as local interneurons. The 276 proprioceptive candidates make 1,682 directed connections totaling 49,702 contacts onto 112 front motors: 19.32% of all >=5-contact input onto the 135 front motors **from within the 4,778 reference**, not of whole-CNS motor input. This is synapse mass, not percent behavioral contribution. Adding the full sensory-motif union would produce 1,952 cells and 33,720 edges, still with 44 isolated cells; this is an illustrative set calculation, not a chosen profile. The larger descending set is a review pool, not an automatic inclusion list.

Concrete omitted examples, already present in the original 4,310-cell inventory:

- `IN08A006`, bodies 800561 (left T1) and 801437 (right T1), has public consensus GABA labels. The left cell receives 253 contacts from front proprioceptors and 182 from the selected descending group, and sends 698 contacts to seven front motors. The right receives 87 and 155 respectively, and sends 802 contacts to eight front motors. These are explicit command/sensory-convergence and premotor candidates, not generic high-degree padding.
- `IN19A076`, body 817443 (left T1), is GABA-labelled, receives 130 contacts from front proprioceptors and 72 from selected descending cells, and sends 242 contacts back to nine front proprioceptors. It has no retained direct front-motor output. It is a candidate sensory-feedback regulator, not a physiologically identified gating cell. Nine excluded GABA-labelled cells have output to front proprioceptors plus input from proprioceptors or selected descending cells.

These observations justify investigating hundreds of additional cells, but not claiming that exactly 400 cells are necessary, that every candidate should be kept, or that the unvalidated 1,285 seeds are irreducible. Shortest paths alone preserve reachability while potentially losing parallel computations and feedback. The [FeCO circuit study](https://www.nature.com/articles/s41467-025-59302-3) supports inspecting local sensory-motor feedback; the [motor-module study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11356479/) supports inspecting structured premotor input rather than treating all alternate routes as redundant. Neither establishes the roles of these exact MaleCNS body IDs in robot control. Tactile motifs also remain conditional on the robot interface, which currently lacks explicit tactile channels.

Evidence: `profiles/connectome/front_leg_coverage/candidate_neurons.csv` (SHA256 `4486dd46e3bcee4b3f87e10e0716b6232d44d95fb887a4a4169490fee99ee862`), `reference_incoming_edges.csv.gz` (`d2e59372049b061c11450e013d9424d15ff369179839ab657ed6f0144de1aa3a`), and public `data/connectomes/raw/malecns_v1_audit/neurotransmitters.feather` (`95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621`). Group masks and edge provenance are owned by `configs/connectome/audits/front_leg_coverage.yaml` and `scripts/audit_front_leg_coverage.py`. Counts were recomputed from these files, not inferred from papers.

## Unrestricted upstream completion boundary

A follow-up request to add everything missing was checked before changing the actor. Starting from the 4,778-cell coordination-and-touch candidate, recursively add every presynaptic partner among public `status: Traced` neurons plus the original IDs. Use the same explicit VNC primary-post ROI list and confidence >=0.5 as the audit, retaining all positive connection counts (minimum one synaptic contact, rather than the candidate table's five). This is a structural incoming-closure test, not a functional pathway identification.

Streaming the pinned public partner file yields 3,869,530 connections whose endpoints are both in that allowed universe. Successive upstream expansions contain 22,897, 23,960, 23,970 and 23,971 cells; another iteration adds none. The resulting induced graph has **23,971 neurons and 3,864,279 directed cell-pair connections**. This is about 5.56 times the production state size and 32.5 times its edge count, with the edge comparison also reflecting the lower threshold. These counts are not the five-synapse counts reported above.

The annotated closure contains 13,143 VNC intrinsic neurons, 6,324 VNC sensory neurons, 1,831 ascending neurons, 1,305 descending neurons, 686 VNC motor neurons, and other classes. It therefore extends well beyond the two front legs. A zero omitted-input boundary within this traced-neuron/ROI/confidence universe requires a substantially broader controller than the discussed front-leg candidate; it still excludes brain synapses, untraced fragments, and unobserved physiology.

The public neurotransmitter table (`body-neurotransmitters-male-cns-v1.0.feather`, SHA-256 `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621`) matches all but five closure IDs. Consensus labels include 1,253 `unclear`, 50 octopamine, 20 serotonin and six histamine cells, alongside acetylcholine, GABA and glutamate. The original blanket non-cholinergic inhibitory convention would be a consequential modelling assumption at this scale, not a validated representation of all those effects.

No expansion was deployed or active profile changed. The choice between VNC-wide incoming closure and a bounded front-leg circuit requires an explicit scope decision; a complete biological front-leg controller has not been established by these graph operations.

## Running and outputs

From the repository root:

```bash
.venv/bin/python scripts/audit_front_leg_coverage.py --config configs/connectome/audits/front_leg_coverage.yaml
```

The entrypoint uses NumPy, pandas, PyArrow, SciPy and PyYAML from the existing environment. `sources` pins public files and hashes; `download_missing` permits downloads when absent. The two required public files occupy approximately 6.8 GB combined. `current_config` locates the pinned production CSVs. `selection` owns status, nerve/class filters, lineages and descending-type matching. `connectivity` owns exact ROI labels, confidence, edge threshold, structural hop count and external-partner report length. `output_directory` owns ignored audit artifacts. All paths resolve from the repository root. A repeated run regenerates that output directory's named reports.

`summary.json` is final only when `status: complete`; it records source/config hashes, group membership counts, input fractions, candidate sizes, bilateral reachability and provenance mismatch. `candidate_neurons.csv` contains every candidate ID and native annotations, with explicit group/case flags and an unmatched-ID flag. The `*_added_body_ids.csv` files are exact additions for each case. `unmatched_current_neurons.csv` preserves the unresolved original row. `reference_incoming_edges.csv.gz`, `production_reference_comparison.csv.gz`, `current_sensory_annotation_groups.csv` and `top_external_inputs.csv` provide the supporting edge and annotation evidence.

There is no new separate helper entrypoint. Within the script, `select_groups` performs the native-annotation selection, `aggregate_batch`/`read_reference_edges` stream and aggregate partner rows, `case_metrics` measures retained input and graph size, and `reachability` measures directed paths. It reuses hash/download/CSV-reading helpers from `simtoolreal_shared/connectome_data.py`; that module's behavior is unchanged.

Validation: full public-table audit completed; three focused tests cover ROI/confidence filtering, cross-batch aggregation before threshold, omitted-input denominators, edge direction/hop count, and avoiding gustatory/other-leg/type-prefix false matches. The audit does not launch training or alter the actor.
