# Relabelling a dataset folder

A dataset folder is labelled once, in the finest vocabulary its writer knows. A trainer that wants a coarser one
(a family per name, a single class) gets it by a remap: `raidar.relabel.RelabelLabels` rewrites the folder's class
ids by name, from the original labels it keeps beside them. Runnable twin:
[`../examples/relabel.yaml`](../examples/relabel.yaml).

```bash
recordstream run examples/relabel.yaml                       # every name to its family
recordstream run examples/relabel.yaml --scope one_class     # every name to `signal`
recordstream run examples/relabel.yaml --scope original      # the folder as it was written
```

```yaml
runnable: !class:raidar.relabel.RelabelLabels
  root: $DATA_ROOT/pictures/b            # ~ and $VAR expanded
  label_map:                             # original name -> new class, many to one
    APCO25: LMR
    DMR: LMR
    "802.11ax HE": WLAN
    "802.11n HT": WLAN
  classes: {0: WLAN, 1: LMR}             # the new ids; left out, numbered as the original vocabulary reaches them
  mode: remap                            # remap | restore
```

## What a remap changes

| file | first remap | every remap |
|---|---|---|
| `labels/` (and each `<split>/labels/`) | renamed to `labels.orig/` | written anew from `labels.orig/` |
| `data.yaml` | copied to `data.orig.yaml` | `names` and `nc` replaced, `label_map` and `remapped_from: data.orig.yaml` added, every other key kept |
| `<split>/_annotations.coco.json`, `annotations/*.json` | copied to `<stem>.orig.json` | written anew: `categories` replaced, each annotation's `category_id` remapped and its original class in `original_label` |

The originals are never written again, so a second remap starts from the names the folder was written with, not
from the first remap's classes. A line before and after `label_map: {"*": signal}`:

```
labels.orig/train/000000.txt    17 0.650010 0.917355 0.002602 0.165290 6.213
labels/train/000000.txt          0 0.650010 0.917355 0.002602 0.165290 6.213
```

Only the class id changes; the rest of each line, the line's place in its file, the pictures and any other folder
(`masks/`, `meta/`) stay as they are. A file that numbers instances by line still matches.

`mode: restore` puts `labels.orig/`, `data.orig.yaml` and every `<stem>.orig.json` back in place: the folder is
then what it was before the first remap, byte for byte, with no original kept.

## Refused

Each refusal is one `raidar.relabel.RelabelError` (a `ValueError`) naming the file, the line or the setting:

| input | message |
|---|---|
| a name of the original vocabulary the map leaves out | `RelabelLabels: label_map does not say where 'DMR' go` |
| a class the map sends to that `classes` lacks | `RelabelLabels: label_map sends names to 'LMR', not in classes` |
| a label line whose id the original vocabulary lacks | `<file>:2: class 7 is not one of the original vocabulary's ids` |
| a label tree without `data.yaml` | `<folder>: a YOLO label tree but no data.yaml naming its classes` |
| a folder with no label tree and no COCO file | `<folder>: no labels/ folder, <split>/labels/ folder or COCO file to remap` |

`"*"` in `label_map` names the class every name it does not list goes to.

Never run it while a trainer reads the folder: `labels/` is replaced while the run writes it. It is built in
`labels.new/` first and the originals stay, so an interrupted run is finished by running it again.
