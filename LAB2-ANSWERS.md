# Lab 2 — Answers

## Question 1: What changed in `pyproject.toml` and `uv.lock`?

`pyproject.toml`'s `dependencies` list gained four new entries: `mlflow`, `torch`,
`torchvision`, and `scikit-learn`.
It also gained a `[[tool.uv.index]]` block pointing at PyTorch's CPU-only wheel index,
plus a `[tool.uv.sources]` block telling uv to pull `torch` and `torchvision`
specifically from that index instead of the default PyPI index.

`uv.lock` changed a lot ,it recorded the exact resolved version of
every package in the full dependency tree, not just the four direct additions. This
includes dozens of transitive dependencies pulled in by mlflow and torch (numpy,
protobuf, sqlalchemy, matplotlib, pandas, etc.), each pinned to an exact version with a
hash. `pyproject.toml`, `uv.lock` records... which is what makes the environment fully reproducible on another machine
via `uv sync`.


## Question 2: What is `--backend-store-uri` for? What is `--default-artifact-root` for? What's the difference between metadata and artifacts?

- `--backend-store-uri` tells MLflow where to store metadata: run parameters, metrics, tags, timestamps, and run status. This is
  small, structured, queryable data, so it goes into a database SQLite here.
- `--default-artifact-root` tells MLflow where to store artifacts: actual files a run produces, most notably the trained model itself.
 
 metadata is really what about the actual run (what we search, sort, or filter by in the UI)
artifacts are files produced by the run
(what you'd actually download and use afterward, like the saved model).


## Question 3: Why shouldn't `mlflow.db` and `mlruns/` be tracked by git, and why not by dvc either?

`mlflow.db` is a binary SQLite file that changes on every training run.
Git would store a new copy of the whole file on every commit, bloating repo history
fast, and a binary database file can't be meaningfully diffed or merged. `mlruns/`
contains potentially large model artifact files the exact kind of content git handles
badly, same reasoning as raw images in Lab 1.

 DVC is meant for data that represents a meaningful, reproducible
snapshot tied to a specific commit something a teammate would deliberately want to
pull down. Local mlflow output is different: it's ephemeral, machine-local experiment
bookkeeping from hyperparameter tuning. If a teammate cloned this repo, they wouldn't
want *my* specific pile of tuning runs ,they'd run their own experiments and generate
their own. When a specific trained model is actually worth keeping as a deliverable,
that gets handled deliberately later (e.g. via MLflow's model registry or by explicitly
exporting it into DVC) , not by versioning the entire raw experiment log by default.

---

## Question 4: What happens the first time you call `set_experiment` with a name that doesn't exist yet?

MLflow automatically creates the experiment no manual setup in the UI is needed
first. Confirmed live in my terminal output on the first training run:

```
mlflow.tracking.fluent: Experiment with name 'food11' does not exist. Creating a new experiment.
```

Before that first run, only the "Default" experiment existed in the UI. After running
`train.py` once, a new "food11" experiment appeared automatically in the sidebar.

---

## Question 5: What is the difference between `mlflow.log_param` and `mlflow.log_metric`? Why does `log_metric` take a `step` argument and `log_param` doesn't?

A **param** is a value fixed before training starts and never changes during the run —
learning rate, batch size, number of epochs, model architecture. A **metric** is a
value that evolves *during* training ,loss and accuracy are different at epoch 1 than
at epoch 5.

`log_metric` takes a `step` argument so MLflow can plot the metric's evolution as a
line chart (value vs. step/epoch) in the UI. A param has no "evolution" over the course
of a run it's a single fixed value for the whole run so a step value wouldn't mean
anything there.

---

## Question 6: Find the params, metric charts, and logged model artifact in the UI. Where does the model artifact actually live on disk?

In the MLflow UI, opening a run shows:
- A **Parameters** section listing `dataset`, `epochs`, `lr`, `batch_size`, `model`
- A **Metrics** section clicking any metric name (e.g. `val_accuracy`) shows it
  plotted as a line chart across steps
- An **Artifacts** section with a file browser showing the `model/` folder

On disk, the model lives at:

```
./mlruns/<experiment-id>/<run-id>/artifacts/model/
```

Confirmed by running `Get-ChildItem -Recurse mlruns | Where-Object { $_.Name -eq "model" }`
locally — the UI's artifact browser is just a view onto this same folder, which is the
`--default-artifact-root` location set when the server started.

---

## Question 7: Which learning rate gave the best `val_accuracy`? Is higher always better?

Four runs, all `dataset=mini`, `epochs=5`:

| Run name | lr | batch_size | best val_accuracy | test_accuracy |
|---|---|---|---|---|
| efficient-ox-684 | 0.0001 | 32 | 0.7117 | 0.7281 |
| skittish-wasp-723 | 0.001 | 64 | 0.5894 | 0.5557 |
| burly-whale-555 | 0.001 | 32 | 0.5438 | 0.5000 |
| exultant-bee-915 | 0.01 | 32 | 0.1788 | 0.1907 |

**The lowest learning rate tested, `lr=0.0001`, gave the best result** (72.8% test
accuracy), and it wasn't close — the next best run reached only 55.6%.

**Higher is definitely not always better.** `lr=0.01` produced the worst run by a wide
margin (19% test accuracy, barely above random guessing for 11 classes). Its
`val_loss` spiked to 30.97 in the very first epoch , a sign the learning rate was large
enough to cause unstable, overshooting weight updates right from the start, and the
model never recovered a useful state in the remaining epochs. Meanwhile, `lr=0.0001`
trained smoothly and steadily (`train_loss` fell from 1.74 to 0.033), but by epoch 5 its
`val_accuracy` had started slightly dipping (0.7117 → 0.7099 → 0.7053) while
`train_loss` kept dropping ,an early sign of overfitting that would likely get worse
with more epochs. So even the best-performing rate here isn't "free" , it's trading a
slower, more stable convergence for an overfitting risk if training continued longer.

---

## Question 8: Parallel coordinates plot — pattern across `lr`, `batch_size`, `val_accuracy`

Viewed on the Compare page's parallel coordinates plot with `lr`, `batch_size`, and
`val_accuracy` as the three axes:

**Learning rate dominates the outcome far more than batch size does.** Holding
`batch_size=32` fixed and only changing `lr` (comparing `efficient-ox-684` vs
`burly-whale-555` vs `exultant-bee-915`) swings `val_accuracy` from 0.71 down to 0.18 —
a huge range. By contrast, holding `lr=0.001` fixed and changing only `batch_size` (32
vs 64, comparing `burly-whale-555` vs `skittish-wasp-723`) moves `val_accuracy` from
0.54 to 0.59 ,a much smaller effect. The clear pattern: in the ranges tested here,
`lr` is the hyperparameter that matters most, and smaller values (within this range)
correlate with better results, while `batch_size` has a comparatively minor influence.

---

## Question 9: Best run by `val_accuracy` — note its run ID

Sorting the runs table by `val_accuracy` descending, the top run is:

**Run name:** efficient-ox-684
**Run ID:** `a3224592877f40f9b6ff73eb857700f2`
**Params:** dataset=mini, epochs=5, lr=0.0001, batch_size=32, model=resnet18
**Best val_accuracy:** 0.7117 (epoch 3)
**Final test_accuracy:** 0.7281

