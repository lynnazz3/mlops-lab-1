# Lab 3 Answers: Containerizing the Model with Docker

## Q1: Model version and run artifact vs. registered model

My best run from Lab 2 (`efficient-ox-684`, the lr=0.0001/batch=32 run) ended up with four versions registered under the name `food11` by the time I finished this lab. Versions 1 and 2 came from that same run because I accidentally ran the registration command twice, which was harmless but left some clutter in the registry. Versions 3 and 4 came later while I was working through the Docker networking issue described in Q7.

The concept itself: a run's logged model artifact is just files stored inside that specific run's folder, permanently tied to it along with its params and metrics. The Model Registry is a separate layer on top of that. Calling `register_model(...)` takes a snapshot of the model and gives it a stable name (`food11`) and a version number that no longer depends on which run produced it. That's really the point: my serving code doesn't need to know or care which run the model came from, it just requests `models:/food11@champion` and mlflow resolves whatever version that currently is.

## Q2: Aliases vs. deprecated stages

The old approach used fixed stage names built into mlflow itself (`Staging`, `Production`, `Archived`), with only one version allowed per stage at a time. Aliases replaced that with something more flexible: you define your own names (`champion`, `challenger`, etc.) and point them at any version you want, and multiple aliases can point at different versions at once.

The reason to version a model separately from its run is that a run captures how the model was trained, but most of the time you just want to know which model is currently live without needing to track its training history. Aliases are more flexible than fixed stages because reassigning `champion` to a new version is a single API call, with no redeployment and no code changes in `serve.py`. I relied on this a lot by the end of the lab, since I moved `champion` from version 2 to 3 to 4 while debugging, and `serve.py` never needed to change.

## Q3: Why load via mlflow URI instead of a raw `.pth` path

If `serve.py` pointed directly at a `.pth` file, I would have to hardcode a path and remember to update it manually every time I trained a better model, with no record of which version was actually being served. Loading through `models:/food11@champion` means the serving code doesn't need to know which file or run it is pulling from, it just resolves to whatever `champion` currently points at.

To serve a newer model version, I would not need to touch `serve.py` at all, just reassign the alias (`client.set_registered_model_alias('food11', 'champion', new_version)`) and restart the container. The pyfunc loader resolves the alias fresh every time the container starts.

## Q4: Layer caching and the two build errors I hit

Copying `pyproject.toml`/`uv.lock` and running `uv sync` before copying `src/` is about taking advantage of Docker's build cache. Docker caches each instruction as its own layer, and if the inputs to that instruction have not changed, it reuses the cached layer instead of redoing the work. My dependency list (torch, mlflow, fastapi, etc.) rarely changes, but I am constantly editing `serve.py`, so I want the slow part (installing torch, which alone took around 980 seconds) cached separately from the fast part (copying a few KB of source code).

If I had copied everything at once, any small edit to `serve.py` would invalidate the cache for the whole dependency install step too, since Docker caches layers in order and a change anywhere in a `COPY` breaks the cache for every instruction after it. With the split I actually used, changing a line in `serve.py` only reruns the small `COPY src/` step and a quick `uv sync` for my own package, so the rebuild takes seconds instead of around 15 minutes.

I hit two real errors getting this split to work correctly, not just in theory:
* First attempt: `uv sync --frozen --no-dev` failed because `src/` had not been copied yet, but uv still tried to build my own `food11` package (not just install dependencies) and failed looking for `src/food11/__init__.py`. Fixed by splitting into two `uv sync` calls: one with `--no-install-project` right after copying just `pyproject.toml`/`uv.lock` (dependencies only, fully cacheable), then copying `src/`, then a second `uv sync` to build my project itself.
* Second error: `uv sync` failed because `pyproject.toml` declares `readme = "README.md"`, but I was only copying `pyproject.toml` and `uv.lock` into the image, so README.md was missing. Fixed by adding it to that same `COPY` line.

## Q5: Image size

The final image comes out to roughly 1.7GB total. Running `docker history food11-api:latest` shows the `COPY /app/.venv /app/.venv` layer alone accounts for 1.52GB, which is nearly the entire image. That is torch and torchvision (even the CPU only builds are large) along with mlflow, fastapi, and the rest of the dependencies in the venv. My actual application code, by comparison, is the `COPY src/` layer at only 57.3kB, essentially negligible next to the dependencies.

I did not build a naive single stage version to compare directly, but the multi stage setup is still clearly doing its job: none of `uv` itself, pip's download cache, or any other builder only tooling shows up in the final image, only the finished `.venv` and my source code made it into the runtime stage. A single stage build would have carried all of that extra build time weight into the final image on top of the 1.7GB it already sits at.

## Q6: `.dockerignore` reasoning

Without it, `docker build` sends the entire build context (including `.venv/`, `data/`, which is the full Food 11 dataset, `mlruns/`, all of `.git/` history, and `__pycache__/` files) to the Docker daemon before the build even starts. That slows the build down significantly, and there is a real risk that any of that content accidentally gets pulled into a `COPY` and bloats an intermediate layer.

The folders that would actually break the build, not just slow it down: `.venv/` would conflict with the fresh venv `uv sync` creates inside the image, and could bring in Windows specific binaries that are incompatible with a Linux container. `data/` would not break the build, but at multiple GB of images it would make even starting a build extremely slow. `.git/` and `mlflow.db` are not needed for the app to run and just add unnecessary weight.

## Q7: `host.docker.internal` and the artifact serving gotcha

The basic networking answer: a container has its own isolated network namespace, so `127.0.0.1` inside the container refers to the container itself, not my host machine. My mlflow server runs on the host and listens on the host's loopback address, which is a completely different and unreachable address from inside the container. `host.docker.internal` is a special DNS name that Docker Desktop provides specifically to resolve to the host machine from inside a container, which is how Windows and Mac get around that isolation.

Getting the networking right (switching to `host.docker.internal`) turned out not to be enough on its own, and this ended up being the most involved debugging of the whole lab. My first container run still failed with `No such artifact: ''`, even though the container could reach the mlflow server fine over HTTP. It turns out that "reaching the host" for metadata (run params, metrics, registry lookups) and "reaching the host" for the actual artifact bytes (the model weights themselves) are two separate problems. By default, mlflow's tracking server only serves metadata over HTTP, and artifact access still assumes direct filesystem access to `./mlruns`, which the container does not have.

The real fix required two parts, and I only got the first one right initially. I restarted the server with `--serve-artifacts`, which is meant to turn the tracking server into a proxy for artifact requests as well. That alone still did not work, because I had also left `--default-artifact-root ./mlruns` on the server command, and that flag forces a plain local filesystem path for any experiment's artifact location, which overrides the proxy. The correct flag for proxied storage is `--artifacts-destination`, not `--default-artifact-root`. On top of that, an experiment's artifact location is fixed permanently the moment the experiment is created, so even after fixing the flags, my existing `food11` experiment (from Lab 2) remained stuck pointing at a local path. I had to create a new experiment (`food11-serving`) after fixing the flags, re-log the model into that fresh experiment, and only then did I get an artifact URI starting with `mlflow-artifacts:/` instead of a local path. That change is what finally let the container pull the model successfully.

So the short version of Q7: the networking configuration gets you to the host, but reaching the host for an mlflow backed service also involves a separate artifact storage layer that requires its own explicit proxying, and that setting is locked in per experiment at creation time rather than something you can enable retroactively.

## Q8: Restart persistence

I stopped the running container (`docker stop <id>`) and started a new one from the exact same image, with no rebuild. It came back up correctly, reloaded the model successfully, and returned the same prediction on my test image both before and after (`{"category":"Bread","confidence":0.3215}`, identical in both cases).

This confirms that the model weights are not baked into the image at all, only my code and dependencies are. Every time a container starts, it fetches the model fresh from the mlflow server at that moment, through the `champion` alias. This is a genuine advantage of the setup: I could point `champion` at a completely different model version and just restart the container, with no rebuild, no image changes, and no edits to `serve.py`. The image itself is static, while the model is live runtime state.

## Q9: What's still missing for another machine to run this

Right now `food11-api:latest` exists only in my local Docker image cache on this one Windows machine. No other machine, including a CI runner or a Kubernetes cluster, could pull it, since it has never been pushed anywhere. The `Dockerfile` is tracked in git, so anyone could technically rebuild an equivalent image from source, but that is not the same as being able to pull the exact, already built and tested image.

This mirrors the same split from Lab 1's git and dvc setup: git tracks the pointer (the recipe, in this case the Dockerfile), while the actual large binary content lives elsewhere (DagsHub for the dataset, and here it would need to be a container registry such as Docker Hub or GHCR for the image itself). To resolve this, I would need to tag the image properly (for example, `docker tag food11-api:latest ghcr.io/lynnazz3/food11-api:latest`), push it to a registry, and then any other machine could pull that exact image directly instead of rebuilding from source and hoping the result matches.
