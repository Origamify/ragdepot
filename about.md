# ragdepot

A small standalone RAG library: chunk text, embed it locally, and search it
hybrid (dense vectors + BM25 keywords) over per-scope sqlite indexes.
Usable from Python or the CLI.

## Usage

```python
from ragdepot import Depot

depot = Depot(root="/path/to/depot-root")
depot.upsert_file(scope="mywiki", path="notes.md")
hits = depot.query("how to store solar energy", scope="mywiki", k=8)
```

CLI (operators only):

```bash
./run.sh --root /path/to/depot-root add --scope global file.md ...
./run.sh --root /path/to/depot-root query --scope mywiki "solar energy storage"
./run.sh --root /path/to/depot-root stats
```

`./test.sh` runs the tests. Flags and output formats: see `interface.md`.
