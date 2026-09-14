"""Optional installation diagnostic, never a generation or source-version gate."""
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[2]
required=('wgp.py','models/minimax_h3/minimax_h3_handler.py','shared/utils/plugins.py')
bad=[str(path) for path in required if not (root/path).is_file()]
print('H3 Latent Continue 0.3.2 - optional installation diagnostic')
print('Missing files: '+', '.join(bad) if bad else 'Wan2GP/H3 files found. No source hashes or versions checked.')
print('This diagnostic does not certify runtime compatibility and is not called during generation.')
raise SystemExit(1 if bad else 0)
