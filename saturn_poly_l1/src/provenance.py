#============
#Imports and module setup
#============
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import uuid
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
L1_ROOT = PROJECT_ROOT / 'saturn_poly_l1'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEDALUS_VERSION = '3.0.5'
CHECKPOINT_SCHEMA = 'saturn-shallow-water-state'

#============
#Calculate the SHA-256 digest
#============
def sha256(data):
    return hashlib.sha256(data).hexdigest()

#============
#Read installed dependency versions
#============
def package_versions():
    versions = {}
    for name in ('dedalus', 'numpy', 'scipy', 'xarray', 'matplotlib', 'mpi4py', 'h5py'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions

#============
#Write JSON atomically
#============
def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

#============
#Record code and configuration fingerprints
#============
def collect_provenance(cfg, config_bytes, arm):
    if json.loads(config_bytes) != cfg:
        raise ValueError('Configuration bytes differ from parsed configuration')
    paths = list((L1_ROOT / 'src').glob('*.py'))
    paths += list((L1_ROOT / 'scripts').glob('*.py')) + list((L1_ROOT / 'tests').glob('*.py'))
    paths += list((PROJECT_ROOT / 'saturn_poly').glob('*.py'))
    paths += list(PROJECT_ROOT.glob('requirements*.txt'))
    inputs = {str(p.relative_to(PROJECT_ROOT)): p.read_bytes() for p in sorted(set(paths))}
    hashes = {name: sha256(data) for name, data in inputs.items()}
    code_hashes = {name: hash_ for name, hash_ in hashes.items() if name.startswith('saturn_poly_l1/src/')}
    compatibility = {'checkpoint_schema': CHECKPOINT_SCHEMA, 'config_sha256': sha256(config_bytes), 'arm': arm, 'code_sha256': code_hashes, 'packages': package_versions(), 'python': sys.version, 'platform': platform.platform()}
    fingerprint = sha256(json.dumps(compatibility, sort_keys=True).encode())
    manifest = {'run_id': str(uuid.uuid4()), 'status': 'prepared', 'created_utc': datetime.now(timezone.utc).isoformat(), 'compatibility': compatibility, 'compatibility_sha256': fingerprint, 'input_sha256': hashes, 'python': sys.version, 'platform': platform.platform(), 'arm': arm, 'config_sha256': sha256(config_bytes), 'checkpoints': {}, 'scientific_readiness': 'not_validated', 'initial_state': cfg.get('verification_case', 'axisymmetric proxy jets unless explicitly restarted'), 'closure': 'unforced verification case with declared initialization' if cfg.get('verification_case') else 'experimental OU momentum forcing; C adds a prescribed anticyclonic trigger'}
    return (manifest, inputs)

#============
#Verify and restore checkpoint state
#============
def load_checkpoint(path, compatibility_hash, cfg):
    path = Path(path).resolve()
    parent = path.parent.parent
    manifest = json.loads((parent / 'manifest.json').read_text())
    if manifest.get('compatibility_sha256') != compatibility_hash:
        raise ValueError('Restart code/config/model/arm/package mismatch')
    relative = str(path.relative_to(parent))
    if manifest.get('checkpoints', {}).get(relative) != sha256(path.read_bytes()):
        raise ValueError('Checkpoint is unregistered or its checksum differs')
    with np.load(path, allow_pickle=False) as cp:
        if str(cp['schema']) != CHECKPOINT_SCHEMA or str(cp['compatibility_sha256']) != compatibility_hash:
            raise ValueError('Unsupported or incompatible checkpoint')
        state = {key: np.array(cp[key], copy=True) for key in ('u', 'h', 'forcing_coeff')}
        raw_step = cp['step'].item()
        if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 1:
            raise ValueError('Invalid checkpoint step')
        state['step'] = raw_step
        state['time_s'] = float(cp['time_s'])
        state['rng_state'] = json.loads(str(cp['rng_state']))
    n = cfg['numerics']
    f = cfg['forcing']
    shape = (n['nlon'], n['ncolat'])
    if state['u'].shape != (2, *shape) or state['h'].shape != shape or state['forcing_coeff'].shape != (f['mode_max'] - f['mode_min'] + 1, 2):
        raise ValueError('Checkpoint array shape mismatch')
    if any((not np.isfinite(state[k]).all() for k in ('u', 'h', 'forcing_coeff'))) or np.min(state['h']) <= 0:
        raise ValueError('Checkpoint contains invalid fields')
    if not np.isfinite(state['time_s']) or not np.isclose(state['time_s'], state['step'] * n['time_step_s'], rtol=1e-12, atol=1e-09):
        raise ValueError('Checkpoint step/time mismatch')
    rng = np.random.default_rng()
    rng.bit_generator.state = state['rng_state']
    state['parent'] = {'run_id': manifest['run_id'], 'checkpoint_path': str(path), 'checkpoint_sha256': sha256(path.read_bytes())}
    return state

#============
#Runrecord
#============
class RunRecord(AbstractContextManager):

    #============
    #Initialize object state
    #============
    def __init__(self, output, manifest, inputs, config_bytes, restart=None):
        self.output = Path(output)
        self.manifest = manifest
        self.inputs = inputs
        self.config_bytes = config_bytes
        if restart is not None:
            self.manifest['restart_parent'] = restart['parent']

    #============
    #Create the run directory and input records
    #============
    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            for folder in ('figures', 'states', 'checkpoints', 'inputs'):
                (self.output / folder).mkdir()
            (self.output / 'config.lock.json').write_bytes(self.config_bytes)
            for name, data in self.inputs.items():
                target = self.output / 'inputs' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            self.update(status='building')
        except BaseException as exc:
            self.update(status='failed', error=str(exc))
            raise
        return self

    #============
    #Update
    #============
    def update(self, **fields):
        self.manifest.update(fields)
        atomic_json(self.output / 'manifest.json', self.manifest)

    #============
    #Checkpoint
    #============
    def checkpoint(self, u, h, forcing, step, time_s):
        path = self.output / 'checkpoints' / f'step{step:08d}.npz'
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, u=u, h=h, forcing_coeff=forcing.coeff, rng_state=json.dumps(forcing.rng.bit_generator.state), step=step, time_s=time_s, schema=CHECKPOINT_SCHEMA, compatibility_sha256=self.manifest['compatibility_sha256'])
        temp.replace(path)
        self.manifest['checkpoints'][str(path.relative_to(self.output))] = sha256(path.read_bytes())
        self.update(last_checkpoint_step=step)

    #============
    #Record completion or interruption
    #============
    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.update(status='interrupted' if issubclass(exc_type, KeyboardInterrupt) else 'failed', error=str(exc))
        return False
