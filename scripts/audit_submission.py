"""Check Git-visible source for local paths, credentials and generated assets.

This is a deterministic submission hygiene check, not a complete secret scanner.
Only filenames and line numbers are reported; matched values are never printed.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'user-specific POSIX path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'root-specific deployment path': re.compile(r'/root/[A-Za-z0-9_.-]+/'),
    'user-specific Windows path': re.compile(r'[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s]+'),
    'private SSH endpoint': re.compile(r'\bssh\s+(?:-p\s+\d+\s+)?(?:root|[A-Za-z][\w-]*)@'),
    'GitHub access token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential-bearing URL': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
}
BINARY_ASSET = 'QD-DETR/run_on_video/clip/bpe_simple_vocab_16e6.txt.gz'


def main():
    output = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    paths = sorted(set(p for p in output.decode().split('\0') if p))
    issues = []
    for name in paths:
        path = ROOT / name
        if path.is_symlink():
            issues.append(f'{name}: symlink is not allowed in this portable submission')
            continue
        if any(p in {'node_modules', '.conda', '.venv', 'storage', 'outputs', 'results', 'data', 'features', 'dist'} for p in Path(name).parts):
            issues.append(f'{name}: generated/private asset directory')
        if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.pkl', '.npz', '.npy', '.mp4', '.mov', '.webm', '.docx', '.dotx', '.zip'}:
            issues.append(f'{name}: model/data/report archive')
        if path.stat().st_size > 2 * 1024 * 1024:
            issues.append(f'{name}: exceeds 2 MiB source-file limit')
        if name == BINARY_ASSET:
            continue
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            issues.append(f'{name}: unexpected binary file')
            continue
        for number, line in enumerate(lines, 1):
            for label, pattern in RULES.items():
                if pattern.search(line):
                    issues.append(f'{name}:{number}: {label}')
    if issues:
        print('\n'.join(issues))
        return 1
    print(f'Submission audit passed: {len(paths)} Git-visible files; no detected private paths/credentials or generated assets.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
