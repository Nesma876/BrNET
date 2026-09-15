"""Setup-only workflow. Full campaign deliberately unavailable until approval."""
import subprocess
import sys
for module in ['scripts.audit_environment','scripts.model_summary','scripts.audit_data','scripts.audit_external','scripts.exact_overlap','scripts.setup_report']:
    subprocess.run([sys.executable,'-m',module],check=True)
subprocess.run([sys.executable,'-m','pytest','-q'],check=True)
