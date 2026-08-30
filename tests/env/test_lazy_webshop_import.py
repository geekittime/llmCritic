import subprocess
import sys


def test_environment_registry_does_not_eagerly_import_webshop_backend():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ragen.env; "
            "assert 'webshop_minimal' not in sys.modules; "
            "assert 'ragen.env.webshop.env' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
